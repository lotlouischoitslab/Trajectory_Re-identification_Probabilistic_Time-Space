from __future__ import print_function
import torch
from model_six_maneuvers import highwayNet_six_maneuver
from utils_works_with_101_80_cnn_modified_passes_history_too_six_maneuvers import ngsimDataset,maskedNLL,maskedMSE,maskedNLLTest, maskedMSETest
from torch.utils.data import DataLoader
import time
import math
import pickle
from pathlib import Path  
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
 
from scipy.integrate import quad
from scipy.special import erf
from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d
   
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # FOR MULTI-GPU system using a single gpu
os.environ["CUDA_VISIBLE_DEVICES"]="1" # The GPU id to use, usually either "0" or "1"

########## Use this temporary but we need to fix the OpenBLAS error #########
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='.*openblas.*')
warnings.simplefilter('ignore', np.RankWarning)
##############################################################################

'''
2/18/2024 
Possible solution for the trajectory prediction part:
1. First we will calculate the line integral 
2. Create a function to detect overlaps using muX, muY, sigX and sigY
3. Taking into account of line integral cost and overlap probability, create cost matrix
4. For trajectories that have a high risk of overlapping, assign a high cost to discourage 
their simultaneous selection. Conversely, trajectories with high line integral costs and low 
overlap risks should have a lower match cost to encourage their selection.


OPTIMIZATION PART:
Objective: Minimize the total match cost across all trajectory pairs, selecting a set of trajectories that maximizes overall 
quality while minimizing collision risk.

Decision Variables: Define binary decision variables indicating whether a trajectory is selected for a vehicle.

Constraints: Each vehicle is assigned exactly one trajectory.
Trajectories with high overlap risks are not assigned simultaneously to different vehicles.




12/4/2023
TO-DO:
1. Start writing the paper. Write down what I want to include (detailed outline) 
Introduction, Methodology (sub-section allowed), Data (Description I-294), Results, Discussions, Conclusion
What we want to include for Introduction: 
Limitations, What this study does, Shortcomings from previous studies but what I want to address now
One paragraph with what people like to see (main contributions)
Finish one paragraph as how it is follows.

MUST-DO: Rest of the chapter, write a detailed outline. Think about what to include
Try learning Overleaf (LaTeX). Submit to TRR (Journal of TRB) or ASCE. ASCE is better option. 
Download the ASCE template journal for transportation. 

11/20/2023 
Take one trajectory, cut it to two pieces (from the point you cut it 5 seconds before), (then go forward and take 10 seconds)
call it left piece and right piece 
Use the right piece
plot the integral value 

Get the final data, cut out a piece of it
What we can do is I pick this section of the road (600 ft) and cut take about 20 ft
Simulate an overpass 
Have one trajectory for 600 ft, between 600 to 620 ft cut it to two pieces, you have before and after trajectory 
Run the code
Look at the middle of the data and revolve 20 ft from the point

Missing part is 600->620 ft part 
Go back 5 seconds from 600 ft. (Feed in to the prediction model)
Integral: Get the trajectories from 620 ft forward 5 seconds (This is the future) assuming there is an overpass 
Connect the trajectories  
Replace the 600->620ft part with the trajectory with 620 ft and 5 sec forward

Format of the output:
- 6 movements, each movement has probability distribution
- Actions are Straight, Accel, straight, decel, right, decel, left, decl 
- This is a possible sequence: Straight, Accel, Straight, Decel, Right, Decel, Left, Decl
- Every point one second during that 10 second horizon, we are getting normal distribution 2d where that car is probabilistically. For 10 seconds, we have 100 points, for every one of those 100 points, we have mean (x,y) and std (vx,vy). These are my outputs. Now, what we want to do is to get highest probability from one of the six movements. 
 
Guidelines to understand the prediction function: 
- There are 6 different maneuvers the car can pick 
- Each maneuver has 50 points
- Each point has probability distribution
- Take each maneuver ALL the 50 points. the corresponding point
- Take the line integral of that particular distrubtuion
- Do this for all 50 points 
- Sum them up 
- Then do this for all trajectories 
- Pick the manuever and the trajectory with the highest total value of the line integral
'''

############################################# LINE INTEGRAL CALCULATIONS #########################################
def line_integral(x1, y1, x2, y2, obj):
    #cost = 1e-5
    cost = []
    dx = x1 - x2
    dy = y1 - y2
    distance = math.sqrt(dx**2 + dy**2)

    for i in range(len(obj)):
        mu_x, mu_y, sigma_sq = obj[i][0],obj[i][1],obj[i][2]

        if sigma_sq < 1e-9:  # Avoid division by a very small number
            continue

        a = (dx**2 + dy**2) / (2 * sigma_sq)

        if a < 1e-9:  # Skip if 'a' is too small
            a = 1e-9 

        b = ((-2 * x1**2 + 2 * x1 * x2 + 2 * x1 * mu_x - 2 * x2 * mu_x) +
             (-2 * y1**2 + 2 * y1 * y2 + 2 * y1 * mu_y - 2 * y2 * mu_y)) / (2 * sigma_sq)
        
        c = ((x1 - mu_x)**2 + (y1 - mu_y)**2) / (2 * sigma_sq)

        exp_arg = ((b**2) / (4 * a)) - c
        exp_arg = max(min(exp_arg, 5), -5)  # Limit the range of the exponential argument
        exp_part = math.exp(exp_arg)
        sqrt_a = math.sqrt(max(a, 0.01))  # Ensure non-negative argument for sqrt
        erf_part = math.erf(sqrt_a + b / (2 * sqrt_a)) - math.erf(b / (2 * sqrt_a)) + 1e-4
        term_to_add = (exp_part / (2 * math.pi * sigma_sq)) * (1 / sqrt_a) * \
                (math.sqrt(math.pi) / 2) * erf_part * distance

        #print(f'term to add: {term_to_add}')
        cost.append(term_to_add)

    #print(f'cost: {len(cost)}')
    return cost

# The heatmap values on the right show the value of the normal distribution
# x and y have to be the prediction values. 
# Now let's plot the trajectories using x and y trajectories. Then bring into the starting point

def generate_normal_distribution(fut_pred, lane, predicted_traj,batch_num):
    num_maneuvers = len(fut_pred)
    x = np.linspace(-100,100,100)  
    y = np.linspace(0,100,100)  
    Xc, Yc = np.meshgrid(x, y)
    combined_Z = np.zeros(Xc.shape)
    plt.figure(figsize=(18, 12)) 

    for m in range(num_maneuvers):
        print(f"Processing maneuver {m+1}/{num_maneuvers}")
        muX = fut_pred[m][:, batch_num, 0]
        muY = fut_pred[m][:, batch_num, 1]
        sigX = fut_pred[m][:, batch_num, 2]
        sigY = fut_pred[m][:, batch_num, 3]
      
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape) # Initialize a zero matrix for the PDF values

        # Calculate the PDF values for each point on the grid
        for i in range(len(muX)):
            mean = [muX[i], muY[i]]
            cov = [[sigX[i]**2, 0], [0, sigY[i]**2]]  # Assuming no covariance
            rv = multivariate_normal(mean, cov)
            Z += rv.pdf(np.dstack((X, Y)))
        
        combined_Z += Z

        # Plot the contour map
        # plt.figure(figsize=(9, 6))
        plt.subplot(2,3,m+1)
        contour = plt.contourf(X, Y, Z, cmap='viridis')
        plt.xlabel('X - Lateral Coordinate')
        plt.ylabel('Y - Longitudinal Coordinate')
        plt.title(f'Contour Plot for Maneuver {m+1}')
        plt.colorbar(contour)
        # plt.savefig('plots/maneuver'+str(m+1)+'.png')
    
    # Plot the combined contour map for all maneuvers
    plt.tight_layout()
    plt.savefig('plots/all_maneuvers_subplot.png')
    plt.figure(figsize=(9, 6))
    combined_contour = plt.contourf(Xc, Yc, combined_Z, cmap='viridis')
    plt.xlabel('X - Lateral Coordinate')
    plt.ylabel('Y - Longitudinal Coordinate')
    plt.title('Combined Contour Plot for All Maneuvers')
    plt.colorbar(combined_contour)
    plt.savefig('plots/combined_maneuver.png')
        

def create_object(muX, muY, sigX, sigY): # Helper function to create an object of muX, muY, sigX, sigY 
    # Ensure that the tensors do not require gradients before converting to numpy
    muX_numpy = muX.detach().numpy() if isinstance(muX, torch.Tensor) else muX
    muY_numpy = muY.detach().numpy() if isinstance(muY, torch.Tensor) else muY
    sigX_numpy = sigX.detach().numpy() if isinstance(sigX, torch.Tensor) else sigX
    sigY_numpy = sigY.detach().numpy() if isinstance(sigY, torch.Tensor) else sigY
    result =  np.column_stack([muX_numpy, muY_numpy, (sigX_numpy-sigY_numpy)**2])
    # print(result) # print the results 
    return result # return the result created by stacking up the statistical variables. 


# NOTE: I need to figure out an optimization algorithm to put here
# TBD with Professor Talebpour (to be negotiated)
 

def predict_trajectories(input_data, overpass_start_loc,overpass_end_loc, lane, fut_pred, batch_num): # predict trajectory function 
    # NOTE: For now, I will ignore current_point and overpass_start variables
    num_maneuvers = len(fut_pred) # We have 6 different maneuvers 
    # print(num_maneuvers)

    input_data = input_data[input_data['lane'] == lane].reset_index(drop=True) # we want to pick for that lane given (this has ALL the trajectories)
    possible_trajectories = input_data[input_data['xloc'] >= overpass_end_loc] # the possible set of trajectories can be pass the overpass location
    IDs_to_traverse = possible_trajectories['ID'].unique() # get all the unique IDs 
    # print(IDs_to_traverse)

    ######################### Initialize storage for all trajectories and the best trajectory #################################
    trajectories = [] # final set of trajectories that we would have traversed 
    best_trajectory = None # initialize the best trajectory as none first 
    highest_integral_value = float('-inf') # assign a really large negative value 
    tol = 0.1  # set a tolerance value 
    start_time_data = input_data[(abs(input_data['xloc'] - overpass_start_loc) <=tol) & (input_data['xloc'] >= overpass_start_loc)] # overpass start time 
    # print(start_time_data)

    start_time = min(start_time_data['time']) # go where the overpass starts and get that specific time
    end_time = start_time + 5 # we are going to check for 5 seconds from start time
  
    print('start time',start_time) 
    print('end time',end_time) 
    # print(possible_trajectories)
    ###########################################################################################################################
    
    for temp_ID in IDs_to_traverse: # for each trajectory ID 
        # Initialize storage for the current trajectory
        current_trajectory = {
            'ID':temp_ID,
            'lane':lane,
            'xloc':[],
            'yloc':[],
            'maneuver':[],
            'muX':[],
            'muY':[],
            'sigX':[],
            'sigY':[],
            'line_integral_values': []
        }

        current_data = possible_trajectories[possible_trajectories['ID'] == temp_ID] # extract the current trajectory data
        current_data = current_data[current_data['time'] <= end_time] # make sure it is given within the boundaries 
        
        if len(current_data) != 0: # we don't want empty trajectories 
            # print('current')
            # print(current_data)
            for i in range(len(current_data) - 1): # Loop through each segment in current_data
                x1, y1 = current_data.iloc[i][['xloc', 'yloc']] # get the (x1,y1) coordinates
                x2, y2 = current_data.iloc[i + 1][['xloc', 'yloc']] # get the (x2,y2) coordinates
                for m in range(num_maneuvers): # Loop through each maneuver
                    # print('check ID',temp_ID)
                    muX, muY, sigX, sigY = fut_pred[m][:, batch_num, :4].T # Extract maneuver-specific predictive parameters
                    obj_for_integral = create_object(muX, muY, sigX, sigY) # get the probabilistic parameters
                    segment_integral = line_integral(x1, y1, x2, y2, obj_for_integral) # Calculate line integral for each segment (return 50 values)
                    current_trajectory['xloc'].append((x1,x2)) # this is individual (x1,x2)
                    current_trajectory['yloc'].append((y1,y2)) # this is individual (y1,y2)
                    current_trajectory['muX'].append(muX) # this has 50 points
                    current_trajectory['muY'].append(muY) # this has 50 points
                    current_trajectory['sigX'].append(sigX) # this has 50 points
                    current_trajectory['sigY'].append(sigY) # this has 50 points
                    current_trajectory['line_integral_values'].append(segment_integral) # 50 line integral values will be appended
                    current_trajectory['maneuver'].append(m+1) # append the maneuver
                    
                    for seg_int in segment_integral: # for each calculated line integral value
                        if seg_int > highest_integral_value: # check if the selected line integral value is greater than or not
                            highest_integral_value = seg_int # assign the highest line integral value
                            best_trajectory = current_trajectory # assign the current trajectory to the best trajectory 
                
            trajectories.append(current_trajectory) # Store the current trajectory
    
    for key,temp in enumerate(trajectories): # for each stored dataframe
        trajectories_df = pd.DataFrame(temp) # convert to DataFrame
        trajectories_df.to_csv('all_combinations_trajectories/batch_'+str(batch_num)+'_trajectory_combo_'+str(key+1)+'.csv', index=False) # Save to CSV
    
    if best_trajectory: # if we have the best trajectory
        best_trajectory_df = pd.DataFrame(best_trajectory) # convert the best trajectory data into dataframe format 
        best_trajectory_df.to_csv('batch_'+str(batch_num)+'_best_trajectory.csv', index=False) # then convert to csv
    
    return trajectories, best_trajectory # return all the trajectories traversed and the best trajectory 


def plot_trajectory(lane, smoothed_file, modified_data): # Function to plot the trajectories 
    print(type(smoothed_file), type(modified_data))
    lane_data = smoothed_file[smoothed_file['lane'] == lane].reset_index(drop=True) # extract the lane data 
    modified_lane_data = modified_data[modified_data['lane'] == lane].reset_index(drop=True) # extract the lane data 
    fig, ax = plt.subplots()
    temp_data = lane_data 
    ts = temp_data['time'].to_numpy()
    ys = temp_data['xloc'].to_numpy()
    ax.plot(ts, ys, color='blue', linewidth=2, alpha=0.7, label='Original') # Plot original trajectory

    md = modified_lane_data 
    mod_ts = md['time'].to_numpy()
    mod_ys = md['xloc'].to_numpy()

    ax.plot(mod_ts, mod_ys, color='red', linewidth=2, alpha=0.7, label='Predicted')
    ax.set_xlabel('Time (s)', fontsize=20)
    ax.set_ylabel('Location (m)', fontsize=20)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.set_size_inches(60, 30)
    fig.savefig(f'plots/Louis_Lane_temp_{lane}-x.png', dpi=300)  # Adjust the DPI for better resolution

 
def main(): # Main function 
    args = {} # Network Arguments
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        args['use_cuda'] = True 
    else:
        args['use_cuda'] = False 
    
    print(f'My device: {device}')
    args['encoder_size'] = 64
    args['decoder_size'] = 128
    args['in_length'] = 16
    #args['out_length'] = 25
    args['out_length'] = 50
    args['grid_size'] = (13,3)
    args['soc_conv_depth'] = 64
    args['conv_3x1_depth'] = 16
    args['dyn_embedding_size'] = 32
    args['input_embedding_size'] = 32
    args['num_lat_classes'] = 3
    args['num_lon_classes'] = 2
    args['use_maneuvers'] = True
    args['train_flag'] = False

    ######################################## TRAJECTORY DIRECTORIES ######################################################################################
    trajectories_directory = '/Users/louis/cee497projects/data/101-80-speed-maneuver-for-GT/train/10_seconds/' # Local Machine
    # trajectories_directory = 'cee497projects/data/101-80-speed-maneuver-for-GT/train/10_seconds/' # HAL GPU Cluster

    ####################################### MODEL DIRECTORIES ############################################################################################
    directory = '/Users/louis/cee497projects/trajectory-prediction/codes/predicted_environment/' # Local Machine
    # directory = 'cee497projects/trajectory-prediction/codes/predicted_environment/'  # HAL GPU Cluster

    model_directory = 'models/trained_models_10_sec/cslstm_m.tar'
    saving_directory = 'predicted_data/highwaynet-10-sec-101-80-speed-maneuver-for-GT-six-maneuvers/'
    
    ######################################### PRED SET DIRECTORY #########################################################################################
    filepath_pred_Set = '/Users/louis/cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10-seconds/test' # Local Machine
    # filepath_pred_Set = 'cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10-seconds/test' # HAL GPU Cluster
    
    ######################################################################################################################################################
    file_to_read = 'lane_2_data.csv'
    temp_lane = 2
    df = pd.read_csv(file_to_read) # read in the data 
    # temp_ID = 4106
    # df = df[df['ID'] == temp_ID]
    # plt.xlabel('time (seconds)') 
    # plt.ylabel('xloc')
    # plt.plot(df['time'],df['xloc'])
    # plt.savefig('temp_plot_lane_'  + str(temp_lane) +'.png')

    original_data = df.copy() # copy the dataframe
    print(df.keys()) # print the keys just in case 

    #lanes_to_analyze = sorted(df['lane'].unique())[1:-1] # lanes to analyze 
    lanes_to_analyze = [temp_lane] # lanes to analyze 
    print(f'Unique lanes: {lanes_to_analyze}') 
    
    batch_size = 512 # batch size for the model and choose from [1,2,4,8,16,32,64,128,256,512,1024,2048]

    ################################## OVERPASS LOCATION (ASSUMPTION) ########################################################################
    overpass_start_loc,overpass_end_loc = 160, 180 # both in meters 

    ################################# NEURAL NETWORK INITIALIZATION ######################################################## 
    net = highwayNet_six_maneuver(args) # we are going to initialize the network 
    full_path = os.path.join(directory, model_directory) # create a full path 
    net.load_state_dict(torch.load(full_path, map_location=torch.device(device))) # load the model onto the local machine 

    ################################# CHECK GPU AVAILABILITY ###############################################################
    if args['use_cuda']: 
        net = net.cuda()
    #########################################################################################################################

    ################################# INITIALIZE DATA LOADERS ################################################################
    predSet = ngsimDataset(filepath_pred_Set, t_h=30, t_f=100, d_s=2)
    predDataloader = DataLoader(predSet,batch_size=batch_size,shuffle=True,num_workers=3,collate_fn=predSet.collate_fn)
    lossVals = torch.zeros(50).to(device) # Louis code
    counts = torch.zeros(50).to(device) # Louis code

    ################################## SAVING DATA ##############################################################################
    fut_predictions = [] # future prediction values 
    maneuver_predictions = [] # maneuver prediction values 

    ################################## OUTPUT DATA ##############################################################################
    print(f'Length of the pred data loader: {len(predDataloader)}') # this prints out 1660040 

    ################################## LANES TO BE ANALYZED #####################################################################################
    predicted_traj = None # we are going to store the predicted trajectories 
    for lane in lanes_to_analyze: # for each lane to be analyzed 
        print(f'Lane: {lane}') # print the lane  
 
        for i, data  in enumerate(predDataloader): # for each index and data in the predicted data loader 
            # if i >= 1:
            #     break 
            print(f'Index of Data: {i}/{len(predDataloader)}') # just for testing, print out the index of the current data to be analyzed 
            
            st_time = time.time() # start the timer 
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, points, maneuver_enc  = data # unpack the data   
    
            if args['use_cuda']:
                hist = hist.cuda()
                nbrs = nbrs.cuda()
                mask = mask.cuda()
                lat_enc = lat_enc.cuda()
                lon_enc = lon_enc.cuda()
                fut = fut.cuda()
                op_mask = op_mask.cuda()
                maneuver_enc = maneuver_enc.cuda()

            fut_pred, maneuver_pred = net(hist, nbrs, mask, lat_enc, lon_enc) # feed the parameters into the neural network for forward pass
            fut_pred_max = torch.zeros_like(fut_pred[0]) # get the max predicted values 

            for k in range(maneuver_pred.shape[0]): # for each value in the maneuver predicted shapes
                indx = torch.argmax(maneuver_pred[k, :]).detach() # get the arg max of the maneuvered prediction values
                fut_pred_max[:, k, :] = fut_pred[indx][:, k, :] # future predicted value max 

            l, c = maskedMSETest(fut_pred_max, fut, op_mask) # get the loss value and the count value 
            lossVals += l.detach() # increment the loss value 
            counts += c.detach() # increment the count value 
            points_np = points.numpy() # convert to numpy arrays 
            fut_pred_np = [] # store the future pred points 

            for k in range(6): #manuevers mean the 
                fut_pred_np_point = fut_pred[k].clone().detach().cpu().numpy()
                fut_pred_np.append(fut_pred_np_point)

            fut_pred_np = np.array(fut_pred_np) # convert the fut pred points into numpy
            trajectory,predicted_traj = predict_trajectories(original_data, overpass_start_loc,overpass_end_loc,lane,fut_pred_np,i) # where the function is called and I feed in maneurver pred and future prediction points         
            
            if i == 0: # Generate and save the distribution plots just for one trajectory
                generate_normal_distribution(fut_pred_np, lane, predicted_traj,i)
                break

        # print('Predicted')
        # print(f"{len(predicted_traj['lane'])} | {len(predicted_traj['time'])} | {len(predicted_traj['xloc'])} | {len(predicted_traj['yloc'])}")
        # print('Original Dataframe')
        # print(f"{len(df['lane'])} | {len(df['time'])} | {len(df['xloc'])} | {len(df['yloc'])}")
        predicted_traj = pd.DataFrame(predicted_traj) # convert the predicted traj into Pandas DataFrame
        #plot_trajectory(lane, df, predicted_traj) # plot the predicted trajectories


if __name__ == '__main__': # run the code
    main() # call the main function 



