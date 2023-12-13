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
    cost = 1e-5
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
        #print(f"exp_part: {exp_part}, sigma_sq: {sigma_sq}, sqrt_a: {sqrt_a}, erf_part: {erf_part}, distance: {distance}")

        term_to_add = (exp_part / (2 * math.pi * sigma_sq)) * (1 / sqrt_a) * \
                (math.sqrt(math.pi) / 2) * erf_part * distance

        # print(f'term to add: {term_to_add}')
        cost += term_to_add # add this term to the cost

    print(f'cost: {cost}')
    return cost # return the cost value


def temp_plot(files):
    count = 0
    for f in files:
        t = f['time']
        x = f['xloc']
        
        plt.figure()  # Start a new figure
        plt.plot(t, x)  # Plot the data
        plt.title('Time vs X Location')  # Optional: Add a title to the plot
        plt.xlabel('Time')  # Optional: Add a label to the x-axis
        plt.ylabel('X Location')  # Optional: Add a label to the y-axis
        plt.grid(True)  # Optional: Add a grid to the plot for better readability
        plot_filename = 'plot{}.png'.format(count)
        plt.savefig(plot_filename)  # Save the current figure
        plt.close()  # Close the figure to free memory
        
        # Create a DataFrame and save it as CSV
        temp = {
            'time': t,
            'xloc': x
        }
        temp_df = pd.DataFrame(temp)
        csv_filename = 'data{}.csv'.format(count)
        temp_df.to_csv(csv_filename, index=False)  # Save the dataframe to a CSV without the index
        count += 1  # Increment the counter for the next plot


def create_object(muX, muY, sigX, sigY): # Helper function to create an object of muX, muY, sigX, sigY 
    # Ensure that the tensors do not require gradients before converting to numpy
    muX_numpy = muX.detach().numpy() if isinstance(muX, torch.Tensor) else muX
    muY_numpy = muY.detach().numpy() if isinstance(muY, torch.Tensor) else muY
    sigX_numpy = sigX.detach().numpy() if isinstance(sigX, torch.Tensor) else sigX
    sigY_numpy = sigY.detach().numpy() if isinstance(sigY, torch.Tensor) else sigY
    result =  np.column_stack([muX_numpy, muY_numpy, (sigX_numpy-sigY_numpy)**2])
    # print(result)
    return result 

 
def predict_trajectories(input_data, overpass_start,overpass_end,lane,fut_pred,count=0): # function to predict trajectories
    num_maneuvers = len(fut_pred) # This would be 6 because we have 6 possible maneuvers 
    input_data = input_data[input_data['lane']==lane].reset_index(drop=True)
    IDs = [] # get all the IDs
    init_ID = -1 # default init_ID value 

    for i in range(len(input_data)): # get all vehicle IDs
        if input_data['ID'][i] != init_ID:
            IDs.append(input_data['ID'][i])
            init_ID = input_data['ID'][i]

    overpass_data = input_data[(input_data['xloc'] >= overpass_start) & (input_data['xloc'] <= overpass_end)] # the overpass section that needs to be analyzed 
    overpass_data = overpass_data[overpass_data['lane'] == lane].reset_index(drop=True)  # Adjust for the specific lane I am analyzing
    min_time = np.min(overpass_data['time'].values) # input minimum time
    max_time = np.max(overpass_data['time'].values) # input maximum time 
    current_time = max_time # assign the current time as the max time
    end_time = current_time + 5  # 5 seconds later

    print(f'min max time: {min_time} | {max_time}') 

    best_trajectory = {
        'ID':[],
        'lane':[],
        'time':[],
        'xloc':[],
        'yloc':[]
    } # Placeholder for the best trajectory's x and y values

    files = {
        'time':[],
        'xloc':[]
    }

    files_saved = []
    highest_integral_value = float('-inf')  # Initialize with a very small number
    # print(f'IDs: {IDs}')
    #IDs = [3799]  # temporarily 
    
    for j in IDs: # for each ID
        temp_data = input_data[input_data['ID']==j] # temp data for filtering
        filtered_data = temp_data[(temp_data['time'] >= current_time) & (temp_data['time'] <= end_time)] # filter the data (cutting it to two)
    
        # print(f'min max time: {min_time} | {max_time}') 
        temp_time = [] # this is the time we want to store
        temp_x = [] # this is the best x trajectory for that ID 
        temp_y = [] # this is the best y trajectory for that ID 
        filtered_data_length = len(filtered_data['xloc']) 
        print(f'filter length: {filtered_data_length}')


        if filtered_data_length > 0:  # added this condition because I do not want to deal with empty arrays 
            for m in range(num_maneuvers): # for each of the 6 maneuvers
                objects_for_integral = create_object(fut_pred[m][:, :, 0], fut_pred[m][:, :, 1], fut_pred[m][:, :, 2], fut_pred[m][:, :, 3]) # get the muX, muY, sigX, sigY values
                x_temp_trajectory = filtered_data['xloc'].values # x trajectory values
                y_temp_trajectory = filtered_data['yloc'].values # y trajectory values 
                total_integral_for_trajectory = 0 # line integral summation for that particular trajectory 

                for i in range(len(x_temp_trajectory)-1): # for each trajectory coordinates
                    x1 = x_temp_trajectory[i] 
                    y1 = y_temp_trajectory[i] 
                    x2 = x_temp_trajectory[i+1]
                    y2 = y_temp_trajectory[i+1]
                    total_integral_for_trajectory += line_integral(x1,y1,x2,y2,objects_for_integral)
                
                print(f'time range: {current_time} | {end_time}')
                min_max_series = np.linspace(current_time,end_time,filtered_data_length) # split the time evenly   
                # print(f'min max series: {min_max_series}')   
                files = {}  # Initialize a new dictionary for this set of data
                files['time'] = min_max_series
                files['xloc'] = x_temp_trajectory  # assign the best trajectories for x
                files_saved.append(files)  # Save this set of data

                print('current integral value',total_integral_for_trajectory) # current integral value
            
                if total_integral_for_trajectory > highest_integral_value: # Check if this trajectory has the highest integral value so far
                    highest_integral_value = total_integral_for_trajectory # update the highest integral value
                    # print(f'Integral value: {total_integral_for_trajectory}') # check if integral value is updated or not
                    temp_time = min_max_series # this is the time variable temporarily assigned
                    temp_x = x_temp_trajectory # this x trajectory is temporarily assigned
                    temp_y = y_temp_trajectory # this y trajectory is temporarily assigned

            # print(temp_time) 
            # print(temp_x)
            # print(temp_y)
            temp_ID = [j for i in range(len(temp_x))] # assign the ID
            temp_lane = [lane for i in range(len(temp_x))] # assign the lane
            best_trajectory['ID'].extend(temp_ID) # assign the IDs 
            best_trajectory['lane'].extend(temp_lane) # assign the lanes
            best_trajectory['time'].extend(temp_time) # the time series plot we need to assign 
            best_trajectory['xloc'].extend(temp_x) # assign the best trajectories for x
            best_trajectory['yloc'].extend(temp_y) # assign the best trajectories for y
          
        
    print(f"assertions: {len(best_trajectory['time'])} | {len(best_trajectory['xloc'])} | {len(best_trajectory['yloc'])}")
    return best_trajectory,files_saved # return the best trajectory dictionary  


def plot_trajectory(lane, smoothed_file, modified_data): # Function to plot the trajectories 
    print(type(smoothed_file), type(modified_data))
    lane_data = smoothed_file[smoothed_file['lane'] == lane].reset_index(drop=True) # extract the lane data 
    modified_lane_data = modified_data[modified_data['lane'] == lane].reset_index(drop=True) # extract the lane data 
    IDs = lane_data['ID'].unique().tolist()  # More efficient way to get unique IDs
    fig, ax = plt.subplots()

    for j in IDs: # for each vehicle ID 
        print(f'j: {j}')
        temp_data = lane_data[lane_data['ID'] == j]
        ts = temp_data['time'].to_numpy()
        ys = temp_data['xloc'].to_numpy()
        ax.plot(ts, ys, color='blue', linewidth=2, alpha=0.7, label='Original' if j == IDs[0] else "") # Plot original trajectory

 
    for k in IDs:
        md = modified_lane_data[modified_lane_data['ID'] == k]  # Plot modified trajectory
        mod_ts = md['time'].to_numpy()
        mod_ys = md['xloc'].to_numpy()
        ax.plot(mod_ts, mod_ys, color='red', linewidth=2, alpha=0.7, label='Predicted' if k == IDs[0] else "")

    
    ax.set_xlabel('Time (s)', fontsize=20)
    ax.set_ylabel('Location (m)', fontsize=20)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.set_size_inches(90, 30)
    fig.savefig(f'Louis_Lane_{lane}-x.png', dpi=300)  # Adjust the DPI for better resolution

 
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
    #trajectories_directory = 'cee497projects/data/101-80-speed-maneuver-for-GT/train/10_seconds/' # HAL GPU Cluster

    ####################################### MODEL DIRECTORIES ############################################################################################
    directory = '/Users/louis/cee497projects/trajectory-prediction/codes/predicted_environment/' # Local Machine
    #directory = 'cee497projects/trajectory-prediction/codes/predicted_environment/'  # HAL GPU Cluster

    model_directory = 'models/trained_models_10_sec/cslstm_m.tar'
    saving_directory = 'predicted_data/highwaynet-10-sec-101-80-speed-maneuver-for-GT-six-maneuvers/'
    
    ######################################### PRED SET DIRECTORY #########################################################################################
    filepath_pred_Set = '/Users/louis/cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10-seconds/test' # Local Machine
    #filepath_pred_Set = 'cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10-seconds/test' # HAL GPU Cluster
    
    ######################################################################################################################################################
    df = pd.read_csv('Run_1_final_rounded.csv') # read in the data 
    original_data = df.copy() # copy the dataframe
    print(df.keys()) # print the keys just in case 

    lanes_to_analyze = sorted(df['lane'].unique())[1:-1] # lanes to analyze 
    print(f'Unique lanes: {lanes_to_analyze}') 
     
    output_results = {key:[] for key in lanes_to_analyze} # output trajectories 
    batch_size = 512 # batch size for the model and choose from [1,2,4,8,16,32,64,128,256,512,1024,2048]
    temp_stop = 5 # index where we want to stop the simulation

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

    
    ################################## OVERPASS LOCATION (ASSUMPTION) ########################################################################
    overpass_start = 180 # overpass start location in feets
    overpass_end = 200 # overpass end location in feets

    ################################## LANES TO BE ANALYZED #####################################################################################
    predicted_traj = None # we are going to store the predicted trajectories 
    for lane in lanes_to_analyze: # for each lane to be analyzed 
        print(f'Lane: {lane}') # print the lane  
        count = 0
        for i, data  in enumerate(predDataloader): # for each index and data in the predicted data loader 
            print(f'Index of Data: {i}') # just for testing, print out the index of the current data to be analyzed 

            if i == temp_stop:
                break
            
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
            predicted_traj,files = predict_trajectories(original_data, overpass_start,overpass_end,lane,fut_pred_np,count) # where the function is called and I feed in maneurver pred and future prediction points         
            count += 1
            
        # temp_plot(files)
        
        print('Predicted')
        print(f"{len(predicted_traj['ID'])} | {len(predicted_traj['lane'])} | {len(predicted_traj['time'])} | {len(predicted_traj['xloc'])} | {len(predicted_traj['yloc'])}")
        print('Original Dataframe')
        print(f"{len(df['ID'])} | {len(df['lane'])} | {len(df['time'])} | {len(df['xloc'])} | {len(df['yloc'])}")

        predicted_traj = pd.DataFrame(predicted_traj) # convert the predicted traj into Pandas DataFrame
        plot_trajectory(lane, df, predicted_traj) # plot the predicted trajectories
 
    
    # with open(directory+saving_directory+"output_results.data", "wb") as filehandle:
    #     pickle.dump(np.array(output_results), filehandle, protocol=4)

if __name__ == '__main__': # run the code
    main() # call the main function 