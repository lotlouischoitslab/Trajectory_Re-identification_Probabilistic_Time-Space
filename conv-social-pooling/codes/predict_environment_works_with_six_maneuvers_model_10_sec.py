from __future__ import print_function
import torch
from model_six_maneuvers import highwayNet_six_maneuver
from utils_works_with_101_80_cnn_modified_passes_history_too_six_maneuvers import ngsimDataset,maskedNLL,maskedMSE,maskedNLLTest, maskedMSETest
from torch.utils.data import DataLoader
import time
import math
import pickle
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
Get the final data, cut out a piece of it
What we can do is I pick this section of the road (200 ft) and cut take about 20 ft
Simulate an overpass 
Have one trajectory for 200 ft, between 100 to 120 ft cut it to two pieces, you have before and after trajectory 
Run the code
Look at the middle of the data and revolve 20 ft from the point
Revolve 20 ft

Missing part is 100->120 ft part 
Go back 5 seconds from 100 ft. (Feed in to the prediction model)
Integral: Get the trajectories from 120 ft forward 5 seconds (This is the future) assuming there is an overpass 
Connect the trajectories  
Replace the 100->120ft part with the trajectory with 120 ft and 5 sec forward



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
    # Compute square distances
    x1_x2_sq = np.square(x1 - x2)
    y1_y2_sq = np.square(y1 - y2)
    
    denom = 2 * obj[:, 2]
    denom = np.clip(denom, 1e-9, np.inf)  # Avoid division by zero

    a = (x1_x2_sq + y1_y2_sq) / denom
    b = (-2 * x1 + 2 * x2 + 2 * obj[:, 0] - 2 * x2 * obj[:, 0] + 
         -2 * y1 + 2 * y2 + 2 * obj[:, 1] - 2 * y2 * obj[:, 1]) / denom
    c = (np.square(x1 - obj[:, 0]) + np.square(y1 - obj[:, 1])) / denom

    # Mask where 'a' is too close to zero
    mask = np.abs(a) >= 1e-9
    
    # Apply the mask
    a_masked = a[mask]
    b_masked = b[mask]
    c_masked = c[mask]
    obj_masked = obj[mask]

    # Return if empty after masking
    if a_masked.size == 0:
        return 1e-9

    denom = 2 * np.sqrt(a_masked)
    denom = np.clip(denom, 1e-9, np.inf)  # Avoid division by zero

    exp_arg = np.clip((b_masked * b_masked) / (4 * a_masked) - c_masked, -20, 20)

    cost = (np.exp(exp_arg) / (2 * np.pi * obj_masked[:, 2])) * \
           (1 / np.sqrt(a_masked)) * (np.sqrt(np.pi) / 2) * \
           (erf(np.sqrt(a_masked) + b_masked / denom) - erf(b_masked / denom)) * \
           np.sqrt(x1_x2_sq + y1_y2_sq)

    # Normalize the cost using Z-score normalization
    std_cost = np.std(cost) + 1e-9
    mean_cost = np.mean(cost)
    normalized_cost = (cost - mean_cost) / std_cost
    return normalized_cost.sum()


def create_object(muX, muY, sigX, sigY): # Helper function to create an object of muX, muY, sigX, sigY 
    # Ensure that the tensors do not require gradients before converting to numpy
    muX_numpy = muX.detach().numpy() if isinstance(muX, torch.Tensor) else muX
    muY_numpy = muY.detach().numpy() if isinstance(muY, torch.Tensor) else muY
    sigX_numpy = sigX.detach().numpy() if isinstance(sigX, torch.Tensor) else sigX
    sigY_numpy = sigY.detach().numpy() if isinstance(sigY, torch.Tensor) else sigY
    return np.column_stack([muX_numpy, muY_numpy, (sigX_numpy-sigY_numpy)**2])

 
def predict_trajectories(input_data, overpass,lane,fut_pred): # function to predict trajectories
    num_maneuvers = len(fut_pred) # This would be 6 because we have 6 possible maneuvers 
    input_data = input_data[input_data['lane']==lane].reset_index(drop=True)
    IDs = [] # get all the IDs
    init_ID = -1

    # print('input data',input_data,input_data.keys())

    # get all vehicle IDs
    for i in range(len(input_data)):
        if input_data['ID'][i] != init_ID:
            IDs.append(input_data['ID'][i])
            init_ID = input_data['ID'][i]

    delta = 50 
    input_data = input_data[(input_data['xloc'] >= overpass-delta) & (input_data['xloc'] <= overpass+delta)] # the overpass section that needs to be analyzed 
    
    input_data = input_data[input_data['lane'] == lane].reset_index(drop=True)  # Adjust for the specific lane I am analyzing
    
    min_time = input_data['time'].min()
    max_time = input_data['time'].max()

    print(f'min max: {min_time} | {max_time}')


    best_trajectory = {
        'lane':lane,
        'time':[],
        'xloc':[],
        'yloc':[],
        'Cost':[]
    } # Placeholder for the best trajectory's x and y values

    highest_integral_value = float('-inf')  # Initialize with a very small number
    
    for j in IDs:
        temp_data = input_data[input_data['ID']==j] 
        #print('temp data',temp_data.keys())
         
        for m in range(num_maneuvers): # for each of the 6 maneuvers
            objects_for_integral = create_object(fut_pred[m][:, :, 0], fut_pred[m][:, :, 1], fut_pred[m][:, :, 2], fut_pred[m][:, :, 3]) # get the muX, muY, sigX, sigY values
            x_temp_trajectory = temp_data['xloc'].values # x trajectory values
            y_temp_trajectory = temp_data['yloc'].values # y trajectory values 
            total_integral_for_trajectory = 0 # line integral summation for that particular trajectory 

            for i in range(len(x_temp_trajectory)-1):
                x1 = x_temp_trajectory[i] 
                y1 = y_temp_trajectory[i] 
                x2 = x_temp_trajectory[i+1]
                y2 = y_temp_trajectory[i+1]
                total_integral_for_trajectory += line_integral(x1,y1,x2,y2,objects_for_integral)

             
            if total_integral_for_trajectory > highest_integral_value: # Check if this trajectory has the highest integral value so far
                highest_integral_value = total_integral_for_trajectory # update the highest integral value
                length_of_time_data = len(x_temp_trajectory) # Get the length of the x trajectory
              
                min_max_series = np.linspace(min_time,max_time,length_of_time_data) # split the time evenly 
                
                best_trajectory['time'] = min_max_series # the time series plot we need to assign 
                best_trajectory['xloc'] = x_temp_trajectory# assign the best trajectories for x
                best_trajectory['yloc'] = y_temp_trajectory # assign the best trajectories for y
                best_trajectory['Cost'] = highest_integral_value # assign the highest_integral_value

    print(best_trajectory['Cost'])
    print(f"assertions: {len(best_trajectory['time'])} | {len(best_trajectory['xloc'])} | {len(best_trajectory['yloc'])}")
    return best_trajectory # return the best trajectory dictionary  


def plot_trajectory(lane, smoothed_file, modified_data): # Function to plot the trajectories 
    lane_data = smoothed_file[smoothed_file['lane'] == lane].reset_index(drop=True) # extract the lane data 
    IDs = lane_data['ID'].unique().tolist()  # More efficient way to get unique IDs
    fig, ax = plt.subplots()

    for j in IDs: # for each vehicle ID 
        temp_data = lane_data[lane_data['ID'] == j]
        ts = temp_data['time'].to_numpy()
        ys = temp_data['xloc'].to_numpy()
        ax.plot(ts, ys, color='blue', linewidth=2, alpha=0.7, label='Original' if j == IDs[0] else "") # Plot original trajectory

        for md in modified_data:  # Plot modified trajectory
            mod_ts = np.array(md['time'])
            mod_ys = np.array(md['xloc'])
            ax.plot(mod_ts, mod_ys, color='red', linewidth=2, alpha=0.7, label='Predicted' if j == IDs[0] else "")


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
    trajectories_directory = 'cee497projects/data/101-80-speed-maneuver-for-GT/train/10_seconds/' # HAL GPU Cluster

    ####################################### MODEL DIRECTORIES ############################################################################################
    directory = '/Users/louis/cee497projects/trajectory-prediction/codes/predicted_environment/' # Local Machine
    directory = 'cee497projects/trajectory-prediction/codes/predicted_environment/'  # HAL GPU Cluster

    model_directory = 'models/trained_models_10_sec/cslstm_m.tar'
    saving_directory = 'predicted_data/highwaynet-10-sec-101-80-speed-maneuver-for-GT-six-maneuvers/'
    
    ######################################### PRED SET DIRECTORY #########################################################################################
    filepath_pred_Set = '/Users/louis/cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10-seconds/test' # Local Machine
    filepath_pred_Set = 'cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10-seconds/test' # HAL GPU Cluster
    
    ######################################################################################################################################################
    df = pd.read_csv('Run_1_final_rounded.csv') # read in the data 
    original_data = df.copy() # copy
    print(df.keys()) # print the keys just in case 

    lanes_to_analyze = [2,3,4,5]
    # print(f'Unique lanes: {lanes_to_analyze}') 
    
    output_results = [] # output trajectories
    output_results = {key:[] for key in lanes_to_analyze}

    batch_size = 2048 # batch size for the model and choose from [1,2,4,8,16,32,64,128,256,512,1024,2048]
    temp_stop = 10 # index where we want to stop the simulation

    # Initialize network 
    net = highwayNet_six_maneuver(args) # we are going to initialize the network 
    full_path = os.path.join(directory, model_directory) # create a full path 
    net.load_state_dict(torch.load(full_path, map_location=torch.device(device))) # load the model onto the local machine 

    ################################ CHECK GPU AVAILABILITY ###############################################################
    if args['use_cuda']: 
        net = net.cuda()
    #########################################################################################################################

    ################################ INITIALIZE DATA LOADERS ################################################################
    predSet = ngsimDataset(filepath_pred_Set, t_h=30, t_f=100, d_s=2)
    predDataloader = DataLoader(predSet,batch_size=batch_size,shuffle=True,num_workers=3,collate_fn=predSet.collate_fn)
    lossVals = torch.zeros(50).to(device) # Louis code
    counts = torch.zeros(50).to(device) # Louis code

    ################################# TRAIN & VALIDATION LOSS VALUES ############################################################
    accuracy = [] # we are going to store the accuracy values 
    train_loss = [] # we are going to store the training loss values 
    val_loss = [] # we are going to store the validation loss values 
    prev_val_loss = math.inf # we are going to store the previous validation loss values
    net.train_flag = False # neural network training flag is initialized to be False by default 

    ################################## SAVING DATA ##############################################################################
    fut_predictions = [] # future prediction values 
    maneuver_predictions = [] # maneuver prediction values 

    ################################## OUTPUT DATA ##############################################################################
    print(f'Length of the pred data loader: {len(predDataloader)}') # this prints out 1660040 

    lanes_with_overpass = { # key is the lane number, value is the xloc where the overpasses are 
        2:[1800,2200],
        3:[1800,2200] 
    }

    for lane in lanes_with_overpass: # for each lane to be analyzed 
        predicted_traj = [] # we are going to store the predicted trajectories 
        for overpass in lanes_with_overpass[lane]: # for each overpass for that lane
            print(f'Lane: {lane} | Overpass: {overpass}') # print the lane and the overpass 
            predicted_traj_temp = None # initialized the predicted trajectory to be appended 
            for i, data  in enumerate(predDataloader): # for each index and data in the predicted data loader 
                print(f'Index of Data: {i}') # just for testing, print out the index of the current data to be analyzed 

                # if i == temp_stop:
                #     break
                
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
                predicted_traj_temp = predict_trajectories(original_data, overpass,lane,fut_pred_np) # where the function is called and I feed in maneurver pred and future prediction points         
            predicted_traj.append(predicted_traj_temp) # append the predicted trajectories 
        
        plot_trajectory(lane, df, predicted_traj) # plot the predicted trajectories
 
    
    # with open(directory+saving_directory+"output_results.data", "wb") as filehandle:
    #     pickle.dump(np.array(output_results), filehandle, protocol=4)

if __name__ == '__main__': # run the code
    main() # call the main function 