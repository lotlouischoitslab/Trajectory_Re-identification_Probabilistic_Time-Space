from __future__ import print_function
import torch
from Louis_model_six_maneuvers import highwayNet_six_maneuver
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
xloc: Longitudinal N/S  movement 
yloc: Lateral E/S Movement


Get the final data, cut out a piece of it
What we can do is I pick this section of the road and cut take about 20 ft
Simulate an overpass 
Have one trajectory for 160 ft, between 160 to 180 ft cut it to two pieces, you have before and after trajectory 
Run the code
Look at the middle of the data and revolve 20 ft from the point

Missing part is 1770->1800 ft part 
Go back 5 seconds. (Feed in to the prediction model)
Integral: Get the trajectories from 1800 ft forward 5 seconds (This is the future) assuming there is an overpass 
Connect the trajectories  
Replace the 1770->1800 ft part with the trajectory with 180 ft and 5 sec forward

Format of the output:
- 6 movements, each movement has probability distribution
- Actions are Straight, Accel, straight, decel, right, decel, left, decl 
- This is a possible sequence: Straight, Accel, Straight, Decel, Right, Decel, Left, Decl
- Every point one second during that 10 second horizon, we are getting normal distribution 2D where that car is probabilistically. For 10 seconds, we have 100 points, for every one of those 100 points, we have mean (x,y) and std (vx,vy). These are my outputs. Now, what we want to do is to get highest probability from one of the six movements. 
 
Guidelines to understand the prediction function: 
- There are 6 different maneuvers the car can pick 
- Each maneuver has 50 points
- Each point has probability distribution
- Take each maneuver ALL the 50 points. the corresponding point
- Take the line integral of that particular distrubtuion
- Do this for all 50 points 
- Sum everything up 
- Then do this for all trajectories 
- Pick the manuever and the trajectory with the highest total value of the line integral
'''

############################################# LINE INTEGRAL CALCULATIONS ######################################################
def line_integral(x1, y1, x2, y2, muX, muY, sigX, sigY):
    cost = 0
    sig = abs(sigX - sigY)/2 

    # Adjusted calculations to use muX, muY, sigX, and sigY directly.
    a = (math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2)) * (1 / (2 * sig))
    b = ((-2 * x1 * x1 + 2 * x1 * x2 + 2 * x1 * muX - 2 * x2 * muX) + \
        (-2 * y1 * y1 + 2 * y1 * y2 + 2 * y1 * muY - 2 * y2 * muY)) * (1 / (2 * sig))
    c = (math.pow(x1 - muX, 2) + math.pow(y1 - muY, 2)) * (1 / (2 * sig))

    cost += (math.exp(((b * b) / (4 * a)) - c) / (2 * math.pi * sig)) * (1 / math.sqrt(a)) * \
            (math.sqrt(math.pi) / 2) * (math.erf(math.sqrt(a) + b / (2 * math.sqrt(a))) - math.erf(b / (2 * math.sqrt(a)))) * \
            math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    
    # print(f'integral value: {cost}')
    return cost


# The heatmap values on the right show the value of the normal distribution
# x and y have to be the prediction values. 
# Now let's plot the trajectories using x and y trajectories. Then bring into the starting point

def generate_normal_distribution(fut_pred, lane,batch_num):
    num_maneuvers = len(fut_pred)
    x = np.linspace(0,100,100)  
    y = np.linspace(-100,100,100)  
    Xc, Yc = np.meshgrid(x, y)
    combined_Z = np.zeros(Xc.shape)
    plt.figure(figsize=(18, 12)) 

    for m in range(num_maneuvers):
        print(f"Processing maneuver {m+1}/{num_maneuvers}")
        muY = fut_pred[m][:,batch_num,0]
        muX = fut_pred[m][:,batch_num,1]
        sigY = fut_pred[m][:,batch_num,2]
        sigX = fut_pred[m][:,batch_num,3]
      
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



def plot_pred_trajectories(IDs_to_traverse,incoming_trajectories,ground_truth_underneath_overpass,possible_traj_list,fut_pred,stat_time_frame,batch_num,overpass_start_time,overpass_end_time,num_maneuvers): # plot trajectory function    
    fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharey=True) 
    IDs_to_traverse = [0] 
    print('PLOTTING TRAJECTORIES')
    for temp_ID in IDs_to_traverse: # for each trajectory ID 
        incoming_trajectories_plot = incoming_trajectories[incoming_trajectories['ID'] == temp_ID]
        ground_truth_plot = ground_truth_underneath_overpass[ground_truth_underneath_overpass['ID'] == temp_ID]
        for poss in possible_traj_list: 
            # axs[0].plot(poss['before_time'], poss['xloc'], label=f'poss ID {temp_ID}', linestyle='--')
            # axs[0].set_xlabel('Time')
            # axs[0].set_ylabel('X Location')
            # axs[0].legend() 

            # axs[1].plot(poss['before_time'], poss['yloc'], label=f'poss ID {temp_ID}', linestyle='--')
            # axs[1].set_xlabel('Time')
            # axs[1].set_ylabel('Y Location')
            # axs[1].legend()
        
            axs[2].plot(poss['xloc'], poss['yloc'], label=f'poss ID {temp_ID}') 
            axs[2].set_xlabel('X Location')
            axs[2].set_ylabel('Y Location')
            axs[2].legend()

        
        # Plot original trajectory points before prediction
        #axs[0].plot(incoming_trajectories_plot['time'], incoming_trajectories_plot['xloc'], label=f'Incoming ID {temp_ID}')
        # axs[0].plot(ground_truth_plot['time'], ground_truth_plot['xloc'], label=f'Trajectory ID {temp_ID} Ground Truth',linewidth=2.0)
        # axs[0].set_title('X Locations over Time')
        # axs[0].set_xlabel('Time')
        # axs[0].set_ylabel('X Location')
        # axs[0].legend()

        
        # #axs[1].plot(incoming_trajectories_plot['time'], incoming_trajectories_plot['yloc'], label=f'Incoming ID {temp_ID}')
        # axs[1].plot(ground_truth_plot['time'], ground_truth_plot['yloc'], label=f'Trajectory ID {temp_ID} Ground Truth',linewidth=2.0)
        # axs[1].set_title('Y Locations over Time')
        # axs[1].set_xlabel('Time')
        # axs[1].set_ylabel('Y Location')
        # axs[1].legend()

        # Plot xloc vs yloc graph
        #axs[2].plot(incoming_trajectories_plot['xloc'], incoming_trajectories_plot['yloc'], label=f'Incoming ID {temp_ID}')
        axs[2].plot(ground_truth_plot['xloc'], ground_truth_plot['yloc'], label=f'Trajectory ID {temp_ID} Ground Truth')
        axs[2].set_xlabel('X Location')
        axs[2].set_ylabel('Y Location')
        axs[2].legend()

    
        colors = ['red', 'green', 'blue', 'purple', 'orange','yellow']  # Plot predictive mean locations for each maneuver
        for m in range(num_maneuvers):
            muY = fut_pred[m][:,batch_num,0]
            muX= fut_pred[m][:,batch_num,1]
            sigY = fut_pred[m][:,batch_num,2]
            sigX = fut_pred[m][:,batch_num,3]
            
            # axs[0].scatter(stat_time_frame, muX, color=colors[m],s=2,label=f'Maneuver {m+1}', zorder=1)
            # axs[1].scatter(stat_time_frame, muY, color=colors[m],s=2,label=f'Maneuver {m+1}', zorder=1)
            axs[2].scatter(muX, muY, color=colors[m],s=2,label=f'Maneuver {m+1}', zorder=1)
            
        # axs[0].legend()
        # axs[1].legend()
        axs[2].legend()
        plt.suptitle('Trajectories X and Y Locations over Time')
        plt.savefig('temp_trajectory_plot.png')
   


def predict_trajectories(input_data, overpass_start_time_input,overpass_start_loc_x,overpass_end_loc_x,overpass_start_loc_y,overpass_end_loc_y, lane, fut_pred, batch_num,delta): # Predict Trajectories function
    num_maneuvers = len(fut_pred) # We have 6 different maneuvers 
    input_data = input_data[input_data['lane'] == lane].reset_index(drop=True) # we want to pick for that lane given (this has ALL the trajectories)
    
    incoming_trajectories = input_data[input_data['xloc'] <= overpass_start_loc_x] # we want to get all the incoming trajectories as well
    ground_truth_underneath_overpass = input_data[(input_data['xloc'] >= overpass_start_loc_x) & (input_data['xloc'] <= overpass_end_loc_x)] # underneath the overpass data
    possible_trajectories = input_data[input_data['xloc'] >= overpass_end_loc_x] # the possible set of trajectories can be pass the overpass location
    IDs_to_traverse = possible_trajectories['ID'].unique() # get all the unique IDs  

    overpass_start_time =  overpass_start_time_input
    overpass_end_time = overpass_start_time + delta # time we are assuming where the overpass ends  

    print(f'overpass xloc: {overpass_start_loc_x} -> {overpass_end_loc_x} meters') # Overpass Longitudinal (North/South) coordinates
    print(f'overpass yloc: {overpass_start_loc_y} -> {overpass_end_loc_y} meters') # Overpass Latitudinal (East/West) coordinates
    print(f'overpass time: {overpass_start_time} -> {overpass_end_time} seconds') # Overpass time frame

    
    ################################## ADJUST DATA POINTS RELATIVE TO OVERPASS START LOCATION ############################################
    for temp_ID in IDs_to_traverse:
        incoming_trajectories.loc[incoming_trajectories['ID'] == temp_ID, 'xloc'] -= overpass_start_loc_x 
        ground_truth_underneath_overpass.loc[ground_truth_underneath_overpass['ID'] == temp_ID, 'xloc'] -= overpass_start_loc_x 
        possible_trajectories.loc[possible_trajectories['ID'] == temp_ID, 'xloc'] -= overpass_start_loc_x

        incoming_trajectories.loc[incoming_trajectories['ID'] == temp_ID, 'yloc'] -= overpass_start_loc_y 
        ground_truth_underneath_overpass.loc[ground_truth_underneath_overpass['ID'] == temp_ID, 'yloc'] -= overpass_start_loc_y
        possible_trajectories.loc[possible_trajectories['ID'] == temp_ID, 'yloc'] -= overpass_start_loc_y
    ###############################################################################################################################################################
    
    incoming_trajectories.to_csv('before/00incoming.csv')
    ground_truth_underneath_overpass.to_csv('before/01ground_truth_underneath_overpass.csv')
    possible_trajectories.to_csv('before/02possible.csv')
    possible_traj_list = [] # we will store all the possible trajectories here
    stat_time_frame = np.arange(0,delta, 0.1) # time frame for muX, muY, sigX and sigY 
    stat_time_frame = np.round(stat_time_frame, 1)  # Round stat_time_frame to 1 decimal place

    for ident in IDs_to_traverse: # for each possible trajectory 
        current_data = possible_trajectories[(possible_trajectories['ID'] == ident) & (possible_trajectories['time'] >= overpass_start_time) & (possible_trajectories['time'] <= overpass_end_time)] # extract the current trajectory data
        possible_traj_data = { # possible trajectory data structure 
            'before_time':[],
            'time':[], 
            'xloc':[],
            'yloc':[]
        }

        if len(current_data) != 0:
            # print('ident',ident)
            current_data.to_csv('louis_traverse/current'+str(ident)+'.csv')
            raw_time_stamps = current_data['time']-overpass_start_time
            time_stamps = [round(t, 1) for t in raw_time_stamps]
            check_traj_time = min(time_stamps) # get the starting time for the trajectory 
            # print(f'check traj time: {check_traj_time}') 

            if check_traj_time in stat_time_frame:
                # print(f'check traj time: {check_traj_time} in stat time frame') 
                possible_traj_data['ID'] = ident # ID number for that trajectory 
                possible_traj_data['before_time'] =  current_data['time'] 
                possible_traj_data['time'] =  time_stamps # subtract the overpass start time from the current time
                possible_traj_data['xloc'] =  current_data['xloc'] # subtract the current x location from the overpass end location x coordinate
                possible_traj_data['yloc'] =  current_data['yloc'] # do the same for the y location as well
                
                possible_traj_list.append(possible_traj_data) # append each possible trajectory data into a list 
                possible_traj_pd = pd.DataFrame(possible_traj_data)
                possible_traj_pd.to_csv('louis_traverse/possible_traj'+str(ident)+'.csv')


    ################################# JUST FOR PLOTTING #######################################################################################################################################################
    stat_time_frame_copy = np.arange(overpass_start_time,overpass_end_time,0.1)
    plot_pred_trajectories(IDs_to_traverse,incoming_trajectories,ground_truth_underneath_overpass,possible_traj_list,fut_pred,stat_time_frame_copy,batch_num,overpass_start_time,overpass_end_time,num_maneuvers)
    ###########################################################################################################################################################################################################

    trajectories = [] # final set of trajectories that we would have traversed 
    best_trajectory = {
        'lane':lane,
        'time':0,
        'xloc':[],
        'yloc':[],
        'maneuver':0,
        'muX':0,
        'muY':0,
        'sigX':0,
        'sigY':0,
        'line_integral_values': 0
    }

    highest_integral_value = float('-inf') # assign a really large negative value 

    temp_id = [0] # temporary placeholder because we will be analyzing one trajectory 
    for ids in temp_id: # For each incoming trajectory
        current_trajectory = {
            'lane':lane,
            'time':[],
            'xloc':[],
            'yloc':[],
            'maneuver':[],
            'muX':[],
            'muY':[],
            'sigX':[],
            'sigY':[],
            'line_integral_values': []
        }

        # print('possible traj list',possible_traj_list ) # we have a list of possible trajectories
        # print('length of possible traj list',len(possible_traj_list)) # should be 9

        for possible_traj_temp in possible_traj_list: # for each possible trajectory 
            # print('poss temp',possible_traj_temp)
            traj_time = possible_traj_temp['time'] # get the time frame for this possible trajectory
            x_list = possible_traj_temp['xloc'].values # get all the x-coordinates of the trajectory
            y_list = possible_traj_temp['yloc'].values # get all the y-coordinates of the trajectory
            
            
            for m in range(num_maneuvers): # for each maneuver
                muY, muX, sigY, sigX = fut_pred[m][:, batch_num, :4].T # Extract maneuver-specific predictive parameters
                start_idx = list(stat_time_frame).index(traj_time[0]) # we want to retrieve the index of that check traj time in the prediction time frame
                end_idx = len(traj_time)-1
                
                mux_store = muX[start_idx:] # we want to extract the muX values from the start_idx -> until 50th index
                muy_store = muY[start_idx:] # we want to extract the muY values from the start_idx -> until 50th index
                sigx_store = sigX[start_idx:] # we want to extract the sigX values from the start_idx -> until 50th index
                sigy_store = sigY[start_idx:] # we want to extract the sigY values from the start_idx -> until 50th index
                
                
                # print(f'index:{start_idx} -> {end_idx}')
                # print(f'traverse traj time: {traj_time}')
                # print(f'muX to analyze:{mux_store}')
                # print(f'muY to analyze:{muy_store}')
                # print(f'sigX to analyze:{sigx_store}')
                # print(f'sigY to analyze:{sigy_store}')

                for i in range(end_idx): # Loop through each segment in current_data
                    x1 = x_list[i]
                    x2 = x_list[i+1]
                    y1 = y_list[i]
                    y2 = y_list[i+1]
            
                    # print(f'first: {(x1,y1)}') # Just for checking 
                    # print(f'second: {(x2,y2)}') # Just for checking 

                    temp_time = stat_time_frame[i] # get the time for that time frame 
                    temp_muX = mux_store[i] # store the muX
                    temp_muY = muy_store[i] # store the muY
                    temp_sigX = sigx_store[i] # store the sigX
                    temp_sigY = sigy_store[i] # store the sigY

                    # print('temp muX',temp_muX)
                    # print('temp muY',temp_muY)
                    # print('temp sigX',temp_sigX)
                    # print('temp sigY',temp_sigY)
                
                    segment_integral = line_integral(x1, y1, x2, y2, temp_muX,temp_muY,temp_sigX,temp_sigY) # Calculate line integral for each segment (return 50 values)
                    
                    current_trajectory['time'].append(temp_time) # this is the individual time stamps 
                    current_trajectory['xloc'].append((x1,x2)) # this is the individual (x1,x2)
                    current_trajectory['yloc'].append((y1,y2)) # this is the individual (y1,y2)
                    current_trajectory['muX'].append(temp_muX) # this is the individual muX
                    current_trajectory['muY'].append(temp_muY) # this is the individual muY
                    current_trajectory['sigX'].append(temp_sigX) # this is the individual sigX
                    current_trajectory['sigY'].append(temp_sigY) # this is the individual sigY 
                    current_trajectory['line_integral_values'].append(segment_integral) # append the line integral
                    current_trajectory['maneuver'].append(m+1) # append the maneuver
                    
                    if segment_integral > highest_integral_value: # check if the selected line integral value is greater than or not
                        highest_integral_value = segment_integral # assign the highest line integral value
                        best_trajectory['time'] = stat_time_frame[i] # assign the time
                        best_trajectory['xloc'] = (x1,x2) # this is the individual (x1,x2)
                        best_trajectory['yloc'] = (y1,y2) # this is the individual (y1,y2)
                        best_trajectory['muX'] = temp_muX # this is the individual muX
                        best_trajectory['muY'] = temp_muY # this is the individual muY
                        best_trajectory['sigX'] = temp_sigX # this is the individual sigX
                        best_trajectory['sigY'] = temp_sigY # this is the individual sigY 
                        best_trajectory['line_integral_values'] = segment_integral # append the line integral
                        best_trajectory['maneuver']= m+1 # assign the maneuver

                    trajectories.append(current_trajectory) # Store the current trajectory
    
    for key,temp in enumerate(trajectories): # for each stored dataframe
        trajectories_df = pd.DataFrame(temp) # convert to DataFrame
        trajectories_df.to_csv('all_combinations_trajectories/batch_'+str(batch_num)+'_trajectory_combo.csv', index=False) # Save to CSV
    
    if best_trajectory: # if we have the best trajectory
        best_trajectory_df = pd.DataFrame(best_trajectory) # convert the best trajectory data into dataframe format 
        best_trajectory_df.to_csv('best_trajectories/batch_'+str(batch_num)+'_best_trajectory.csv', index=False) # then convert to csv
    
    return trajectories, best_trajectory # return all the trajectories traversed and the best trajectory 
    
 
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
    file_to_read = 'I294_Cleaned.csv' # or 'raw_trajectory.csv'
    
    df = pd.read_csv(file_to_read) # read in the data 
    original_data = df.copy() # copy the dataframe 

    #lanes_to_analyze = sorted(df['lane'].unique())[1:-1] # lanes to analyze 
    temp_lane = -2 
    lanes_to_analyze = [temp_lane] # lanes to analyze 
    print(f'Unique lanes: {lanes_to_analyze}') 
    
    batch_size = 512 # batch size for the model and choose from [1,2,4,8,16,32,64,128,256,512,1024,2048]

    ################################## OVERPASS LOCATION (ASSUMPTION) ########################################################################
    overpass_start_loc_x,overpass_end_loc_x = 1770, 1800 # both in meters 
    overpass_start_loc_y,overpass_end_loc_y = 161.8, 162.4 # both in meters 
    overpass_start_time = 190.7 # time where the overpass begins in seconds
    delta = 5 # time interval that we will be predicting for

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
            trajectory,predicted_traj = predict_trajectories(original_data,overpass_start_time, overpass_start_loc_x,overpass_end_loc_x,overpass_start_loc_y,overpass_end_loc_y,lane,fut_pred_np,i,delta) # where the function is called and I feed in maneurver pred and future prediction points         
            generate_normal_distribution(fut_pred_np, lane,i)
            
            if i == 0: # Generate and save the distribution plots just for one trajectory
                generate_normal_distribution(fut_pred_np, lane,i)
                break

        # print('Predicted')
        # print(f"{len(predicted_traj['lane'])} | {len(predicted_traj['time'])} | {len(predicted_traj['xloc'])} | {len(predicted_traj['yloc'])}")
        # print('Original Dataframe')
        # print(f"{len(df['lane'])} | {len(df['time'])} | {len(df['xloc'])} | {len(df['yloc'])}")
        # predicted_traj = pd.DataFrame(predicted_traj) # convert the predicted traj into Pandas DataFrame
         


if __name__ == '__main__': # run the code
    main() # call the main function 