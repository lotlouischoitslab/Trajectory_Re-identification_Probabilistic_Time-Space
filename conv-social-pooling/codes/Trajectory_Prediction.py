 



from __future__ import print_function
import torch
from model_six_maneuvers import highwayNet_six_maneuver
from TGSIM_utils import tgsimDataset,maskedNLL,maskedMSE,maskedNLLTest
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
from ast import literal_eval
   
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # FOR MULTI-GPU system using a single gpu
os.environ["CUDA_VISIBLE_DEVICES"]="0" # The GPU id to use, usually either "0" or "1" # this should be 0
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
########## Use this temporary but we need to fix the OpenBLAS error #########
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='.*openblas.*')
warnings.simplefilter('ignore', np.RankWarning)
##############################################################################

'''
xloc: Longitudinal N/S movement 
yloc: Lateral E/S Movement
'''

############################################# LINE INTEGRAL CALCULATIONS ######################################################
# def line_integral(x1, y1, x2, y2, muX, muY, sigX, sigY): 
#     cost = 0
#     sig = np.sqrt((sigX**2 + sigY**2)/2)

#     # Adjusted calculations to use muX, muY, sigX, and sigY directly.
#     a = (math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2)) * (1 / (2 * sig))
#     b = ((-2 * x1 * x1 + 2 * x1 * x2 + 2 * x1 * muX - 2 * x2 * muX) + \
#         (-2 * y1 * y1 + 2 * y1 * y2 + 2 * y1 * muY - 2 * y2 * muY)) * (1 / (2 * sig))
#     c = (math.pow(x1 - muX, 2) + math.pow(y1 - muY, 2)) * (1 / (2 * sig))

#     cost += (math.exp(((b * b) / (4 * a)) - c) / (2 * math.pi * sig)) * (1 / math.sqrt(a)) * \
#             (math.sqrt(math.pi) / 2) * (math.erf(math.sqrt(a) + b / (2 * math.sqrt(a))) - math.erf(b / (2 * math.sqrt(a)))) * \
#             math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    
#     # print(f'integral value: {cost}')
#     return cost
 
def line_integral(x1, y1, x2, y2, muX, muY, sigX, sigY):
    epsilon = 1e-10  # Small value to prevent division by zero
    cost = 0
    sig = np.sqrt((sigX**2 + sigY**2)/2) + epsilon

    # Adjusted calculations to use muX, muY, sigX, and sigY directly.
    a = (math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2)) * (1 / (2 * sig)) + epsilon
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
        # print(f"Processing maneuver {m+1}/{num_maneuvers}")
        muX = fut_pred[m][:,batch_num,0]
        muY = fut_pred[m][:,batch_num,1]
        sigX = fut_pred[m][:,batch_num,2]
        sigY = fut_pred[m][:,batch_num,3]
      
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
            muX = fut_pred[m][:,batch_num,0]
            muY= fut_pred[m][:,batch_num,1]
            sigX = fut_pred[m][:,batch_num,2]
            sigY = fut_pred[m][:,batch_num,3]
            
            # axs[0].scatter(stat_time_frame, muX, color=colors[m],s=2,label=f'Maneuver {m+1}', zorder=1)
            # axs[1].scatter(stat_time_frame, muY, color=colors[m],s=2,label=f'Maneuver {m+1}', zorder=1)
            axs[2].scatter(muX, muY, color=colors[m],s=2,label=f'Maneuver {m+1}', zorder=1)
            
        # axs[0].legend()
        # axs[1].legend()
        axs[2].legend()
        plt.suptitle('Trajectories X and Y Locations over Time')
        plt.savefig('temp_trajectory_plot.png')


def plot_original_trajectories(incoming_trajectories,outgoing_trajectories):
    incoming_IDs = incoming_trajectories['ID'].unique()
    outgoing_IDs = outgoing_trajectories['ID'].unique()

    IDs = []
    all_ts = []
    all_ys = [] 

    axis_coordinates = ['xloc','yloc']

    for axis_temp in axis_coordinates:
        fig, ax = plt.subplots() # get xs and ts of each vehicle

        for i in incoming_IDs:
            # print(f'ID: {i}')
            temp_data = incoming_trajectories[incoming_trajectories['ID']==i]
            ys = temp_data[axis_temp].to_numpy() 
            ts = temp_data.time.to_numpy()
            ax.scatter(ts, ys,s=1) 
        

        for j in outgoing_IDs:
            # print(f'ID: {j}')
            temp_data = outgoing_trajectories[outgoing_trajectories['ID']==j]
            ys = temp_data[axis_temp].to_numpy() 
            ts = temp_data.time.to_numpy()
            ax.scatter(ts, ys,s=1)
           
        
        if axis_temp == 'xloc':
            ax.set_xlim(0, 320)
            ax.set_ylim(1000, 2200)  # Set y-axis range from 0 to 2200

        ax.set_xlabel('Time (s)', fontsize = 20)
        ax.set_ylabel('Location (m)', fontsize = 20)
        ax.xaxis.set_major_locator(plt.MaxNLocator(100)) # Increase the number of grid lines on the x-axis 
        ax.yaxis.set_major_locator(plt.MaxNLocator(60)) # Increase the number of grid lines on the y-axis
        ax.grid()

        fig.set_size_inches(120,30)
        fig.savefig(f'trajectory_plots/trajectory-'+ axis_temp+'.png')


def plot_total_trajectories(incoming_trajectories,predicted_traj,outgoing_trajectories,possible_traj_df):
    IDs = possible_traj_df['ID'].unique() 
    all_ts = []
    all_ys = [] 

    fig, ax = plt.subplots() # get xs and ts of each vehicle

    first_time_pred = []
    first_point_pred = []
    last_time_pred = []
    last_point_pred = []

    for i in IDs:
        # print(f'ID: {i}')
        temp_data = incoming_trajectories[incoming_trajectories['ID']==i]
        ys = temp_data['old_xloc'].to_numpy() 
        ts = temp_data.time.to_numpy()
        # print(ts)
        ax.scatter(ts, ys,s=1) 
        if len(ts) != 0:
            first_time_pred.append(ts[-1])
            first_point_pred.append(ys[-1])
    

    for j in IDs:
        # print(f'ID: {j}')
        temp_data = outgoing_trajectories[outgoing_trajectories['ID']==j]
        ys = temp_data['old_xloc'].to_numpy() 
        ts = temp_data.time.to_numpy()
        ax.scatter(ts, ys,s=1)
        if len(ts) != 0:
            last_time_pred.append(ts[0])
            last_point_pred.append(ys[0])

    for t1,t2,f1,f2 in zip(first_time_pred,last_time_pred,first_point_pred,last_point_pred):
        plt.plot([t1,t2],[f1,f2],color='r',label='connect')
     
    
    # print('first_time_pred',first_time_pred)
    # print('last_time_pred',last_time_pred)

    # print('first_point_pred',first_point_pred)
    # print('last_point_pred',last_point_pred)

    ax.set_xlim(40, 250)
    ax.set_ylim(1000, 2200)  # Set y-axis range from 0 to 2200 
    ax.set_xlabel('Time (s)', fontsize = 20)
    ax.set_ylabel('Location (m)', fontsize = 20)
    ax.xaxis.set_major_locator(plt.MaxNLocator(100)) # Increase the number of grid lines on the x-axis 
    ax.yaxis.set_major_locator(plt.MaxNLocator(60)) # Increase the number of grid lines on the y-axis
    ax.grid()

    fig.set_size_inches(120,30)
    fig.savefig(f'trajectory_plots/selected_trajectory.png')



def predict_trajectories(input_data, overpass_start_loc_x, overpass_end_loc_x, lane, fut_pred, batch_num, delta):
    num_maneuvers = len(fut_pred)  # Number of different maneuvers 
    input_data = input_data[input_data['lane'] == lane].reset_index(drop=True)  # Filter data for the given lane
    incoming_trajectories = input_data[input_data['xloc'] <= overpass_start_loc_x] # Incoming trajectory before overpass 
    outgoing_trajectories = input_data[input_data['xloc'] >= overpass_end_loc_x] # Groundtruth trajectory after the overpass  
    possible_trajectories = input_data[input_data['xloc'] >= overpass_end_loc_x] # All possible trajectories that we need to consider
    IDs_to_traverse = possible_trajectories['ID'].unique() # Vehicle IDs that needs to be traversed
    overpass_length = overpass_end_loc_x - overpass_start_loc_x # length of the overpass 
    # overpass_start_time = overpass_start_time_input 
    # overpass_end_time = overpass_start_time + delta
    # print(f'overpass start time: {overpass_start_time}')
    # print(f'overpass end time: {overpass_end_time}')

    plot_original_trajectories(incoming_trajectories,outgoing_trajectories)

    incoming_trajectories_copy = incoming_trajectories.copy()
    outgoing_trajectories_copy = outgoing_trajectories.copy() 
    possible_trajectories_copy = possible_trajectories.copy()  

    incoming_trajectories['old_xloc'] = incoming_trajectories_copy['xloc']
    outgoing_trajectories['old_xloc'] = outgoing_trajectories_copy['xloc']
    possible_trajectories['old_xloc'] = possible_trajectories_copy['xloc']

    incoming_trajectories['xloc'] -= overpass_start_loc_x
    outgoing_trajectories['xloc'] -= overpass_start_loc_x
    possible_trajectories['xloc'] -= overpass_start_loc_x

    incoming_trajectories['old_yloc'] = incoming_trajectories_copy['yloc']
    outgoing_trajectories['old_yloc'] = outgoing_trajectories_copy['yloc']
    possible_trajectories['old_yloc'] = possible_trajectories_copy['yloc']

   
    ingoing_pd = pd.DataFrame(incoming_trajectories)
    ingoing_pd.to_csv('before/incoming.csv')

    outgoing_pd = pd.DataFrame(outgoing_trajectories)
    outgoing_pd.to_csv('before/outgoing.csv')

    possible_before_pd = pd.DataFrame(possible_trajectories)
    possible_before_pd.to_csv('before/possible_before.csv')


    possible_traj_list = []  # Store all the possible trajectories
    stat_time_frame = np.arange(0, delta+0.1, 0.1)
    stat_time_frame = np.round(stat_time_frame, 1)



    for key, ident in enumerate(IDs_to_traverse): 
        ingoing_temp_data = incoming_trajectories[incoming_trajectories['ID'] == ident]
        current_data = possible_trajectories[possible_trajectories['ID'] == ident] 
        current_outgoing = outgoing_trajectories[outgoing_trajectories['ID'] == ident] 
        # print(f'Ident traverse: {ident}')
        
        if len(current_data['time']) == 0:
            print(f"Skipping ID {ident} due to no incoming data")
            continue  # Skip this ID if there's no incoming data

        # print('time checked',current_data['time'].values)
        overpass_start_time = current_data['time'].values[0]
        overpass_end_time = overpass_start_time + 5
        # overpass_start_loc_y = current_data['yloc'].values[0]        
        possible_trajectories.loc[:, 'adjusted_time'] = (possible_trajectories_copy['time'] - overpass_start_time).round(1)

        possible_trajectories = possible_trajectories[(possible_trajectories['adjusted_time']>= 0.0)&(possible_trajectories['adjusted_time']<= 5.0)] 

        possible_trajectories.to_csv('possible_trajectories/ID'+str(ident)+'possible.csv')
        outgoing_trajectories.to_csv('outgoing'+str(ident)+'traj.csv')
        print('overpass time',overpass_start_time,'to',overpass_end_time)
        
        best_trajectory = None
        trajectory_updates = []
        possible_IDS = possible_trajectories['ID'].unique()

        for possible_traj_temp_ID in possible_IDS: 
            #possible_trajectories.loc[possible_trajectories['ID'] == possible_traj_temp_ID, 'yloc'] = possible_trajectories.loc[possible_trajectories['ID'] == possible_traj_temp_ID, 'old_yloc']-overpass_start_loc_y
            #possible_trajectories.to_csv('possible_trajectories/ID'+str(ident)+'possible.csv')
            highest_integral_value = float('-inf')  # Reset for each ID
            temp_proj_to_traverse = possible_trajectories[possible_trajectories['ID']==possible_traj_temp_ID]
            traj_time = temp_proj_to_traverse['adjusted_time'].values
            x_list = temp_proj_to_traverse['xloc'].values
            y_list = temp_proj_to_traverse['yloc'].values 
            old_x_list = temp_proj_to_traverse['old_xloc'].values
            old_y_list = temp_proj_to_traverse['old_yloc'].values
            segment_integral = 0.0  # Reset for each maneuver
            overpass_start_loc_y = possible_trajectories.loc[possible_trajectories['ID'] == possible_traj_temp_ID, 'yloc'].values[0]

            for m in range(num_maneuvers):
                muX = fut_pred[m][:,batch_num,0]
                muY = fut_pred[m][:,batch_num,1]
                sigX = fut_pred[m][:,batch_num,2]
                sigY = fut_pred[m][:,batch_num,3]
                print(traj_time)

                start_idx = list(stat_time_frame).index(traj_time[0]) # minimum adjusted time value 
                # print('stat time frame',stat_time_frame)
                # print('index',start_idx)

                mux_store = muX[start_idx:]
                muy_store = muY[start_idx:]
                sigx_store = sigX[start_idx:]
                sigy_store = sigY[start_idx:]



                for i in range(len(traj_time) - 2):  # you already avoid the last index to ensure x2 can be accessed
                    index = len(traj_time) - len(mux_store) + i

                    # Ensure the index is valid for both x1 and x2 (since you access index + 1 for x2)
                    if 0 <= index < len(x_list) - 1:  
                        x1, x2 = x_list[index], x_list[index + 1]
                        y1, y2 = y_list[index], y_list[index + 1]

                        # Ensure your time indexing also does not go out of bounds
                        if i < len(stat_time_frame) - 1:
                            temp_time = stat_time_frame[i]
                            temp_muX, temp_muY = mux_store[i], muy_store[i]
                            temp_sigX, temp_sigY = sigx_store[i], sigy_store[i]

                            # Now perform your line integral or any other calculations
                            segment_integral += line_integral(x1, y1-overpass_start_loc_y, x2, y2-overpass_start_loc_y, temp_muX, temp_muY, temp_sigX, temp_sigY)
                    else:
                        print(f"Skipping index {index} as it is out of bounds.")
                        continue  # Skip this iteration if indices are out of range


            if segment_integral > highest_integral_value:
                highest_integral_value = segment_integral
                best_traj_info = {
                    'Vehicle_ID':ident,
                    'ID': possible_traj_temp_ID,
                    'time': traj_time,
                    'xloc': x_list,
                    'yloc': y_list,
                    'old_xloc':old_x_list,
                    'old_yloc':old_y_list,
                    'line_integral_values': segment_integral,
                    'maneuver': m + 1 
                }

                trajectory_updates.append(best_traj_info.copy())  # Track updates

            if best_traj_info and (best_trajectory is None or best_traj_info['line_integral_values'] > best_trajectory['line_integral_values']):
                best_trajectory = best_traj_info
            
        # Convert the list of trajectories into a DataFrame 
        trajectory_updates_df = pd.DataFrame(trajectory_updates)
        trajectory_updates_df.to_csv(f'temp_best/simulation_{ident}_trajectory_updates.csv', index=False) 


        if best_trajectory:
            best_trajectory_df = pd.DataFrame([best_trajectory])
            best_trajectory_df.to_csv(f'best_trajectories/simulation_{ident}_best_trajectory.csv', index=False)
        
         
        #break

 

def plot_predicted_trajectories(predicted_xlist, predicted_ylist, ground_truth_xlist, ground_truth_ylist, traj_id,timeframe):
    plt.figure(figsize=(10, 5))
    plt.scatter(timeframe, predicted_xlist, color='r', label='Predicted Trajectory')
    plt.plot(timeframe, ground_truth_xlist, 'b-', label='Ground Truth Trajectory')
    plt.xlabel('Time (seconds)')
    plt.ylabel('X Location')
    plt.title(f'Trajectory Comparison for Vehicle ID {traj_id}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'trajectory_plots/trajectory_comparison_{traj_id}.png')
 
 
 
def evaluate_trajectory_prediction(): 
    best_trajectories_dir = "best_trajectories"
    best_trajectories_files = os.listdir(best_trajectories_dir)
    possible_trajectories_dir = os.listdir("possible_trajectories")
    outgoing_trajectories = pd.read_csv('before/outgoing.csv')
    correct_predictions = []
    print(f'files: {len(best_trajectories_files)}')

    for predicted_trajectory_input_file in best_trajectories_files: 
        predicted_trajectory_input_path = os.path.join(best_trajectories_dir, predicted_trajectory_input_file)
        predicted_trajectory_input = pd.read_csv(predicted_trajectory_input_path) 
        
        #ID_to_check = predicted_trajectory_input['ID'].values[0]  # Extract the single ID value
        ID_to_check = predicted_trajectory_input['Vehicle_ID'].values[0]
        ground_truth_trajectory = outgoing_trajectories[outgoing_trajectories['ID'] == ID_to_check]
        print(f'check ground id: {ID_to_check}')

        if len(predicted_trajectory_input['xloc']) != 0:
            s_clean_x = predicted_trajectory_input['xloc'].values[0].strip("[]")
            list_str_x = s_clean_x.split()
            predicted_xlist = [round(float(x),2) for x in list_str_x]

            s_clean_y = predicted_trajectory_input['yloc'].values[0].strip("[]")
            list_str_y = s_clean_y.split()
            predicted_ylist = [round(float(y),2) for y in list_str_y]
 

            ground_truth_xlist = np.round(ground_truth_trajectory['xloc'].values[:len(predicted_xlist)], 2)
            ground_truth_ylist = np.round(ground_truth_trajectory['yloc'].values[:len(predicted_ylist)], 2)
    
            predicted_xlist_plot = predicted_xlist
            predicted_ylist_plot = predicted_ylist

            ground_truth_xlist_plot = np.round(ground_truth_trajectory['xloc'].values[:len(predicted_xlist)], 2)
            ground_truth_ylist_plot = np.round(ground_truth_trajectory['yloc'].values[:len(predicted_ylist)], 2)

            timeframe = np.linspace(0, 5, len(predicted_xlist))  # space out from 0 to 5 seconds with length of the predicted trajectory  

            print(f'predicted_xlist: {predicted_xlist}') 
            print(f'predicted_ylist: {predicted_ylist}') 

            print(f'ground_truth_xlist: {ground_truth_xlist}') 
            print(f'ground_truth_ylist: {ground_truth_ylist}')  

            # plot_predicted_trajectories(predicted_xlist_plot, predicted_ylist_plot, ground_truth_xlist_plot, ground_truth_ylist_plot, ID_to_check, timeframe)

            # Check if the lengths of the lists match
            if len(predicted_xlist) != len(ground_truth_xlist) or len(predicted_ylist) != len(ground_truth_ylist):
                print(len(predicted_xlist), len(ground_truth_xlist), len(predicted_ylist), len(ground_truth_ylist))
                print("The lengths of predicted and ground truth trajectories do not match.")
                correct_predictions.append(0)
                continue

            # Check if the trajectories match
            for px, py, gx, gy in zip(predicted_xlist, predicted_ylist, ground_truth_xlist, ground_truth_ylist):
                if px != gx or py != gy:
                    correct_predictions.append(0)
                    break
            else:
                correct_predictions.append(1)
 
    return correct_predictions



def calculate_accuracy(correct_predictions_data):
    correct_predictions = sum(data == 1 for data in correct_predictions_data)
    accuracy = (correct_predictions / len(correct_predictions_data)) * 100
    return accuracy


# def plot_accuracy_graph(overpass_start_times,accuracy_score_list):
#     plt.figure(figsize=(10,6))
#     plt.plot(overpass_start_times,accuracy_score_list)
#     plt.xlabel('Overpass Start Time (s)') 
#     plt.ylabel('Accuracy Score %') 
#     plt.savefig('Accuracy.png')


 
def main(): # Main function 
    args = {} # Network Arguments
    print('cuda available',torch.cuda.is_available())
    print('torch version',torch.__version__)
    print('cuda version', torch.version.cuda)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     
    if device == 'cuda':
        args['use_cuda'] = True 
    else:
        args['use_cuda'] = False 
    
    print(f'My device: {device}')
    args['encoder_size'] = 64
    args['decoder_size'] = 128
    args['in_length'] = 16
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

 
    ######################################### PRED SET DIRECTORY #########################################################################################
    # filepath_pred_Set = '/Users/louis/cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10-seconds/test' # Local Machine
    filepath_pred_Set = 'cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10_seconds/test' # HAL GPU Cluster
    
    ######################################################################################################################################################
    file_to_read = 'I294_Cleaned.csv'  
    
    df = pd.read_csv(file_to_read) # read in the data 
    original_data = df.copy() # copy the dataframe 

    #lanes_to_analyze = sorted(df['lane'].unique())[1:-1] # lanes to analyze 
    temp_lane = -2 
    lanes_to_analyze = [temp_lane] # lanes to analyze 
    # lanes_to_analyze = sorted(df['lane'].unique())  # lanes to analyze 
    print(f'Unique lanes: {lanes_to_analyze}') 
    
    batch_size = 128 # batch size for the model and choose from [1,2,4,8,16,32,64,128,256,512,1024,2048]

    ################################## OVERPASS LOCATION (ASSUMPTION) ########################################################################
    overpass_start_loc_x,overpass_end_loc_x = 1570, 1600 # both in meters 
    delta = 5 # time interval that we will be predicting for 

    # overpass_start_time_list = np.arange(overpass_start_time,overpass_start_time+2,1)
    accuracy_score_list = []
  
    ################################# NEURAL NETWORK INITIALIZATION ######################################################## 
    net = highwayNet_six_maneuver(args) # we are going to initialize the network 
    model_path = 'trained_model_TGSIM/cslstm_m.tar'
    net.load_state_dict(torch.load(model_path, map_location=torch.device(device))) # load the model onto the local machine 

    ################################# CHECK GPU AVAILABILITY ###############################################################
    if args['use_cuda']: 
        net = net.to(device)
    #########################################################################################################################

    ################################# INITIALIZE DATA LOADERS ################################################################
    predSet = tgsimDataset(filepath_pred_Set, t_h=30, t_f=100, d_s=2)
    predDataloader = DataLoader(predSet,batch_size=batch_size,shuffle=True,num_workers=1,collate_fn=predSet.collate_fn)
    lossVals = torch.zeros(50).to(device) # Louis code
    counts = torch.zeros(50).to(device) # Louis code

    ################################## SAVING DATA ##############################################################################
    fut_predictions = [] # future prediction values 
    maneuver_predictions = [] # maneuver prediction values 

    ################################## OUTPUT DATA ##############################################################################
    print(f'Length of the pred data loader: {len(predDataloader)}') # this prints out 1660040 

    predictions_data = []

    ################################## LANES TO BE ANALYZED #####################################################################################
    predicted_traj = None # we are going to store the predicted trajectories 
    for lane in lanes_to_analyze: # for each lane to be analyzed 
        print(f'Lane: {lane}') # print the lane   
        for i, data  in enumerate(predDataloader): # for each index and data in the predicted data loader 
        
            print(f'Index of Data: {i}/{len(predDataloader)}') # just for testing, print out the index of the current data to be analyzed 
            
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, maneuver_enc  = data # unpack the data   

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

            # l, c = maskedMSETest(fut_pred_max, fut, op_mask) # get the loss value and the count value 
            # l = l.to(device)  # device is the device you determined earlier (cuda or cpu)
            # c = c.to(device)

            # lossVals += l.detach() # increment the loss value 
            # counts += c.detach() # increment the count value  
            fut_pred_np = [] # store the future pred points 

            for k in range(6): #manuevers mean the 
                fut_pred_np_point = fut_pred[k].clone().detach().cpu().numpy()
                fut_pred_np.append(fut_pred_np_point)

            fut_pred_np = np.array(fut_pred_np) # convert the fut pred points into numpy

            predict_trajectories(original_data, overpass_start_loc_x,overpass_end_loc_x,lane,fut_pred_np,batch_size-1,delta) # where the function is called and I feed in maneurver pred and future prediction points         
            generate_normal_distribution(fut_pred_np, lane,batch_size-1)

            analyzed_traj = evaluate_trajectory_prediction()
            print(f'analyzed trajectory: {analyzed_traj}')

            accuracy_score = calculate_accuracy(analyzed_traj)    
            print(f'Accuracy Score: {accuracy_score}%')
            accuracy_score_list.append(accuracy_score)

            #plot_total_trajectories(incoming_trajectories,predicted_traj,outgoing_trajectories,possible_traj_df)

            if i == 0: # Generate and save the distribution plots just for one trajectory
                generate_normal_distribution(fut_pred_np, lane,batch_size-1)
                break 
    
    #plot_accuracy_graph(overpass_start_time_list,accuracy_score_list)
 

if __name__ == '__main__': # run the code
    main() # call the main function 