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

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import numpy as np
import scipy.integrate as integrate
from scipy.special import erf
import math

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
   
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
################################################################################################################################################

'''
xloc: Longitudinal N/S movement 
yloc: Lateral E/S Movement
''' 

############################################# LINE INTEGRAL CALCULATIONS #######################################################################
# def line_integral(x1, x2, muX, sigX):  # Line Integral Function 
#     epsilon = 1e-8 # Small value to prevent division by zero
#     cost = 0  # cost 
#     sig = abs(sigX) + epsilon  # Since we're only using sigX

#     # Adjusted calculations to use muX and sigX directly.
#     a = (math.pow(x1 - x2, 2)) * (1 / (2 * sig)) + epsilon
#     b = ((-2 * x1 * x1 + 2 * x1 * x2 + 2 * x1 * muX - 2 * x2 * muX)) * (1 / (2 * sig))
#     c = (math.pow(x1 - muX, 2)) * (1 / (2 * sig))

#     cost += (math.exp(((b * b) / (4 * a)) - c) / (2 * math.pi * sig)) * (1 / math.sqrt(a)) * \
#             (math.sqrt(math.pi) / 2) * (math.erf(math.sqrt(a) + b / (2 * math.sqrt(a))) - math.erf(b / (2 * math.sqrt(a)))) * \
#             math.sqrt(math.pow(x1 - x2, 2))


#     return cost

def line_integral(x1, y1, x2, y2, muX, muY, sigX, sigY): # Line Integral Function 
    epsilon = 0.00001 # Small value to prevent division by zero 1e-5 5e-5 1e-6 1e-7 optimal
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
    
    return cost
 
 
 
##################################################################################################################################################

# The heatmap values on the right show the value of the normal distribution.
# x and y have to be the prediction values. 
# Now let's plot the trajectories using x and y trajectories. Then bring into the starting point.

def generate_normal_distribution(fut_pred, lane,batch_num):
    num_maneuvers = len(fut_pred)
    x = np.linspace(0,100,100)  
    y = np.linspace(-100,100,100)  
    Xc, Yc = np.meshgrid(x, y)
    combined_Z = np.zeros(Xc.shape)
    plt.figure(figsize=(18, 12)) 

    for m in range(num_maneuvers):
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

        ax.set_xlabel('Time (s)', fontsize = 30)
        ax.set_ylabel('Location (m)', fontsize = 30)
        ax.xaxis.set_major_locator(plt.MaxNLocator(100)) # Increase the number of grid lines on the x-axis 
        ax.yaxis.set_major_locator(plt.MaxNLocator(60)) # Increase the number of grid lines on the y-axis
        ax.grid()

        fig.set_size_inches(80,30)
        fig.savefig(f'trajectory_plots/trajectory-'+ axis_temp+'.png')



def plot_trajectories_with_threshold_original(x_list, muX, sigX, ident, possible_traj_temp_ID, overpass_start_time, overpass_end_time, overpass_end_loc_x, threshold):
    y_axis = len(x_list)

    muX_time = []
    prev_time = overpass_start_time
    for i in range(len(muX[0])):
        muX_time.append(prev_time)
        prev_time += 0.2
    
    time_list = []
    prev_time = overpass_start_time
    for j in range(len(x_list)):
        time_list.append(prev_time)
        prev_time += 0.1
    
    plt.plot(time_list, x_list, label=f'Possible Trajectory {possible_traj_temp_ID}', linestyle='-', color='black')

    # Plot predicted muX for each maneuver with color gradient based on sigX
    for m, (muX_m, sigX_m) in enumerate(zip(muX, sigX)): 
        for i in range(len(muX_m)):
            # Determine the color based on the normalized sigX value
            normalized_sigX = sigX_m[i] / np.max(sigX_m)
            color = plt.cm.Blues(1 - normalized_sigX) if normalized_sigX <= 0.5 else plt.cm.YlOrRd(normalized_sigX)
            plt.scatter(muX_time[i], muX_m[i], color=color, edgecolors='none')

    plt.xlabel('Time (s)', fontsize=15)
    plt.ylabel('X Location (m)', fontsize=15)   
    plt.grid(True)
    plt.tight_layout() 
    plt.savefig(f'possible_plots/{ident}_muX_vs_X.png')
    plt.close()


def determine_overpass_y(incoming_trajectories): 
    y_list = []
    y_ids = incoming_trajectories['ID'].unique()
    for temp in y_ids:
        temp_y = incoming_trajectories[incoming_trajectories['ID']==temp]['yloc'].values 
        y_list.append(temp_y[-1])
    
    result = np.mean(y_list)
    result = round(np.mean(y_list),1) 
    return result


def scale_data(muX, muY, method='minmax'): 
    muX = np.array(muX).reshape(-1, 1)
    muY = np.array(muY).reshape(-1, 1)
    
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'zscore':
        scaler = StandardScaler()
    elif method == 'log':
        scaler = None
        muX = np.log1p(abs(muX))
        muY = np.log1p(abs(muY))
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid method. Choose from 'minmax', 'zscore', 'log', 'robust'.")

    if scaler:
        muX_scaled = scaler.fit_transform(muX).flatten()
        muY_scaled = scaler.fit_transform(muY).flatten()
    else:
        muX_scaled = muX.flatten()
        muY_scaled = muY.flatten()

    return muX_scaled.tolist(), muY_scaled.tolist()



def predict_trajectories(input_data, overpass_start_loc_x, overpass_end_loc_x, fut_pred, batch_num, delta, alpha):
    num_maneuvers = len(fut_pred)  # Number of different maneuvers 
    overpass_length = overpass_end_loc_x - overpass_start_loc_x  # Length of the overpass   
    incoming_trajectories = input_data[input_data['xloc'] <= overpass_start_loc_x]  # Incoming trajectory before overpass  
    outgoing_trajectories = input_data[(input_data['xloc'] >= overpass_end_loc_x)]  # Groundtruth trajectory after the overpass  
    possible_trajectories = input_data[(input_data['xloc'] >= overpass_end_loc_x)]  # All possible trajectories that we need to consider
    print(len(possible_trajectories['ID'].unique()),possible_trajectories['ID'].unique()) #
    
    IDs_to_traverse = possible_trajectories['ID'].unique()[:100]  # Vehicle IDs that need to be traversed  
    
    overpass_start_loc_y = determine_overpass_y(incoming_trajectories)  
    incoming_trajectories_copy = incoming_trajectories.copy()
    outgoing_trajectories_copy = outgoing_trajectories.copy() 
    possible_trajectories_copy = possible_trajectories.copy()  

    # Save intermediate data
    ingoing_pd = pd.DataFrame(incoming_trajectories)
    ingoing_pd.to_csv('before/incoming.csv', index=False)
    outgoing_pd = pd.DataFrame(outgoing_trajectories)
    outgoing_pd.to_csv('before/outgoing.csv', index=False)
    possible_before_pd = pd.DataFrame(possible_trajectories)
    possible_before_pd.to_csv('before/possible_before.csv', index=False)

    possible_traj_list = []  # Store all the possible trajectories
    stat_time_frame = np.arange(0, delta, 0.1)  # This is the 0.1 seconds increment part 
    stat_time_frame = np.round(stat_time_frame, 1)

    # Store results
    results = []
    traj_details = []

    for key, ident in enumerate(IDs_to_traverse):
        ingoing_temp_data = incoming_trajectories[incoming_trajectories['ID'] == ident]
        
        if len(ingoing_temp_data['time']) == 0:
            print(f"Skipping ID {ident} due to no incoming data")
            continue  # Skip this ID if there's no incoming data

        
        overpass_start_time = ingoing_temp_data['time'].values[-1]

        for poss_id in IDs_to_traverse:
            current_data = possible_trajectories[(possible_trajectories['ID'] == poss_id)]
            if len(current_data) != 0:
                raw_time_stamps = current_data['time'] - overpass_start_time
                time_stamps = [round(t, 1) for t in raw_time_stamps]
                check_traj_time = min(time_stamps)
                
                if check_traj_time in stat_time_frame:
                    time_stamps_edited = current_data['time'].values - overpass_start_time
                    time_stamps_edited_rounded = [round(t, 1) for t in time_stamps_edited]
                    possible_traj_data = {
                        'ID': ident,'Poss_ID':poss_id, 'overpass_start_time': overpass_start_time,'before_time': current_data['time'].values, 'time': time_stamps_edited_rounded, 
                        'xloc': current_data['xloc'].values, 'yloc': current_data['yloc'].values
                    } # Store the trajectory data in the list
                    possible_traj_list.append(possible_traj_data)

    possible_traj_df = pd.DataFrame(possible_traj_list)
    possible_traj_df.to_csv(f'possible_trajectories/batch_{batch_num}_possible_trajectories.csv', index=False)

    trajectories = []  # Final set of trajectories that we would have traversed 
    best_trajectory = {'ID': None, 'time': None, 'xloc': None, 'yloc': None, 'maneuver': None, 'line_integral_values': None}
    
    # Filter out IDs that are not present in the possible trajectories 
    modified_IDs_to_traverse = possible_traj_df['ID'].unique()  

    for counter,ids in enumerate(modified_IDs_to_traverse):
        best_traj_info = None
        
        traj_to_check = possible_traj_df[possible_traj_df['ID'] == ids]
        traj_to_check_IDs = traj_to_check['Poss_ID'].values
        print(traj_to_check_IDs)

        highest_integral_value = float('-inf')

        for poss_counter,traj_id in enumerate(traj_to_check_IDs):   
            print(f'Counter: {counter+1}/{len(modified_IDs_to_traverse)} | Poss_Counter: {poss_counter+1}/{len(traj_to_check_IDs)} | ID: {ids} | Trajectory ID: {traj_id}')
            
            possible_traj_temp = possible_traj_df[(possible_traj_df['Poss_ID'] == traj_id)]

            x_list = possible_traj_temp['xloc'].values[0] 
            y_list = possible_traj_temp['yloc'].values[0]  
            original_time = possible_traj_temp['before_time'].values[0] 
            traj_time = possible_traj_temp['time'].values[0]
 
            x_ref = overpass_start_loc_x
            y_ref = overpass_start_loc_y
            
            
            for m in range(num_maneuvers):
                segment_integral = 0 
                muX_before = fut_pred[m][:,batch_num,1]  # swap the muX, muY with muY and muX
                muY_before = fut_pred[m][:,batch_num,0]  # swap the muX, muY with muY and muX
                sigX = fut_pred[m][:,batch_num,3]  # swap the sigX, sigY with sigY and sigX
                sigY = fut_pred[m][:,batch_num,2]  # swap the sigX,

                start_idx = list(stat_time_frame).index(traj_time[0])

                sigx_store = sigX[start_idx:]
                mux_store_temp = [mx + x_ref for mx in muX_before]
                mux_store  = mux_store_temp[start_idx:]

                sigy_store = sigY[start_idx:]
                muy_store_temp = [my + y_ref for my in muY_before]
                muy_store  = muy_store_temp[start_idx:]
                

                # Ensure lengths match
                min_len = min(len(x_list), len(mux_store))
                x_list = x_list[:min_len]
                mux_store = mux_store[:min_len]
                sigx_store = sigx_store[:min_len]


                y_list = y_list[:min_len]
                muy_store = muy_store[:min_len]
                sigy_store = sigy_store[:min_len]

                N = min_len - 2

                # Line integral or other calculations
                for i in range(0, N):
                    x1, x2, x3 = x_list[i], x_list[i + 1], x_list[i + 2]
                    temp_muX, temp_sigX = mux_store[i], sigx_store[i]
                    temp_muX2, temp_sigX2 = mux_store[i + 1], sigx_store[i + 1]

                    y1, y2, y3 = y_list[i], y_list[i + 1], y_list[i + 2]
                    temp_muY, temp_sigY = muy_store[i], sigy_store[i]
                    temp_muY2, temp_sigY2 = muy_store[i + 1], sigy_store[i + 1]

                    # segment_integral += line_integral(x1, x2, temp_muX, temp_sigX)
                    # segment_integral += line_integral(x2, x3, temp_muX, temp_sigX)

                    segment_integral += line_integral(x1, y1, x2, y2, temp_muX, temp_muY, temp_sigX, temp_sigY)
                    segment_integral += line_integral(x2, y2, x3, y3, temp_muX, temp_muY, temp_sigX, temp_sigY)

                # Save detailed trajectory information for each Vehicle_ID and possible ID
                traj_details.append({
                    'Vehicle_ID': ids,
                    'Possible_ID': traj_id,
                    
                    'before_time':original_time,
                    'time': traj_time,
                    'xloc': x_list,
                    'yloc': y_list,
                    'muX': mux_store,
                    'muY': muy_store,
                    'sigX': sigx_store,
                    'sigY': sigy_store,
                    'line_integral_values': segment_integral,
                    'maneuver': m + 1
                })

                if segment_integral > highest_integral_value:
                    highest_integral_value = segment_integral
                    best_traj_info = {
                        'Vehicle_ID': ids,
                        'Predicted_ID': traj_id,
                        
                        'before_time':original_time,
                        'time': traj_time,
                        'xloc': x_list, 
                        'yloc': y_list,
                        'muX': mux_store, 
                        'muY': muy_store,
                        'sigX': sigx_store, 
                        'sigY': sigy_store,
                        'line_integral_values': segment_integral, 
                    }

            if best_traj_info:
                best_trajectory = best_traj_info

        if best_trajectory:
            results.append(best_trajectory)

    # Save the overall results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('overall_results/overall_results.csv', index=False)

    # Save detailed trajectory information
    if traj_details:
        traj_details_df = pd.DataFrame(traj_details)
        traj_details_df.to_csv('details/trajectory_details.csv', index=False)
    


 

def evaluate_trajectory_prediction():   
    overall_results_df = pd.read_csv('overall_results/overall_results.csv') 

    # Initialize a list to store whether each ID has the maximum line integral value
    correctness = []

    # Group the dataframe by 'Vehicle_ID'
    grouped_df = overall_results_df.groupby('Vehicle_ID')
    

    # Check for each group if the ID has the maximum line integral value
    for vehicle_id, group in grouped_df:
        max_integral_value = group['line_integral_values'].max()
        print(f'max integral: {max_integral_value}')
        for idx, row in group.iterrows():
            if row['Predicted_ID'] == row['Vehicle_ID'] and row['line_integral_values'] == max_integral_value:
                correctness.append(1)
            else:
                print('Wrong',row['Vehicle_ID'])
                correctness.append(0)
    
 
    return correctness 


def calculate_accuracy(correct_predictions_data):
    correct_predictions = sum(data == 1 for data in correct_predictions_data)
    accuracy = (correct_predictions / len(correct_predictions_data)) * 100
    return np.round(accuracy,2)



 
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
    filepath_pred_Set = 'cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10_seconds/test' # HAL GPU Cluster
    
    file_to_read = 'Louis_ngsim/Louis_ngsim.xlsx'  # NGSIM csv dataset 
    df = pd.read_excel(file_to_read) # read in the data 

    # file_to_read = 'I294_Cleaned.csv'  # TGSIM csv dataset 
    # df = pd.read_csv(file_to_read) # read in the data


    original_data = df.copy() # copy the dataframe 
    lanes_to_analyze = [-2] # lanes to analyze  
    batch_size = 128 # batch size for the model and choose from [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192] 
    batch_num = batch_size-1

    ################################## OVERPASS LOCATION (ASSUMPTION) #################################################################################################################################################################
    ################################## SUCCESS CASES ##################################################################################################################################################################################
    # overpass_start_loc_x,overpass_end_loc_x = 1930, 1945 # both in meters Overpass width 15 meters (50 feets)  74.37% | 38.03% Accuracy  
    # overpass_start_loc_x,overpass_end_loc_x = 1895, 1910 # both in meters Overpass width 15 meters (50 feets)  78.51% | 33.80% Accuracy 
    overpass_start_loc_x,overpass_end_loc_x = 1800, 1815 # 15 meters 80.66% | 25.00% Accuracy
    overpass_start_loc_y = 157
    # overpass_start_loc_x,overpass_end_loc_x = 1705, 1720 # 15 meters 70.96% | 37.00% Accuracy
    # overpass_start_loc_x,overpass_end_loc_x = 1755, 1770 # 15 meters 70.00% | 23.75% Accuracy
    # overpass_start_loc_x,overpass_end_loc_x = 2065, 2080 # 15 meters 70.02% | 27.75% Accuracy
    # overpass_start_loc_x,overpass_end_loc_x = 2111, 2126 # 15 meters 71.65% | 38.06% Accuracy
    # overpass_start_loc_x,overpass_end_loc_x = 1050, 1065 # 15 meters 71.67% | 20.24% Accuracy
    # overpass_start_loc_x,overpass_end_loc_x = 1120, 1135 # 15 meters 77.03% | 14.65% Accuracy

    # overpass_start_loc_x,overpass_end_loc_x = 1165, 1180 # 15 meters 77.67% | 26.77% Accuracy
    #################################################################################################################################################################################
 
    ################################### FAILED CASES ############################################################################################################
    # overpass_start_loc_x,overpass_end_loc_x = 1570, 1585 # both in meters Overpass width 15 meters (50 feets)   35.07% | 24.62% Accuracy 
    # overpass_start_loc_x,overpass_end_loc_x = 1320, 1335 # both in meters Overpass width 15 meters (50 feets)  40.80% | 15.87% Accuracy 
    #################################################################################################################################################################################
    
    ################################### TRIAL RUNS #################################################################################################
    # overpass_start_loc_x,overpass_end_loc_x = 1800, 1805 # 5 meters  96.12% | 20.59% Accuracy
    # overpass_start_loc_x,overpass_end_loc_x = 1800, 1810 # 10 meters 89.73% | 23.53% Accuracy
    # overpass_start_loc_x,overpass_end_loc_x = 1800, 1815 # 15 meters 80.66% | 25.00% Accuracy
    # overpass_start_loc_x,overpass_end_loc_x = 1800, 1820 # 20 meters 50.33% | 30.88% Accuracy
    # overpass_start_loc_x,overpass_end_loc_x = 1800, 1825 # 25 meters 33.00% | 25.00% Accuracy
    # overpass_start_loc_x,overpass_end_loc_x = 1800, 1830 # 30 meters 18.12% | 22.06% Accuracy


    delta = 5 # time interval that we will be predicting for 
    alpha = 12 # value to adjust for the statistical parameters
 
    ################################# NEURAL NETWORK INITIALIZATION ######################################################## 
    net = highwayNet_six_maneuver(args) # we are going to initialize the network 
    model_path = 'trained_model_TGSIM/cslstm_m.tar' # The model that achieved 80.86% accuracy  (Between 78.12% to 81.66%, I chose 80.86%)
    model_path = 'ngsim_trained_models/cslstm_m.tar' # NGSIM model
    net.load_state_dict(torch.load(model_path, map_location=torch.device(device))) # load the model onto the local machine 

    ################################# CHECK GPU AVAILABILITY ###############################################################
    if args['use_cuda']: 
        net = net.to(device)
    #########################################################################################################################

    ################################# INITIALIZE DATA LOADERS ################################################################
    predSet = tgsimDataset(filepath_pred_Set, t_h=30, t_f=100, d_s=2)
    predDataloader = DataLoader(predSet,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=predSet.collate_fn)
    lossVals = torch.zeros(50).to(device) # Louis code
    counts = torch.zeros(50).to(device) # Louis code

    ################################## SAVING DATA ##############################################################################
    fut_predictions = [] # future prediction values 
    maneuver_predictions = [] # maneuver prediction values 

    ################################## OUTPUT DATA ##############################################################################
    print(f'Length of the pred data loader: {len(predDataloader)}')

    predictions_data = [] # prediction data to store 

    ################################## LANES TO BE ANALYZED #####################################################################################
    predicted_traj = None # we are going to store the predicted trajectories 
    lane = -2 
    for i, data  in enumerate(predDataloader): # for each index and data in the predicted data loader  
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

        
        fut_pred_np = [] # store the future pred points 

        for k in range(6): #manuevers mean the 
            fut_pred_np_point = fut_pred[k].clone().detach().cpu().numpy()
            fut_pred_np.append(fut_pred_np_point)

        fut_pred_np = np.array(fut_pred_np) # convert the fut pred points into numpy
        predict_trajectories(original_data, overpass_start_loc_x,overpass_end_loc_x,fut_pred_np,batch_size-1,delta,alpha) # where the function is called and I feed in maneurver pred and future prediction points         
            
        # pred_man = {
        #     'muX':[],
        #     'muY':[],
        #     'sigX':[],
        #     'sigY':[],
        #     'time':[],
        #     'man':[]
        # }
 

        # for m in range(6):
        #     muX_before = fut_pred[m][:, batch_num, 1].detach().numpy()  # detach and convert to numpy
        #     muY_before = fut_pred[m][:, batch_num, 0].detach().numpy()  # detach and convert to numpy
        #     sigX = fut_pred[m][:, batch_num, 3].detach().numpy()  # detach and convert to numpy
        #     sigY = fut_pred[m][:, batch_num, 2].detach().numpy()  # detach and convert to numpy
            
        #     # Add offsets
        #     muX_before = [mx + overpass_start_loc_x for mx in muX_before]
        #     muY_before = [my + overpass_start_loc_y for my in muY_before]
            
        #     # Append to pred_man dictionary
        #     pred_man['muX'].extend(muX_before)
        #     pred_man['muY'].extend(muY_before)
        #     pred_man['sigX'].extend(sigX.tolist())
        #     pred_man['sigY'].extend(sigY.tolist())
            
        #     # Time and maneuver information
        #     time = 0
        #     time_frame = []
        #     man_list = []
        #     for i in range(len(muX_before)):
        #         time_frame.append(np.round(time, 1))
        #         man_list.append(m + 1)
        #         time += 0.2
            
        #     pred_man['time'].extend(time_frame)
        #     pred_man['man'].extend(man_list)
        
        
        # predicted_maneuvers = pd.DataFrame(pred_man)
        # predicted_maneuvers = predicted_maneuvers[['time','man','muX','muY','sigX','sigY']]
        # predicted_maneuvers.to_csv('predicted_maneuvers/predicted_maneuvers.csv',index=False)

        # Plot muX over time for each maneuver with a different color
        # plt.figure()
        # for m in range(1, 7):  # Assuming 6 maneuvers labeled 1 to 6
        #     maneuver_data = predicted_maneuvers[predicted_maneuvers['man'] == m]
        #     plt.scatter(maneuver_data['time'], maneuver_data['muX'], label=f'Maneuver {m}')

        # plt.xlabel('Time (s)')
        # plt.ylabel('X')
        # plt.legend()
        # plt.savefig('predicted_maneuvers_muX_time_scatter.png')
        

        # # Scatter plot muX vs muY for each maneuver with different colors
        # plt.figure()
        # for m in range(1, 7):  # Assuming 6 maneuvers labeled 1 to 6
        #     maneuver_data = predicted_maneuvers[predicted_maneuvers['man'] == m]
        #     plt.scatter(maneuver_data['muX'], maneuver_data['muY'], label=f'Maneuver {m}')

        # plt.xlabel('muX')
        # plt.ylabel('muY')
        # plt.legend()
        # plt.savefig('predicted_maneuvers_muX_muY_scatter.png')
         

        analyzed_traj = evaluate_trajectory_prediction()
        print(f'analyzed trajectory: {analyzed_traj}')

        accuracy_score = calculate_accuracy(analyzed_traj)    
        print(f'Accuracy Score: {accuracy_score}%') 


        break 
     
 

if __name__ == '__main__': # run the code
    main() # call the main function 