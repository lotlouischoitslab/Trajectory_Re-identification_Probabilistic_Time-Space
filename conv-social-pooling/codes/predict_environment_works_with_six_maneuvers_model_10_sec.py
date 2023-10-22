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
Format of the output:
- 6 movements, each movement has probability distribution
- Straight, Accel, straight, decel, right, decel, left, decl 
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
- Set of trajectories 

FOCUS:
- Write a function that take 50 points. Each has 4 variables. Get all the trajectories and write the line integral 
- Do that it's done 
- Rest is one for loop 
'''

############################################# DATA CLEANING AND PARSING #########################################

def load_from_pickle(filename): # Function to load in the data 
    with open(filename, 'rb') as file: # open the file 
        data = pickle.load(file) # load the file into the computer

        if isinstance(data, np.ndarray):  # Convert potential numpy arrays to Python lists
            data = data.tolist()

        for i, item in enumerate(data): # If the items of data are also numpy arrays, convert them to lists
            if isinstance(item, np.ndarray): # check if they are numpy arrays 
                data[i] = item.tolist() # convert each item into list 

    return data # return the trajectory data plots 

def get_shape(obj): # Recursive function to calculate the shape of the trajectory data 
    if isinstance(obj, list): # if it is a list 
        if len(obj) == 0: # if empty 
            return 0, # just return 0
        shapes = [get_shape(sub_obj) for sub_obj in obj] # get the shapes
        max_shape = max(shapes, key=lambda x: len(x))
        return (len(obj),) + max_shape
    elif isinstance(obj, np.ndarray): # if numpy array
        return obj.shape # just return the shape 
    else:
        return () # This is a leaf node

########################## FUNCTIONS TO FILTER OUT EMPTY LISTS #################################################
def filter_empty(sublist): # Filter the empty elements in our given input data 
    if isinstance(sublist, list): # Remove empty lists or arrays
        return [item for item in (filter_empty(item) for item in sublist) if (isinstance(item, list) and len(item) > 0) or (isinstance(item, np.ndarray) and item.size > 0) or item]
    elif isinstance(sublist, np.ndarray): # If numpy array is not empty, return the array
        return sublist if sublist.size > 0 else None # make sure we return the sublist if it is greater than 0, otherwise just return None 
    return sublist # return the sublist 


def filter_points(x, y): # Recursive function to call the filter empty function 
    return filter_empty(x), filter_empty(y) # call the helper functions 
################################################################################################################



def line_integral(x1, y1, x2, y2, obj): # Line integral calculator assuming obj has a shape of (N, 3), where N is the number of maneuvers
    x1_x2_sq = np.power(x1 - x2, 2)
    y1_y2_sq = np.power(y1 - y2, 2)
    
    denom = 2 * obj[:, 2]
    denom[denom < 1e-9] = 1e-9  # Ensure no values too close to zero

    a = (x1_x2_sq + y1_y2_sq) / denom
    b = (-2 * x1 + 2 * x2 + 2 * obj[:, 0] - 2 * x2 * obj[:, 0] + 
         -2 * y1 + 2 * y2 + 2 * obj[:, 1] - 2 * y2 * obj[:, 1]) / denom
    c = (np.power(x1 - obj[:, 0], 2) + np.power(y1 - obj[:, 1], 2)) / denom
    
    # Mask where 'a' is too close to zero
    mask = np.abs(a) >= 1e-9
    a = a[mask]
    b = b[mask]
    c = c[mask]
    
    # Check if the masked values lead to an empty array. Return 0 if true.
    if a.size == 0:
        return 0.0
    
    denom = 2 * np.sqrt(a)
    denom[denom < 1e-9] = 1e-9  # Ensure no values too close to zero

    exp_arg = (b * b) / (4 * a) - c
    exp_arg = np.clip(exp_arg, -20, 20)  # Avoid too large/small arguments for exponential function

    cost = (np.exp(exp_arg) / (2 * np.pi * obj[mask, 2])) * \
           (1 / np.sqrt(a)) * (np.sqrt(np.pi) / 2) * \
           (erf(np.sqrt(a) + b / denom) - erf(b / denom)) * \
           np.sqrt(x1_x2_sq + y1_y2_sq)

    return cost.sum()  # return the sum of the calculated cost



def create_object(muX, muY, sigX,sigY): # Helper function create an object of normal distribution parameters
    return np.column_stack([muX, muY, (sigX-sigY)**2]) # Stack up the values

 
def predict_trajectories(x_trajectory, y_trajectory, fut_pred, traj_length, batch_size=16): # function to predict trajectories
    num_maneuvers = len(fut_pred) # This would be 6 because we have 6 possible maneuvers 
    highest_integral_value = float('-inf')  # Initialize with a very small number
    best_trajectory = {
        'X':[],
        'y':[]
    } # Placeholder for the best trajectory's x and y values

    for m in range(num_maneuvers): # for each of the 6 maneuvers
        objects_for_integral = create_object(fut_pred[m][:, :, 0], fut_pred[m][:, :, 1], fut_pred[m][:, :, 2], fut_pred[m][:, :, 3]) # get the muX, muY, sigX, sigY values
        for b in range(batch_size): # for each b in batch size 
            x_traj = x_trajectory[m][b][0] # extract the x_traj trajectories
            y_traj = y_trajectory[m][b][0] # extract the y_traj trajectories
            total_integral_for_trajectory = sum(line_integral(x_traj[i], y_traj[i], x_traj[i+1], y_traj[i+1], objects_for_integral) for i in range(traj_length-1)) # sum up the line integrals
            
            if total_integral_for_trajectory > highest_integral_value: # Check if this trajectory has the highest integral value so far
                highest_integral_value = total_integral_for_trajectory # update the highest integral value
                best_trajectory['X'] = x_traj # assign the x trajectories
                best_trajectory['y'] = y_traj # assign the y trajectories 
   
    return best_trajectory # return the best trajectory dictionary  


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
    temp_x_trajectory_directory = 'train_trajectory_x.data'
    temp_y_trajectory_directory = 'train_trajectory_y.data'

    x_trajectory_directory = trajectories_directory + temp_x_trajectory_directory
    y_trajectory_directory = trajectories_directory + temp_y_trajectory_directory

    x_trajectory = load_from_pickle(x_trajectory_directory)
    y_trajectory = load_from_pickle(y_trajectory_directory)
    # print("Shape of x data:", get_shape(x_trajectory))
    # print("Shape of y data:", get_shape(y_trajectory))

    x_trajectory,y_trajectory = filter_points(x_trajectory,y_trajectory)

    # print(f'x-traj: {x_trajectory}')
    # print(f'y-traj: {y_trajectory}')

    # x_trajectory = torch.tensor(x_trajectory,device=device)
    # y_trajectory = torch.tensor(y_trajectory,device=device)
    
    print("Shape of x data:", get_shape(x_trajectory))
    print("Shape of y data:", get_shape(y_trajectory))    

    manuever_len,d2,d3,traj_len = get_shape(x_trajectory)

    batch_size = 16 # batch size for the model and choose from [1,2,4,8,16,32,64,128,256]
    temp_stop = 1000 # index where we want to stop the simulation

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

    # predDataloader = DataLoader(predSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=predSet.collate_fn)
    predDataloader = DataLoader(predSet,batch_size=batch_size,shuffle=True,num_workers=0,collate_fn=predSet.collate_fn)
    lossVals = torch.zeros(50).to(device) # Louis code
    counts = torch.zeros(50).to(device) # Louis code

    ######################### TRAIN & VALIDATION LOSS VALUES #######################################
    accuracy = [] # we are going to store the accuracy values 
    train_loss = [] # we are going to store the training loss values 
    val_loss = [] # we are going to store the validation loss values 
    prev_val_loss = math.inf # we are going to store the previous validation loss values
    net.train_flag = False # neural network training flag is initialized to be False by default 

    ########################## SAVING DATA ##############################################################################
    data_points = [] # the data points
    fut_predictions = [] # future prediction values 
    lat_predictions = [] # lateral prediction values
    lon_predictions = [] # longitudinal prediction values 
    maneuver_predictions = [] # maneuver prediction values 
    num_points = 0 # number of points we have analyzed 

    ######################### OUTPUT DATA ##############################################################################
    output_results = [] # output trajectories
    
    print(f'Length of the pred data loader: {len(predDataloader)}') # this prints out 12970 
    # 6 movements (maneuvers) with probability distributions: 
    # Actions are either straight or accelerate or deccelerate or right or left or rear 
    # This is a possible sequence: Straight, Accel, Straight, Decel, Right, Decel, Left, Decl

    for i, data  in enumerate(predDataloader): # for each index and data in the predicted data loader 
        print(f'Index of Data: {i}') # just for testing, print out the index of the current data to be analyzed 
         
        ########################## ASSIGN A VALUE WHERE WE WANT TO STOP ###############################################
        if i == temp_stop: 
            break 
        ###############################################################################################################
        
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
        # print(f'fut_pred: {fut_pred}') 
        # print(f'maneuver_pred: {maneuver_pred}') 
        
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
        fut_pred_np = np.array(fut_pred_np)
        print(f'trained and tested fut pred point: {fut_pred_np.shape}')

        for j in range(points_np.shape[0]):
            point = points_np[j]
            data_points.append(point)
            fut_pred_point = fut_pred_np[:,:,j,:]
            fut_predictions.append(fut_pred_point)
            maneuver_m = maneuver_pred[j].detach().to(device).numpy()
            maneuver_predictions.append(maneuver_m)
            num_points += 1
            # if num_points%10000 == 0:
            #     print('point: ', num_points)
            #     print('point.shape should be (49,): ', point.shape)
            #     print('fut_pred_point.shape should be (6,t_f//d_s,5): ', fut_pred_point.shape)
            #     print('maneuver_m.shape should be (6,):', maneuver_m.shape)

        predicted_traj = predict_trajectories(x_trajectory,y_trajectory,fut_pred_np,traj_len,batch_size=batch_size) # where the function is called and I feed in maneurver pred and future prediction points 
        output_results.append(predicted_traj) # output result is a list of predicted trajectory dictionaries 
        print(f'output results: {output_results}')
   
 
 
    # Test Error
    mse = lossVals / counts # mean squared error
    rmse = np.sqrt(mse) # root mean sqaured error 
    
    # print(f'MSE: {mse}')
    # print(f'RMSE: {rmse}')   # Calculate RMSE, feet
    # print(f'Number of data points: {num_points}')
    # print(f'Output results shape: {len(output_results)} | {len(output_results[0])}')
    # print(f'Output results: {output_results}')

    with open(directory+saving_directory+"data_points.data", "wb") as filehandle:
        pickle.dump(np.array(data_points), filehandle, protocol=4)

    with open(directory+saving_directory+"fut_predictions.data", "wb") as filehandle:
        pickle.dump(np.array(fut_predictions), filehandle, protocol=4)

    with open(directory+saving_directory+"maneuver_predictions.data", "wb") as filehandle:
        pickle.dump(np.array(maneuver_predictions), filehandle, protocol=4)

if __name__ == '__main__': # run the code
    main() # call the main function 