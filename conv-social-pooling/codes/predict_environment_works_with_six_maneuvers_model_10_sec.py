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
from scipy.stats import multivariate_normal

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

def compute_accuracy(predicted_maneuvers, ground_truth_maneuvers):
    print(f'predicted man: {predicted_maneuvers}')
    print(f'ground truth man: {ground_truth_maneuvers}')
    correct_predictions = np.sum(np.array(predicted_maneuvers) == np.array(ground_truth_maneuvers))
    total_predictions = len(predicted_maneuvers)
    accuracy = correct_predictions / total_predictions
    return accuracy
 

def line_integral(x1, y1, x2, y2, obj):
    cost = 0 
    for i in range(len(obj)):
        denom = 2*obj[i][2]
        if denom == 0:
            denom = 1

        a = (math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2)) / denom
        b = ((-2 * x1 * x1 + 2 * x1 * x2 + 2 * x1 * obj[i][0] - 2 * x2 * obj[i][0]) + 
             (-2 * y1 * y1 + 2 * y1 * y2 + 2 * y1 * obj[i][1] - 2 * y2 * obj[i][1])) / denom
        c = (math.pow(x1 - obj[i][0], 2) + math.pow(y1 - obj[i][1], 2)) / denom
        
        # If a is zero or too close to zero, continue to the next iteration
        if abs(a) < 1e-9:  # a threshold to handle floating point inaccuracies
            continue

        denom = 2 * math.sqrt(a)
        if denom == 0:
            denom = 1

        cost += (math.exp((b * b) / (4 * a) - c) / (2 * math.pi * obj[i][2])) * (1 / math.sqrt(a)) * \
                (math.sqrt(math.pi) / 2) * (math.erf(math.sqrt(a) + b / denom) - math.erf(b / denom)) * \
                math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    return cost


def create_object(muX, muY, sigX, sigY): # creates an object to feed into the line integral 
    to_return = [] # will return a 2d list of [[mu_x, mu_y, sigma_square], [mu_x, mu_y, sigma_square], ...]
    for mux,muy,sig in zip(muX,muY,sigX): # so for each mu_x, mu_y, sig_x 
        temp = [mux,muy,sig**2] # create a temporary list
        to_return.append(temp) # append to the list 
    return to_return # return 


def predict_trajectories(x_trajectory, y_trajectory, fut_pred, traj_length, batch_size=1):
    """
    Predict trajectories based on the provided x_trajectory and y_trajectory for different maneuvers.
    Returns the best maneuvers.
    """
    num_maneuvers = len(fut_pred)
    best_maneuvers = [-1] * num_maneuvers  # Initialize with a default value
    
    for m in range(num_maneuvers):
        max_integral_value = float('-inf')
        
        for n in range(batch_size):
            muX = fut_pred[m][:, n, 0]
            muY = fut_pred[m][:, n, 1]
            sigX = fut_pred[m][:, n, 2]
            sigY = fut_pred[m][:, n, 3]
            x_test = x_trajectory[m][n][0]
            y_test = y_trajectory[m][n][0]
            
            objects_for_integral = create_object(muX, muY, sigX, sigY)

            total_integral_for_batch = sum(
                line_integral(x_test[i], y_test[i], x_test[i+1], y_test[i+1], objects_for_integral) 
                for i in range(traj_length-1)
            )
            
            if total_integral_for_batch > max_integral_value:
                max_integral_value = total_integral_for_batch
                best_maneuver_for_m = m
        
        best_maneuvers[m] = best_maneuver_for_m

    best_maneuvers = np.array(best_maneuvers)
    print(f'best maneuvers: {best_maneuvers}')
    return best_maneuvers




def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)

        # Convert potential numpy arrays to Python lists
        if isinstance(data, np.ndarray):
            data = data.tolist()

        # If the items of data are also numpy arrays, convert them to lists
        for i, item in enumerate(data):
            if isinstance(item, np.ndarray):
                data[i] = item.tolist()

        # You can continue this logic further if there are more nested numpy arrays.
        # If the depth is unknown or very deep, a recursive approach will be needed.

    return data

def get_shape(obj):
    if isinstance(obj, list):
        if len(obj) == 0:
            return 0,
        shapes = [get_shape(sub_obj) for sub_obj in obj]
        max_shape = max(shapes, key=lambda x: len(x))
        return (len(obj),) + max_shape
    elif isinstance(obj, np.ndarray):
        return obj.shape
    else:
        # This is a leaf node
        return ()

########################## FUNCTIONS TO FILTER OUT EMPTY LISTS #################################################
def filter_empty(sublist):
    if isinstance(sublist, list):
        # Remove empty lists or arrays
        return [item for item in (filter_empty(item) for item in sublist) if (isinstance(item, list) and len(item) > 0) or (isinstance(item, np.ndarray) and item.size > 0) or item]
    elif isinstance(sublist, np.ndarray):
        # If numpy array is not empty, return the array
        return sublist if sublist.size > 0 else None
    return sublist


def filter_points(x, y):
    return filter_empty(x), filter_empty(y)
################################################################################################################


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

    ######################################## TRAJECTORY DIRECTORIES #################################################
    #trajectories_directory1 = 'cee497projects/data/101-80-speed-maneuver-for-GT/train/10_seconds/'
  
    trajectories_directory = '/Users/louis/cee497projects/data/101-80-speed-maneuver-for-GT/train/10_seconds/'
    
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

    ####################################### MODEL DIRECTORIES ######################################################
    directory = '/Users/louis/cee497projects/trajectory-prediction/codes/predicted_environment/'
    #directory = 'cee497projects/trajectory-prediction/codes/predicted_environment/'

    model_directory = 'models/trained_models_10_sec/cslstm_m.tar'
    saving_directory = 'predicted_data/highwaynet-10-sec-101-80-speed-maneuver-for-GT-six-maneuvers/'
    
    ######################################### PRED SET DIRECTORY #############################################################
    filepath_pred_Set = '/Users/louis/cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10-seconds/test'
    #filepath_pred_Set = 'cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10-seconds/test'
    
    batch_size = 128 # batch size for the model 

    # Initialize network 
    net = highwayNet_six_maneuver(args) # we are going to initialize the network 
    full_path = os.path.join(directory, model_directory) # create a full path 
    net.load_state_dict(torch.load(full_path, map_location=torch.device(device))) # load the model onto the local machine 

    ############################### Check if GPU is available ###############################################################
    if args['use_cuda']: 
        net = net.cuda()
    #########################################################################################################################

    ################################ Initialize data loaders ################################################################

    predSet = ngsimDataset(filepath_pred_Set, t_h=30, t_f=100, d_s=2)

    # predDataloader = DataLoader(predSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=predSet.collate_fn)
    predDataloader = DataLoader(predSet,batch_size=batch_size,shuffle=True,num_workers=0,collate_fn=predSet.collate_fn)
    lossVals = torch.zeros(50).to(device) # Louis code
    counts = torch.zeros(50).to(device) # Louis code

    ######################### Variables holding train and validation loss values #######################################
    accuracy = [] # we are going to store the accuracy values 
    train_loss = [] # we are going to store the training loss values 
    val_loss = [] # we are going to store the validation loss values 
    prev_val_loss = math.inf # we are going to store the previous validation loss values
    net.train_flag = False # neural network training flag is initialized to be False by default 

    ######################### Saving data ##############################################################################
    data_points = [] # the data points
    fut_predictions = [] # future prediction values 
    lat_predictions = [] # lateral prediction values
    lon_predictions = [] # longitudinal prediction values 
    maneuver_predictions = [] # maneuver prediction values 
    num_points = 0 # number of points we have analyzed 

    ######################### Output data ##############################################################################
    output_results = []
    
    print(f'Length of the pred data loader: {len(predDataloader)}')
    # 6 movements (maneuvers) with probability distributions: 
    # Actions are either straight or accelerate or deccelerate or right or left or rear 
    # This is a possible sequence: Straight, Accel, Straight, Decel, Right, Decel, Left, Decl

    for i, data  in enumerate(predDataloader): # for each index and data in the predicted data loader 
        print(f'Index of Data: {i}') # just for testing, print out the index of the current data to be analyzed 
         
        ############ Comment this out if deploying to GPU Cluster #############################################
        if i == 1: # we are just going to stop at index 100 for testing 
            break 
        #######################################################################################################
        
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

        outputs = predict_trajectories(x_trajectory,y_trajectory,fut_pred_np,traj_len,batch_size=batch_size) # where the function is called and I feed in maneurver pred and future prediction points 
    

        # print(f'prped shape: {predicted_maneuvers.shape}')
        # print(f'ground shape: {ground_truth_maneuvers.shape}')
        #temp_acc = compute_accuracy(predicted_maneuvers, ground_truth_maneuvers)
        #accuracy.append(temp_acc)
        # output_results.append(outputs)
        
        break 
    
    # Accuracy 
    print(f'Accuracies: {accuracy}')
 

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