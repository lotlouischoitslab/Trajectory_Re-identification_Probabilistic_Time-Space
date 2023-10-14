from __future__ import print_function
import torch
from model_six_maneuvers import highwayNet_six_maneuver
from utils_works_with_101_80_cnn_modified_passes_history_too_six_maneuvers import ngsimDataset,maskedNLL,maskedMSE,maskedNLLTest, maskedMSETest
from torch.utils.data import DataLoader
import time
import math
from scipy.stats import multivariate_normal

import pickle
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # FOR MULTI-GPU system using a single gpu
os.environ["CUDA_VISIBLE_DEVICES"]="1" # The GPU id to use, usually either "0" or "1"

########## Use this temporary but we need to fix the OpenBLAS error #########
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='.*openblas.*')
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

def line_integral(X, muX, muY, sigX, sigY): # Line integral function
    integral_values = [] # integral values 
    # print('X shape',X.shape) 
    # print('muX shape',len(muX)) 
    # print('muY shape',len(muY)) 
    # print('sigX shape',len(sigX)) 
    # print('sigY shape',len(sigY)) 
    
    for i in range(X.shape[1]): 
        mean = [muX[i], muY[i]] # get the mean values
        covariance_matrix = np.array([  # construct covariance matrix 
            [sigX[i]**2, 0],
            [0, sigY[i]**2]
        ])
        
        values = multivariate_normal.pdf(X[:, i].reshape(-1, 2), mean=mean, cov=covariance_matrix) # multivariate normal distribution
        integral_values.append(np.sum(values)) # sum up all the values 

    total_integral_value = np.sum(integral_values) # sum them up again 
    return total_integral_value # return the total value 
  

def predict_trajectories(points_np,fut_pred, maneuver_pred): # Function to predict trajectories
    best_maneuvers = [] # store all the best manuevers
    #print(f'fut pred point shape: {fut_pred.shape}')
    X = points_np # assign the X variable to be the points_np
 
    for j in range(points_np.shape[0]):
        #point = points_np[j] # get the points to analyze 
        # print(f'point: {point}') # print the point for debugging 
        # print(f'length of point: {len(point)}')
 
        fut_pred_point = fut_pred[:,:,j,:] # future prediction point
        #print(f'fut pred point: {fut_pred_point}')
        
        max_integral_value = float('-inf') # this is assigned as the negative infinity 
        # print(max_integral_value,'max int')
        best_maneuver_point = None # best maneuver point is initialized as None 
         
        total_integral = 0 # total value of the line integral
        for i in range(6): # for six possible manuever choices 
            muX = fut_pred_point[i, :, 0][1:] # mean x 50 data points
            muY = fut_pred_point[i, :, 1][1:] # mean y
            sigX = fut_pred_point[i, :, 2][1:] # std x 50 data points
            sigY = fut_pred_point[i, :, 3][1:] # std y
            total_integral = line_integral(X,muX, muY, sigX, sigY)

            print(total_integral,'tot integral')
            if total_integral > max_integral_value: # if the total integral is greater than the current max integral value
                max_integral_value = total_integral # max integral value is assigned as the total integral value
                best_maneuver_point = i # best maneuver point is assigned as the current maneuver point
            
        best_maneuvers.append(best_maneuver_point) # append the best maneuver point into the list
    
    print(f'Best maneuver: {best_maneuvers}')
    return best_maneuvers  # return the list of possible maneuvers 



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

    directory = '/Users/louis/cee497projects/trajectory-prediction/codes/predicted_environment/'
    # directory = 'cee497projects/trajectory-prediction/codes/predicted_environment/'

    model_directory = 'models/trained_models_10_sec/cslstm_m.tar'
    saving_directory = 'predicted_data/highwaynet-10-sec-101-80-speed-maneuver-for-GT-six-maneuvers/'
    
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
    # predSet = ngsimDataset('/reza/projects/trajectory-prediction/data/NGSIM/101-80-speed-maneuver-for-GT/10-seconds/train', t_h=30, t_f=100, d_s=2)
    # predSet = ngsimDataset('/reza/projects/trajectory-prediction/data/NGSIM/101-80-speed-maneuver-for-GT/10-seconds/valid', t_h=30, t_f=100, d_s=2)
    # predSet = ngsimDataset('/reza/projects/trajectory-prediction/data/NGSIM/101-80-speed-maneuver-for-GT/10-seconds/test', t_h=30, t_f=100, d_s=2)

    filepath_pred_Set = '/Users/louis/cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10-seconds/test'
    # filepath_pred_Set = 'cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10-seconds/test'
    predSet = ngsimDataset(filepath_pred_Set, t_h=30, t_f=100, d_s=2)

    # predDataloader = DataLoader(predSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=predSet.collate_fn)
    predDataloader = DataLoader(predSet,batch_size=batch_size,shuffle=True,num_workers=0,collate_fn=predSet.collate_fn)
    lossVals = torch.zeros(50).to(device) # Louis code
    counts = torch.zeros(50).to(device) # Louis code

    ######################### Variables holding train and validation loss values #######################################
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
    
    # print(f'Length of the pred data loader: {len(predDataloader)}')
    # 6 movements (maneuvers) with probability distributions: 
    # Actions are either straight or accelerate or deccelerate or right or left or rear 
    # This is a possible sequence: Straight, Accel, Straight, Decel, Right, Decel, Left, Decl

    for i, data  in enumerate(predDataloader): # for each index and data in the predicted data loader 
        print(f'Index of Data: {i}') # just for testing, print out the index of the current data to be analyzed 
     
         
        ############ Comment this out if deploying to GPU Cluster #############################################
        if i == 10: # we are just going to stop at index 100 for testing 
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
        fut_pred_max = torch.zeros_like(fut_pred[0]) # get the max predicted values 
        for k in range(maneuver_pred.shape[0]): # for each value in the maneuver predicted shapes
            indx = torch.argmax(maneuver_pred[k, :]).detach() # get the arg max of the maneuvered prediction values
            fut_pred_max[:, k, :] = fut_pred[indx][:, k, :] # future predicted value max 
        l, c = maskedMSETest(fut_pred_max, fut, op_mask) # get the loss value and the count value 
        lossVals += l.detach() # increment the loss value 
        counts += c.detach() # increment the count value 

        ##DEBUG
        #print(f"len(fut_pred), must be 6: {len(fut_pred)}")
        for m in range(len(fut_pred)):
            #print(f"shape of fut_pred[m], must be (t_f//d_s,batch_size,5): {fut_pred[m].shape}")
            for n in range(batch_size):
                muX = fut_pred[m][:,n,0]
                muY = fut_pred[m][:,n,1]
                sigX = fut_pred[m][:,n,2]
                sigY = fut_pred[m][:,n,3]
                # print(f'muX: {muX}')
                # print(f'muY: {muY}')
                # print(f'sigX: {sigX}')
                # print(f'sigY: {sigY}')
                # print('len of muX',len(muX)) 
                # print('len of muY',len(muY)) 
                # print('len of sigX',len(sigX)) 
                # print('len of sigY',len(sigY)) 
        ##END OF DEBUG

        points_np = points.numpy() # convert to numpy arrays 
        fut_pred_np = [] # store the future pred points 
        for k in range(6): #manuevers mean the 
            fut_pred_np_point = fut_pred[k].clone().detach().cpu().numpy()
            fut_pred_np.append(fut_pred_np_point)
        fut_pred_np = np.array(fut_pred_np)
        outputs = predict_trajectories(points_np,fut_pred_np, maneuver_pred) # where the function is called and I feed in maneurver pred and future prediction points 
        output_results.append(outputs)
        for j in range(points_np.shape[0]):
            point = points_np[j]
            # print(f'point.shape should be (49,): {point.shape}')
            # print(f'point: {point}')
            data_points.append(point)
            fut_pred_point = fut_pred_np[:,:,j,:]

            ###DEBUG
            # print('fut_pred_point.shape should be (6,t_f//d_s,5): ',fut_pred_point.shape) #6 is for different lon and lat maneuvers
            # print("check this: \n")
            for i in range(6):
                muX = fut_pred_point[i, :, 0]
                muY = fut_pred_point[i, :, 1]
                sigX = fut_pred_point[i, :, 2]
                sigY = fut_pred_point[i, :, 3]
                # print(f'muX: {muX}')
                # print(f'muY: {muY}')
                # print(f'sigX: {sigX}')
                # print(f'sigY: {sigY}')
                # print('len of muX',len(muX)) 
                # print('len of muY',len(muY)) 
                # print('len of sigX',len(sigX)) 
                # print('len of sigY',len(sigY)) 
            ###END OF DEBUG

            fut_predictions.append(fut_pred_point)
            maneuver_m = maneuver_pred[j].detach().to(device).numpy()
            maneuver_predictions.append(maneuver_m)
            num_points += 1
            # if num_points%10000 == 0:
            #     print('point: ', num_points)
            #     print('point.shape should be (49,): ', point.shape)
            #     print('fut_pred_point.shape should be (6,t_f//d_s,5): ', fut_pred_point.shape)
            #     print('maneuver_m.shape should be (6,):', maneuver_m.shape)


    
    # Print Test Error
    # print('MSE: ', lossVals / counts)
    # print('RMSE: ', torch.pow(lossVals / counts,0.5))   # Calculate RMSE, feet
    # print('number of data points: ', num_points)
    print(f'Output results shape: {len(output_results)} | {len(output_results[0])}')
    print(f'Output results: {output_results}')

    with open(directory+saving_directory+"data_points.data", "wb") as filehandle:
        pickle.dump(np.array(data_points), filehandle, protocol=4)

    with open(directory+saving_directory+"fut_predictions.data", "wb") as filehandle:
        pickle.dump(np.array(fut_predictions), filehandle, protocol=4)

    with open(directory+saving_directory+"maneuver_predictions.data", "wb") as filehandle:
        pickle.dump(np.array(maneuver_predictions), filehandle, protocol=4)

if __name__ == '__main__': # run the code
    main() # call the main function 