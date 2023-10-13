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

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # FOR MULTI-GPU system using a single gpu
os.environ["CUDA_VISIBLE_DEVICES"]="1" # The GPU id to use, usually either "0" or "1"

########## Use this temporary but we need to fix the OpenBLAS error #########
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='.*openblas.*')
##############################################################################


''' 
File Name: predict_environment_works_with_six_maneuvers_model_10_sec.py
NOTES For Louis Sungwoo Cho NCAS HAL Cluster:
Reference: https://www.youtube.com/watch?v=l1dV25xwo0o&list=PLO8UWE9gZTlCtkZbWtEcKgxYVVLIvN2IS&index=1 
Run the GPU: swrun -p gpux1 -r louissc2
Exit the terminal: exit 
We are using CEE497 conda environment 
Go here for more reference: https://wiki.ncsa.illinois.edu/display/ISL20/HAL+cluster 
Make sure to upload the files to the cluster if you have made any changes
0. We need to: conda install -c "conda-forge/label/cf202003" libopenblas
1. To connect to NCSA Hal Cluster: ssh louissc2@hal.ncsa.illinois.edu
conda config --add channels https://ftp.osuosl.org/pub/open-ce/1.5.1/
2. Type in Password & Enter the Authentication code
3. module load opence
4. conda activate CEE497
5. To save: If you're using vim, you can press ESC, then type :wq and press Enter.

./demo.swb
Type the following:
#!/bin/bash 
#SBATCH --job-name="louis_trajectory"
#SBATCH --output="louis_trajectory.out"
#SBATCH --partition=gpux1
#SBATCH --time=2
#SBATCH --reservation=louissc2

module load wmlce

hostname 

./demo2.swb
Type the following: 
#!/bin/bash 
#SBATCH --job-name="louis_trajectory"
#SBATCH --output="louis_trajectory.out"
#SBATCH --partition=gpu
#SBATCH --time=2

module load wmlce

hostname 


########### Run the batch ###########
swbatch ./demo.swb

########### Check Status ############
squeue -u louissc2

########## To launch vim ############
vim ./demo.s
Quit: :q!
#####################################


########### To run GPU on HAL Cluster #############
1. swqueue (GPUs and the queue of users)
2. squeue (List of currently running clusters)   
3. sinfo
4. swrun -p gpux1 
5. module load wmlce
#####################################################
 
'''

def predict_trajectories():
    pass 


if __name__ == '__main__':
    ## Network Arguments
    args = {}
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

    # directory = '/reza/projects/trajectory-prediction/codes/predicted_environment/'
    # model_directory = 'models/highwaynet-10-sec-101-80-speed-maneuver-for-GT-six-maneuvers/cslstm_m.tar'

    directory = '/Users/louis/cee497projects/trajectory-prediction/codes/predicted_environment/'
    #directory = 'cee497projects/trajectory-prediction/codes/predicted_environment/'

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
    train_loss = []
    val_loss = []
    prev_val_loss = math.inf
    net.train_flag = False

    ######################### Saving data ##############################################################################
    data_points = []
    fut_predictions = []
    lat_predictions = []
    lon_predictions = []
    maneuver_predictions = []
    num_points = 0
    
    # print(f'Length of the pred data loader: {len(predDataloader)}')
    # 6 movements, each movement has probability distributions
    # Straight, Accel, Straight, Decel, Right, Decel, Left, Decl

    for i, data  in enumerate(predDataloader):
        print(f'Index of Data: {i}')
        if i == 100: # we are just going to stop at index 100 for testing 
            break 
        st_time = time.time()
        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, points, maneuver_enc  = data        

        if args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()
            maneuver_enc = maneuver_enc.cuda()

        # Forward pass
        fut_pred, maneuver_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
        fut_pred_max = torch.zeros_like(fut_pred[0])
        for k in range(maneuver_pred.shape[0]):
            indx = torch.argmax(maneuver_pred[k, :]).detach()
            fut_pred_max[:, k, :] = fut_pred[indx][:, k, :]
        l, c = maskedMSETest(fut_pred_max, fut, op_mask)

        lossVals += l.detach()
        counts += c.detach()

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

        ##END OF DEBUG

        points_np = points.numpy() # convert to numpy arrays 
        fut_pred_np = [] # store the future pred points 
        for k in range(6): #manuevers
            fut_pred_np_point = fut_pred[k].clone().detach().cpu().numpy()
            fut_pred_np.append(fut_pred_np_point)
        fut_pred_np = np.array(fut_pred_np)

        for j in range(points_np.shape[0]):
            point = points_np[j]
            print(f'point.shape should be (49,): {point.shape}')
            print(f'point: {point}')
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
                print(f'muX: {muX}')
                print(f'muY: {muY}')
                print(f'sigX: {sigX}')
                print(f'sigY: {sigY}')
            ###END OF DEBUG

            fut_predictions.append(fut_pred_point)
            maneuver_m = maneuver_pred[j].detach().to(device).numpy()
            maneuver_predictions.append(maneuver_m)

            num_points += 1
            if num_points%10000 == 0:
                print('point: ', num_points)
                print('point.shape should be (49,): ', point.shape)
                print('fut_pred_point.shape should be (6,t_f//d_s,5): ', fut_pred_point.shape)
                print('maneuver_m.shape should be (6,):', maneuver_m.shape)

    
    # Print Test Error
    print('MSE: ', lossVals / counts)
    print('RMSE: ', torch.pow(lossVals / counts,0.5))   # Calculate RMSE, feet
    print('number of data points: ', num_points)

    with open(directory+saving_directory+"data_points.data", "wb") as filehandle:
        pickle.dump(np.array(data_points), filehandle, protocol=4)

    with open(directory+saving_directory+"fut_predictions.data", "wb") as filehandle:
        pickle.dump(np.array(fut_predictions), filehandle, protocol=4)

    with open(directory+saving_directory+"maneuver_predictions.data", "wb") as filehandle:
        pickle.dump(np.array(maneuver_predictions), filehandle, protocol=4)

 