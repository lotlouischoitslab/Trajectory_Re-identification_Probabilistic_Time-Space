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
 
import scipy.integrate as integrate
from scipy.special import erf
import math
   
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
      
 
def generate_normal_distribution(fut_pred, lane, batch_num):
    num_maneuvers = len(fut_pred)
    x = np.linspace(-10, 100, 100)
    y = np.linspace(-10, 10, 100)  # Reduced y-range for better visibility
    X, Y = np.meshgrid(x, y)
    time = 0

    for count in range(10, 51, 10):
        time += 2
        plt.figure(figsize=(12, 8))  # Ensure consistent figure size for combined plots

        combined_Zs = np.zeros((num_maneuvers, X.shape[0], X.shape[1]))  # Reset combined_Z for each time step

        for m in range(num_maneuvers):
            for t in range(1, count + 1):
                muX = fut_pred[m][t - 1, batch_num, 0].detach().numpy()
                muY = fut_pred[m][t - 1, batch_num, 1].detach().numpy()
                sigX = fut_pred[m][t - 1, batch_num, 2].detach().numpy()
                sigY = fut_pred[m][t - 1, batch_num, 3].detach().numpy()

                mean = [muX, muY]
                cov = [[sigX**2, 0], [0, sigY**2]]  # Assuming no covariance
                rv = multivariate_normal(mean, cov)
                combined_Zs[m] += rv.pdf(np.dstack((X, Y)))

        # Plot the combined contour map for all maneuvers
        for m in range(num_maneuvers):
            contour = plt.contourf(X, Y, combined_Zs[m], levels=50, alpha=0.5, cmap='viridis')  # Increased levels for more detail
        plt.xlabel('X - Lateral Coordinate')
        plt.ylabel('Y - Longitudinal Coordinate')
        plt.ylim(-10, 10)  # Set y-limit for better visibility
        plt.title(f'Combined Contour Plot for All Maneuvers at time {time} seconds')
        plt.colorbar(contour)

        plt.tight_layout()
        plt.savefig(f'contour_maps/combined_maneuvers_{time}_seconds.png')
        plt.close()
        

def main():
    args = {}  # Network Arguments
    print('cuda available', torch.cuda.is_available())
    print('torch version', torch.__version__)
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
    args['grid_size'] = (13, 3)
    args['soc_conv_depth'] = 64
    args['conv_3x1_depth'] = 16
    args['dyn_embedding_size'] = 32
    args['input_embedding_size'] = 32
    args['num_lat_classes'] = 3
    args['num_lon_classes'] = 2
    args['use_maneuvers'] = True
    args['train_flag'] = False

    filepath_pred_Set = 'cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10_seconds/test'  # HAL GPU Cluster
    file_to_read = 'I294_Cleaned.csv'
    
    df = pd.read_csv(file_to_read)  # read in the data 
    original_data = df.copy()  # copy the dataframe  
    temp_lane = -2 
    lanes_to_analyze = [temp_lane]  # lanes to analyze  
    print(f'Unique lanes: {lanes_to_analyze}') 
    
    batch_size = 1024  # batch size for the model and choose from [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    overpass_start_loc_x, overpass_end_loc_x = 1800, 1817  # both in meters 
    delta = 5  # time interval that we will be predicting for 
  
    net = highwayNet_six_maneuver(args)  # initialize the network 
    model_path = 'trained_model_TGSIM/cslstm_m.tar'
    net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))  # load the model onto the local machine 

    if args['use_cuda']: 
        net = net.to(device)

    predSet = tgsimDataset(filepath_pred_Set, t_h=30, t_f=100, d_s=2)
    predDataloader = DataLoader(predSet, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=predSet.collate_fn)
    lossVals = torch.zeros(50).to(device)
    counts = torch.zeros(50).to(device)

    fut_predictions = []  # future prediction values 
    maneuver_predictions = []  # maneuver prediction values 

    print(f'Length of the pred data loader: {len(predDataloader)}')

    for lane in lanes_to_analyze:  # for each lane to be analyzed 
        print(f'Lane: {lane}')  # print the lane   
        for i, data in enumerate(predDataloader):  # for each index and data in the predicted data loader 
        
            print(f'Index of Data: {i}/{len(predDataloader)}')  # print out the index of the current data to be analyzed 
            
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, maneuver_enc = data  # unpack the data   

            if args['use_cuda']:
                hist = hist.cuda()
                nbrs = nbrs.cuda()
                mask = mask.cuda()
                lat_enc = lat_enc.cuda()
                lon_enc = lon_enc.cuda()
                fut = fut.cuda()
                op_mask = op_mask.cuda()
                maneuver_enc = maneuver_enc.cuda()

            fut_pred, maneuver_pred = net(hist, nbrs, mask, lat_enc, lon_enc)  # feed the parameters into the neural network for forward pass
            fut_pred_max = torch.zeros_like(fut_pred[0])  # get the max predicted values 

            for k in range(maneuver_pred.shape[0]):  # for each value in the maneuver predicted shapes
                indx = torch.argmax(maneuver_pred[k, :]).detach()  # get the arg max of the maneuvered prediction values
                fut_pred_max[:, k, :] = fut_pred[indx][:, k, :]  # future predicted value max  

            fut_pred_np = []  # store the future pred points 

            for k in range(6):  # maneuvers mean the 
                fut_pred_np_point = fut_pred[k].clone().detach().cpu().numpy()
                fut_pred_np.append(fut_pred_np_point)

            fut_pred_np = np.array(fut_pred_np)  # convert the fut pred points into numpy
            if i == 0: # Generate and save the distribution plots just for one trajectory
                generate_normal_distribution(fut_pred,lane ,batch_size-1)
                break
              
    

if __name__ == '__main__': # run the code
    main() # call the main function 


  