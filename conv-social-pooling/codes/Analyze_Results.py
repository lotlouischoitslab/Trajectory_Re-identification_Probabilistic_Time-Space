 



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

import re

def extract_number_from_filename(filename):
    match = re.search(r'\d+', filename)
    if match:
        return match.group()
    else:
        return None

    
def analyze_trajectories():  
    incoming_trajectories = pd.read_csv('before/incoming.csv')
    unique_ids = incoming_trajectories['ID'].unique() 


    best_trajectories_dir = "best_trajectories"
    best_trajectories_files = os.listdir(best_trajectories_dir)
    possible_trajectories_dir = "possible_trajectories"
    possible_trajectories_files = os.listdir(possible_trajectories_dir)
    outgoing_trajectories = pd.read_csv('before/outgoing.csv')
    
    correct_predictions = [] 

    for temp_id in unique_ids:
        temp_incoming = incoming_trajectories[incoming_trajectories['ID']==temp_id]
        average_speed = np.mean(temp_incoming['speed'].values[-2:])
        print(f'average speed: {average_speed} m/s')



    # for possible_trajectory_input_file in possible_trajectories_files:  
    #     print(possible_trajectory_input_file)
    #     vehicle_ID = int(extract_number_from_filename(possible_trajectory_input_file))
        
    #     ground_truth_trajectory = outgoing_trajectories[outgoing_trajectories['ID'] == vehicle_ID]
    #     ground_truth_x = ground_truth_trajectory['xloc'].values[:10]
    #     ground_truth_y = ground_truth_trajectory['yloc'].values[:10]

    #     if len(ground_truth_x) == 0:
    #         continue

    #     possible_trajectory_input_path = os.path.join(possible_trajectories_dir, possible_trajectory_input_file)
    #     possible_trajectory_file = pd.read_csv(possible_trajectory_input_path)
    #     ID_to_check = possible_trajectory_file['ID'].unique()

    #     error = float('inf') 
    #     store_x = []
    #     store_y = []

    #     for temp_id in ID_to_check:
    #         poss_temp = possible_trajectory_file[possible_trajectory_file['ID'] == temp_id]
    #         poss_temp_x = poss_temp['xloc'].values[:10]
    #         poss_temp_y = poss_temp['yloc'].values[:10]
    #         print(len(ground_truth_x),len(poss_temp_x),len(ground_truth_y),len(poss_temp_y))

    #         # Calculate the error between ground truth and possible trajectory
    #         temp_error = sum((ground_truth_x[:len(poss_temp_x)] - poss_temp_x)**2 + (ground_truth_y[:len(poss_temp_y)] - poss_temp_y)**2)
            
    #         if temp_error < error:
    #             error = temp_error
    #             store_x = poss_temp_x
    #             store_y = poss_temp_y

    #     # if store_x == ground_truth_x and store_y == ground_truth_y:  # Define some_threshold as per your requirement
    #     if np.array_equal(store_x, ground_truth_x[:len(poss_temp_x)]) and np.array_equal(store_y, ground_truth_y[:len(poss_temp_y)]):
    #         correct_predictions.append(vehicle_ID)

    # accuracy = 100*(len(correct_predictions) / len(possible_trajectories_files))
    # print(f'Accuracy: {accuracy:.2f}%')


def straight_line_method():
    pass 
   
 
def main(): # Main function 
    analyze_trajectories()
  

if __name__ == '__main__': # run the code
    main() # call the main function 