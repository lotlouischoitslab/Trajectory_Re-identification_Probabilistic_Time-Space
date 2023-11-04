import numpy as np
import pandas as pd
import pickle
import math

# Define directories and filenames
directory = 'cee497projects/trajectory-prediction/codes/predicted_environment/'
directory = '/Users/louis/cee497projects/data/101-80-speed-maneuver-for-GT/train/'
directory = '/Users/louis/cee497projects/trajectory-prediction/codes/predicted_environment/predicted_data/highwaynet-10-sec-101-80-speed-maneuver-for-GT-six-maneuvers/'

train_traj_x = 'train_trajectory_x.data'
train_traj_y = 'train_trajectory_y.data'
filenamex = directory + train_traj_x
filenamey = directory + train_traj_y

filename = directory + 'output_results.data'

def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

loaded_data = load_from_pickle(filename)
print(loaded_data)

# loaded_datax = load_from_pickle(filenamex)
# print(loaded_datax.shape)
# print(len(loaded_datax[0]))

# loaded_datay = load_from_pickle(filenamey)
# print(loaded_datay.shape)
# print(len(loaded_datay[0]))