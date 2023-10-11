import numpy as np
import pandas as pd
import pickle
import math

# Define directories and filenames
directory = 'cee497projects/trajectory-prediction/codes/predicted_environment/'
directory = '/Users/louis/cee497projects/trajectory-prediction/codes/predicted_environment/'
saving_directory = 'predicted_data/highwaynet-10-sec-101-80-speed-maneuver-for-GT-six-maneuvers/'
file_test = 'fut_predictions.data'
filename = directory + saving_directory + file_test
 

def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

 
loaded_data = load_from_pickle(filename)
print(loaded_data.shape)
print(loaded_data)
