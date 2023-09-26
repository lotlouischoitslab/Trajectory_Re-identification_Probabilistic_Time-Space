import numpy as np 
import pandas as pd 

filepath_pred_Set = '/Users/louis/cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10-seconds/'
file_test = 'test_trajectory.data'

filename = filepath_pred_Set + file_test

import pickle
import csv

with open(filename, 'rb') as file:
    data = pickle.load(file)

# If the data is not already in the desired format (numpy array or torch tensor), convert it:
if isinstance(data, np.ndarray):
    pass  # Already a numpy array
elif torch.is_tensor(data):
    data = data.numpy()  # Convert torch tensor to numpy array
else:
    data = np.array(data)  # Convert other data types to numpy array


csv_filename = 'output.csv'

# Using numpy
np.savetxt(csv_filename, data, delimiter=',', fmt='%s')

# Or using pandas
df = pd.DataFrame(data)
df.to_csv(csv_filename, index=False)
 


