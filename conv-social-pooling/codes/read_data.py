import numpy as np
import pandas as pd
import pickle
import csv

# Define directories and filenames
trajectories_directory = '/Users/louis/cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10-seconds/train/'
file_name = trajectories_directory + 'train'

# with open(filename, 'rb') as file:
#     data = pickle.load(file)

# with open('temp_write/traj_data.csv', 'w', newline='') as file:
#     writer = csv.writer(file)

#     # Write the data to the csv file
#     writer.writerows(data)

# with open(filename, 'rb') as file:
#     data = np.array(pickle.load(file)) 

 
# print(data)

# with open(file_name+'_trajectory.data', 'rb') as filehandle:
#     trajectories = pickle.load(filehandle)


# with open(file_name+'.data', 'rb') as filehandle:
#     data_points = np.array(pickle.load(filehandle))

# with open(file_name+'_trajectory.data', 'rb') as filehandle:
#     data_points = np.array(pickle.load(filehandle))

# # print('trajectories') 
# # print(trajectories)

# print('data points') 
# print(data_points)

# data = pd.DataFrame(data_points)
# data.to_csv('temp_write/data.csv')

filename = 'I294_Cleaned.csv'

data = pd.read_csv(filename) 

matrix = np.array([data['xloc'].values,data['yloc'].values]).T
print(matrix.shape)