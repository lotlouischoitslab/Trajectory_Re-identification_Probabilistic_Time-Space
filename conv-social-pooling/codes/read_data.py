import numpy as np
import pandas as pd
import pickle
import csv

# Define directories and filenames
trajectories_directory = 'cee497projects/data/101-80-speed-maneuver-for-GT/10_seconds/train/' 

 

filename = trajectories_directory + 'train.data'

with open(filename, 'rb') as file:
    data = pickle.load(file)

with open('temp_write/data.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the data to the csv file
    writer.writerows(data)

 

# loaded_datax = load_from_pickle(filenamex)
# print(loaded_datax.shape)
# print(len(loaded_datax[0]))

# loaded_datay = load_from_pickle(filenamey)
# print(loaded_datay.shape)
# print(len(loaded_datay[0]))