import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize

def plot_original_trajectories():
    lane = -2
    file_to_read = 'I294_Cleaned.csv'  # TGSIM csv dataset 
    df = pd.read_csv(file_to_read) # read in the data 
    input_data = df.copy() # copy the dataframe  
    overpass_start_loc_x, overpass_end_loc_x = 1800, 1817 # both in meters Overpass width 17 meters (56 feet) 
    overpass_length = overpass_end_loc_x - overpass_start_loc_x # length of the overpass
    input_data = input_data[input_data['lane'] == lane].reset_index(drop=True)  # Filter data for the given lane
    incoming_trajectories = input_data[input_data['xloc'] <= overpass_start_loc_x] # Incoming trajectory before overpass  
    outgoing_trajectories_temp = input_data[(input_data['xloc'] >= overpass_end_loc_x) & (input_data['xloc'] <= overpass_end_loc_x + overpass_length)] # Groundtruth trajectory after the overpass  
    possible_trajectories = input_data[(input_data['xloc'] >= overpass_end_loc_x) & (input_data['xloc'] <= overpass_end_loc_x + overpass_length)] # All possible trajectories that we need to consider

    outgoing_trajectories = input_data[input_data['xloc'] >= overpass_end_loc_x] # Groundtruth trajectory after the overpass  
    incoming_IDs = incoming_trajectories['ID'].unique()
    outgoing_IDs = outgoing_trajectories['ID'].unique()

    axis_coordinates = ['xloc', 'yloc']

    for axis_temp in axis_coordinates:
        fig, ax = plt.subplots() # get xs and ts of each vehicle

        for i in incoming_IDs:
            temp_data = incoming_trajectories[incoming_trajectories['ID'] == i]
            ys = temp_data[axis_temp].to_numpy() 
            ts = temp_data['time'].to_numpy()
            speeds = temp_data['speed'].to_numpy()
            norm = Normalize(vmin=speeds.min(), vmax=speeds.max())  # Normalize speed values
            colors = plt.cm.viridis(norm(speeds))  # Use the 'viridis' colormap for more diverse colors
            ax.scatter(ts, ys, s=1, c=colors) 

        for j in outgoing_IDs:
            temp_data = outgoing_trajectories[outgoing_trajectories['ID'] == j]
            ys = temp_data[axis_temp].to_numpy() 
            ts = temp_data['time'].to_numpy()
            speeds = temp_data['speed'].to_numpy()
            norm = Normalize(vmin=speeds.min(), vmax=speeds.max())  # Normalize speed values
            colors = plt.cm.viridis(norm(speeds))  # Use the 'viridis' colormap for more diverse colors
            ax.scatter(ts, ys, s=1, c=colors)

        if axis_temp == 'xloc':
            ax.set_xlim(0, 320)
            ax.set_ylim(1000, 2200)  # Set y-axis range from 1000 to 2200

        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('Location (m)', fontsize=20)
        ax.xaxis.set_major_locator(plt.MaxNLocator(100)) # Increase the number of grid lines on the x-axis 
        ax.yaxis.set_major_locator(plt.MaxNLocator(60)) # Increase the number of grid lines on the y-axis
        ax.grid()

        fig.set_size_inches(80, 30)
        fig.savefig(f'trajectory_plots/trajectory-{axis_temp}.png')


def plot_raw_trajectories():
    smoothed_file = pd.read_csv('I294_Cleaned.csv') # read in the data 
    lanes = sorted(smoothed_file['lane'].unique()) # lanes to analyze

    for lane in lanes:
        lane_data = smoothed_file[smoothed_file['lane']==lane].reset_index(drop=True)

        IDs = []
        all_ts = []
        all_ys = []
        init_ID = -1

        # get all vehicle IDs
        for i in range(len(lane_data)):
            if lane_data['ID'][i] != init_ID:
                IDs.append(lane_data['ID'][i])
                init_ID = lane_data['ID'][i]

        # get xs and ts of each vehicle
        fig, ax = plt.subplots()
        for j in IDs:
            temp_data = lane_data[lane_data['ID']==j]
            ys = temp_data['xloc'].to_numpy()
            #ys = temp_data['yloc'].to_numpy()
            ts = temp_data.time.to_numpy()
            ax.scatter(ts, ys,s=1)
            ax.text(ts[0], ys[0], str(j))

        ax.set_xlabel('Time (s)', fontsize = 20)
        ax.set_ylabel('Location (m)', fontsize = 20)
        ax.xaxis.set_major_locator(plt.MaxNLocator(100)) # Increase the number of grid lines on the x-axis 
        ax.yaxis.set_major_locator(plt.MaxNLocator(60)) # Increase the number of grid lines on the y-axis
        ax.grid()

        fig.set_size_inches(120,30)
        fig.savefig(f'lane_plots/Lane_{lane}-x.png')

def main():
    plot_original_trajectories()
    plot_raw_trajectories()

if __name__ == '__main__':
    main()
