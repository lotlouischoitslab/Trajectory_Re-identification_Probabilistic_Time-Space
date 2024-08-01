import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plot_original_trajectories_raw():
    lane = -2
    file_to_read = 'I294_Cleaned.csv'  # TGSIM csv dataset 
    df = pd.read_csv(file_to_read)  # Read in the data 
    input_data = df.copy()  # Copy the dataframe  
    overpass_start_loc_y, overpass_end_loc_y = 1800, 1815  # Both in meters Overpass width 15 meters (50 feet)  
    input_data = input_data[input_data['lane'] == lane].reset_index(drop=True)  # Filter data for the given lane
    incoming_trajectories = input_data[input_data['xloc'] <= overpass_start_loc_y]  # Incoming trajectory before overpass  
    outgoing_trajectories = input_data[input_data['xloc'] >= overpass_end_loc_y]  # Groundtruth trajectory after the overpass  
    incoming_IDs = incoming_trajectories['ID'].unique()
    outgoing_IDs = outgoing_trajectories['ID'].unique()

    axis_coordinates = ['xloc', 'yloc']

    for axis_temp in axis_coordinates:
        fig, ax = plt.subplots(figsize=(20, 10))  # get xs and ts of each vehicle

        for i in incoming_IDs:
            temp_data = incoming_trajectories[incoming_trajectories['ID'] == i]
            ys = temp_data[axis_temp].to_numpy() 
            ts = temp_data['time'].to_numpy()
            speeds = temp_data['speed'].to_numpy()
            norm = Normalize(vmin=speeds.min(), vmax=speeds.max())  # Normalize speed values
            colors = plt.cm.viridis(norm(speeds))  # Use the 'viridis' colormap for more diverse colors
            scatter = ax.scatter(ts, ys, s=1, c=colors)

        for j in outgoing_IDs:
            temp_data = outgoing_trajectories[outgoing_trajectories['ID'] == j]
            ys = temp_data[axis_temp].to_numpy() 
            ts = temp_data['time'].to_numpy()
            speeds = temp_data['speed'].to_numpy()
            norm = Normalize(vmin=speeds.min(), vmax=speeds.max())  # Normalize speed values
            colors = plt.cm.viridis(norm(speeds))  # Use the 'viridis' colormap for more diverse colors
            scatter = ax.scatter(ts, ys, s=1, c=colors)

        # Highlight the missing gap
        ax.axhspan(overpass_start_loc_y, overpass_end_loc_y, color='black', alpha=0.3, label='Overpass')

        # Adding a color bar
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Speed (m/s)', fontsize=20)
        cbar.ax.tick_params(labelsize=20)  # Increase color bar tick size

        if axis_temp == 'xloc':
            ax.set_xlim(0, 320)
            ax.set_ylim(1000, 2200)  # Set y-axis range from 1000 to 2200

        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('Location (m)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20) 

        fig.savefig(f'trajectory_plots/trajectory-{axis_temp}.png') # Save the figure 

def plot_original_trajectories_overpass():
    lane = -2
    file_to_read = 'I294_Cleaned.csv'  # TGSIM csv dataset 
    df = pd.read_csv(file_to_read)  # Read in the data 
    input_data = df.copy()  # Copy the dataframe  
    input_data = input_data[input_data['lane'] == lane].reset_index(drop=True)  # Filter data for the given lane
    
    # incoming_trajectories = input_data[input_data['xloc'] <= 2200]  # Incoming trajectory before overpass  
    # outgoing_trajectories = input_data[input_data['xloc'] >= 1000]  # Groundtruth trajectory after the overpass  
    incoming_trajectories = input_data  
    outgoing_trajectories = input_data    
    incoming_IDs = incoming_trajectories['ID'].unique()
    outgoing_IDs = outgoing_trajectories['ID'].unique()

    # overpass_start = [1800,1800,1800,1800,1800,1800,1895,1930,1705,1755,2065,2111,1050,1120,1165]
    # overpass_end = [1805,1810,1815,1820,1825,1830,1910,1945,1720,1770,2080,2126,1065,1135,1180]

    overpass_start = [1570]
    overpass_end = [1585]

    axis_coordinates = ['xloc', 'yloc']

    for axis_temp in axis_coordinates:
        fig, ax = plt.subplots(figsize=(20, 10))  # Adjusted figure size for better visibility

        for i in incoming_IDs:
            temp_data = incoming_trajectories[incoming_trajectories['ID'] == i]
            ys = temp_data[axis_temp].to_numpy() 
            ts = temp_data['time'].to_numpy()
            speeds = temp_data['speed'].to_numpy()
            norm = Normalize(vmin=speeds.min(), vmax=speeds.max())  # Normalize speed values
            colors = plt.cm.viridis(norm(speeds))  # Use the 'viridis' colormap for more diverse colors
            scatter = ax.scatter(ts, ys, s=1, c=colors)

        for j in outgoing_IDs:
            temp_data = outgoing_trajectories[outgoing_trajectories['ID'] == j]
            ys = temp_data[axis_temp].to_numpy() 
            ts = temp_data['time'].to_numpy()
            speeds = temp_data['speed'].to_numpy()
            norm = Normalize(vmin=speeds.min(), vmax=speeds.max())  # Normalize speed values
            colors = plt.cm.viridis(norm(speeds))  # Use the 'viridis' colormap for more diverse colors
            scatter = ax.scatter(ts, ys, s=1, c=colors)

        # Highlight each overpass on the y-axis
        for start, end in zip(overpass_start, overpass_end):
            ax.axhspan(start, end, color='black', alpha=0.3, label='Overpass')

        # Adding a color bar
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Speed (m/s)', fontsize=20)
        cbar.ax.tick_params(labelsize=20)  # Increase color bar tick size

        if axis_temp == 'xloc':
            ax.set_xlim(0, 320)
            ax.set_ylim(1000, 2200)  # Set y-axis range from 1000 to 2200

        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('Location (m)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20) 

        # Save the figure 
        fig.savefig(f'trajectory_plots/trajectory-{axis_temp}_overpass'+str(overpass_start[0])+'.png') 

def main():
    plot_original_trajectories_raw()
    plot_original_trajectories_overpass()

if __name__ == '__main__':
    main()
