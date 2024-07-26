import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_trajectory_contour(input_data, overpass_start_loc_x, overpass_end_loc_x, lane,delta):
    IDs_to_traverse = input_data['ID'].unique()  # Vehicle IDs that need to be traversed 
    input_data = input_data[input_data['lane'] == lane].reset_index(drop=True)  # Filter data for the given lane

    for ident in IDs_to_traverse:
        vehicle_data = input_data[input_data['ID'] == ident]

        if len(vehicle_data) == 0:
            print(f"No data for vehicle ID {ident} in lane {lane}")
            continue

        # Filter data within the overpass x-coordinate range
        vehicle_data = vehicle_data[(vehicle_data['xloc'] >= overpass_end_loc_x) & (vehicle_data['xloc'] <= overpass_end_loc_x+delta)]

        if len(vehicle_data) == 0:
            print(f"No data within the overpass range for vehicle ID {ident}")
            continue

        x = vehicle_data['xloc'].values
        y = vehicle_data['yloc'].values

        # Create a 2D histogram
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=30, range=[[overpass_end_loc_x, overpass_end_loc_x+delta+0.5], [y.min(), y.max()+0.05]])

        # Generate the mesh grid for the contour plot
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

        plt.figure(figsize=(10, 6))
        contour = plt.contourf(X, Y, heatmap.T, levels=60, cmap='viridis')
        plt.colorbar(contour)
        plt.plot(x, y, marker='o', color='red', label=f'Trajectory for ID {ident}')  # Plot the trajectory
        plt.xlabel('X - Longitudinal Coordinate')
        plt.ylabel('Y - Lateral Coordinate')
        # plt.title(f'Contour Plot for Vehicle ID {ident} in Lane {lane}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'contour_maps/vehicle_{ident}_contour.png')
        plt.close()
        break

def main():
    # Load your data here
    input_data = pd.read_csv('before/outgoing.csv')
    overpass_start_loc_x = 1800
    overpass_end_loc_x = 1815
    lane = -2
    delta = 10

    plot_trajectory_contour(input_data, overpass_start_loc_x, overpass_end_loc_x, lane, delta)

if __name__ == '__main__':
    main()
