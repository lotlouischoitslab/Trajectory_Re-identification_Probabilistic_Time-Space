import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def plot_trajectory_contour(input_data, overpass_start_loc_x, overpass_end_loc_x, lane):
    IDs_to_traverse = input_data['ID'].unique()  # Vehicle IDs that need to be traversed 
    input_data = input_data[input_data['lane'] == lane].reset_index(drop=True)  # Filter data for the given lane

    for ident in IDs_to_traverse:
        vehicle_data = input_data[input_data['ID'] == ident]

        if len(vehicle_data) == 0:
            print(f"No data for vehicle ID {ident} in lane {lane}")
            continue

        # Filter data within the overpass x-coordinate range
        vehicle_data = vehicle_data[(vehicle_data['xloc'] >= overpass_end_loc_x) & (vehicle_data['xloc'] <= overpass_end_loc_x+10)]

        if len(vehicle_data) == 0:
            print(f"No data within the overpass range for vehicle ID {ident}")
            continue

        x = np.linspace(overpass_start_loc_x, overpass_end_loc_x, 100)
        y = np.linspace(vehicle_data['yloc'].min(), vehicle_data['yloc'].max(), 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)

        muX = vehicle_data['xloc'].values
        muY = vehicle_data['yloc'].values
        sigX = np.std(muX)
        sigY = np.std(muY)

        for i in range(len(muX)):
            mean = [muX[i], muY[i]]
            cov = [[sigX**2, 0], [0, sigY**2]]  # Assuming no covariance
            rv = multivariate_normal(mean, cov)
            Z += rv.pdf(np.dstack((X, Y)))

        plt.figure(figsize=(10, 6))
        contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(contour)
        plt.plot(muX, muY, marker='o', color='red', label=f'Trajectory for ID {ident}')  # Plot the trajectory
        plt.xlabel('X - Longitudinal Coordinate')
        plt.ylabel('Y - Lateral Coordinate')
        plt.title(f'Contour Plot for Vehicle ID {ident} in Lane {lane}')
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

    plot_trajectory_contour(input_data, overpass_start_loc_x, overpass_end_loc_x, lane)

if __name__ == '__main__':
    main()
