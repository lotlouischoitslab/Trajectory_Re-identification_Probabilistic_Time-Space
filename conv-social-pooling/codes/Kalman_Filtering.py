import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

def kalman_predict(trajectories, n_steps_ahead=50):
    num_objects = len(trajectories)
    results = [[] for _ in range(num_objects)]
    
    for obj_idx, trajectory in enumerate(trajectories):
        # Ensure there are enough points for Kalman filtering
        if len(trajectory) < 2:
            print(f"Not enough data points for object {obj_idx}")
            continue
        
        # Extract the x and y coordinates
        x_data = trajectory[:, 0]
        y_data = trajectory[:, 1]

        # Calculate velocity and acceleration for each point
        velocities = np.diff(trajectory, axis=0)
        if len(velocities) < 1:
            print(f"Not enough velocity data for object {obj_idx}")
            continue
        accelerations = np.diff(velocities, axis=0)
        if len(accelerations) < 1:
            print(f"Not enough acceleration data for object {obj_idx}")
            continue
        
        # Truncate the x and y data to match the length of the accelerations
        x_data = x_data[:len(accelerations) + 1]
        y_data = y_data[:len(accelerations) + 1]

        # Truncate the velocities to match the length of the accelerations
        velocities = velocities[:len(accelerations)]

        # Initial state [x, y, vx, vy, ax, ay]
        initial_state_mean = [
            x_data[0], y_data[0],
            velocities[0, 0], velocities[0, 1],
            accelerations[0, 0], accelerations[0, 1]
        ]

        # State transition matrix for 2D with acceleration
        dt = 0.1  # Time step, assuming 1 for simplicity
        transition_matrix = [
            [1, 0, dt, 0, 0.5 * dt**2, 0],  # x' = x + vx*dt + 0.5*ax*dt^2
            [0, 1, 0, dt, 0, 0.5 * dt**2],  # y' = y + vy*dt + 0.5*ay*dt^2
            [0, 0, 1, 0, dt, 0],            # vx' = vx + ax*dt
            [0, 0, 0, 1, 0, dt],            # vy' = vy + ay*dt
            [0, 0, 0, 0, 1, 0],             # ax' = ax
            [0, 0, 0, 0, 0, 1]              # ay' = ay
        ]

        # Observation matrix for 2D
        observation_matrix = [
            [1, 0, 0, 0, 0, 0],  # observe x
            [0, 1, 0, 0, 0, 0]   # observe y
        ]

        # Covariance matrices
        transition_covariance = np.eye(6) * 0.1
        observation_covariance = np.eye(2) * 0.1
        initial_state_covariance = np.eye(6) * 1.0

        # Create the Kalman Filter
        kf = KalmanFilter(
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance
        )

        # Use the Kalman Filter to estimate the states
        observations = np.column_stack((x_data, y_data))
        state_means, _ = kf.filter(observations)

        # Extract the estimated positions
        estimated_positions_x = state_means[:, 0]
        estimated_positions_y = state_means[:, 1]

        # Predict the next n_steps_ahead points
        future_state_means = state_means[-1]

        future_positions = []

        for _ in range(n_steps_ahead):
            future_state_means = np.dot(transition_matrix, future_state_means)
            future_positions.append([future_state_means[0], future_state_means[1]])

        results[obj_idx] = future_positions
    
    return np.array(results)

def plot_trajectories(predicted, actual, vehicle_id):
    predicted_x = predicted[:, 0]
    actual_x = actual['xloc'].values[:len(predicted_x)]
    time_step = np.arange(len(predicted_x))
    
    plt.figure(figsize=(16, 10))
    plt.plot(time_step, predicted_x, marker='o', color='red', label='Kalman Predicted')
    plt.plot(time_step, actual_x, marker='x', color='blue', label='Actual')
    plt.xlabel('Time (s)', fontsize=30)
    plt.ylabel('X Coordinates (m)', fontsize=30) 
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.savefig(f'kalman/kalman_vehicle_{vehicle_id}.png')

def analyze_trajectories(overpass_start_loc_x, overpass_end_loc_x):  
    delta = 5  # Set the delta as needed for the time duration after the overpass
    steps = overpass_end_loc_x - overpass_start_loc_x 
    input_data = pd.read_csv('I294_Cleaned.csv')
    lane = -2
    input_data = input_data[input_data['lane'] == lane].reset_index(drop=True)  # Filter data for the given lane
    incoming_trajectories = input_data[input_data['xloc'] <= overpass_start_loc_x] # Incoming trajectory before overpass  
    unique_ids = incoming_trajectories['ID'].unique()  
   
    outgoing_trajectories = input_data[(input_data['xloc'] >= overpass_end_loc_x)] # Groundtruth trajectory after the overpass  
    outgoing_ids = outgoing_trajectories['ID'].unique() 

    correct_predictions = [] 

    for temp_id in unique_ids:
        ground_truth_trajectory = outgoing_trajectories[outgoing_trajectories['ID'] == temp_id]
        ground_truth_x = ground_truth_trajectory['xloc'].values
        temp_incoming = incoming_trajectories[incoming_trajectories['ID'] == temp_id] 
        overpass_start_time = temp_incoming['time'].values[-1]
        
        # Predict points using the Kalman filter
        kalman_df = temp_incoming[temp_incoming['xloc'] <= overpass_end_loc_x]
        if len(kalman_df) < 2:
            print(f"Not enough points for vehicle ID {temp_id} after filtering.")
            continue

        trajectory = kalman_df[['xloc', 'yloc']].values.reshape(1, -1, 2)
        kalman_predicted_positions = kalman_predict(trajectory, steps)

        if len(kalman_predicted_positions) == 0:
            print(f"Kalman prediction failed for vehicle ID {temp_id}.")
            continue

        kalman_predicted_x = kalman_predicted_positions[0, :, 0]
        
        error = float('inf')
        best_possible_x = None

        print(kalman_predicted_x)


        for second_id in unique_ids:
            possible_trajectory = outgoing_trajectories[(outgoing_trajectories['ID'] == second_id)]
            poss_x = possible_trajectory['xloc'].values 
            
            if len(poss_x) == 0:
                continue  # Skip if there are no points to compare
            
            min_len = min(len(kalman_predicted_x), len(poss_x))
            temp_error = sum((kalman_predicted_x[:min_len] - poss_x[:min_len]) ** 2)
            
            if temp_error < error:
                error = temp_error
                best_possible_x = poss_x[:min_len]

        if best_possible_x is not None:
            min_len = min(len(best_possible_x), len(ground_truth_x))
            for bx, gx in zip(best_possible_x[:min_len], ground_truth_x[:min_len]):
                if bx == gx:
                    correct_predictions.append(1)
                else:
                    correct_predictions.append(0)
            
            # Plot the predicted vs. actual trajectories 
            # plot_trajectories(kalman_predicted_positions[0], ground_truth_trajectory, temp_id)

    correct_predictions_results = sum(correct_predictions)
    accuracy = (correct_predictions_results / len(correct_predictions)) * 100
    accuracy = np.round(accuracy, 2)
    print(f'Accuracy: {accuracy}%')


def main(): 
    overpass_start_loc_x, overpass_end_loc_x = 1800, 1805
    analyze_trajectories(overpass_start_loc_x, overpass_end_loc_x)

    # 1800 to 1805: 51.79% Accuracy
    # 1800 to 1810: % Accuracy
    # 1800 to 1815: % Accuracy
    # 1800 to 1820: % Accuracy
    # 1800 to 1825: % Accuracy
    # 1800 to 1830: % Accuracy

if __name__ == '__main__': 
    main()
