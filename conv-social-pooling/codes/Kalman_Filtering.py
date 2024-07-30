import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

def kalman_predict(trajectory, n_steps_ahead, dt=0.1):
    # Extract the x and y coordinates
    x_data = trajectory['xloc'].values
    y_data = trajectory['yloc'].values
    
    # Ensure there are enough points for Kalman filtering
    if len(x_data) < 2:
        print("Not enough data points for Kalman filtering.")
        return np.array([])
    
    # Calculate velocity for each point
    velocities = np.diff(np.column_stack((x_data, y_data)), axis=0)
    velocities = np.vstack((velocities, velocities[-1]))  # Repeat the last velocity

    # Initial state [x, y, vx, vy]
    initial_state_mean = [x_data[0], y_data[0], velocities[0, 0], velocities[0, 1]]

    # State transition matrix for 2D
    transition_matrix = [
        [1, 0, dt, 0],  # x' = x + vx*dt
        [0, 1, 0, dt],  # y' = y + vy*dt
        [0, 0, 1, 0],   # vx' = vx
        [0, 0, 0, 1]    # vy' = vy
    ]

    # Observation matrix for 2D
    observation_matrix = [
        [1, 0, 0, 0],  # observe x
        [0, 1, 0, 0]   # observe y
    ]

    # Covariance matrices
    transition_covariance = np.eye(4) * 0.05  # Process noise
    observation_covariance = np.eye(2) * 0.1  # Measurement noise
    initial_state_covariance = np.eye(4) * 0.5  # Initial state covariance

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

    return np.array(future_positions)

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
    plt.savefig(f'prediction_plots/kalman_vehicle_{vehicle_id}.png')

def analyze_trajectories():  
    overpass_start_loc_x, overpass_end_loc_x = 1800, 1805
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
        kalman_df = temp_incoming[temp_incoming['xloc'] <= overpass_start_loc_x]
        if len(kalman_df) < 2:
            print(f"Not enough points for vehicle ID {temp_id} after filtering.")
            continue

        kalman_predicted_positions = kalman_predict(kalman_df, steps)

        if len(kalman_predicted_positions) == 0:
            print(f"Kalman prediction failed for vehicle ID {temp_id}.")
            continue

        kalman_predicted_x = kalman_predicted_positions[:, 0]
        
        error = float('inf')
        best_possible_x = None

        for second_id in unique_ids:
            possible_trajectory = outgoing_trajectories[(outgoing_trajectories['ID'] == second_id) & (outgoing_trajectories['time'] >= overpass_start_time)]
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
            for bx,gx in zip(best_possible_x[:min_len], ground_truth_x[:min_len]):
                if bx == gx:
                    correct_predictions.append(1)
                else:
                    correct_predictions.append(0)
            
            # Plot the predicted vs. actual trajectories 
            #plot_trajectories(kalman_predicted_positions, ground_truth_trajectory, temp_id)

    correct_predictions_results = sum(correct_predictions)
    accuracy = (correct_predictions_results / len(correct_predictions)) * 100
    accuracy = np.round(accuracy, 2)
    print(f'Accuracy: {accuracy}%')

def main(): 
    analyze_trajectories()

if __name__ == '__main__': 
    main()
