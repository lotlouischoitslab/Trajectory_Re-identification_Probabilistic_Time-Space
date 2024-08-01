import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

def kalman_predict(trajectory, n_steps_ahead, dt=0.1):
    x_data = trajectory['xloc'].values
    y_data = trajectory['yloc'].values
    
    if len(x_data) < 2:
        print("Not enough data points for Kalman filtering.")
        return np.array([])
    
    velocities = np.diff(np.column_stack((x_data, y_data)), axis=0)
    velocities = np.vstack((velocities, velocities[-1]))

    initial_state_mean = [x_data[0], y_data[0], velocities[0, 0], velocities[0, 1]]

    transition_matrix = [
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

    observation_matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ]

    transition_covariance = np.eye(4) * 0.5  # Adjusted value for better accuracy
    observation_covariance = np.eye(2) * 0.25  # Adjusted value for better accuracy
    initial_state_covariance = np.eye(4) * 0.25

    kf = KalmanFilter(
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance
    )

    observations = np.column_stack((x_data, y_data))
    state_means, _ = kf.filter(observations)

    future_state_means = state_means[-1]
    future_positions = []

    for _ in range(n_steps_ahead):
        future_state_means = np.dot(transition_matrix, future_state_means)
        future_positions.append([future_state_means[0], future_state_means[1]])

    return np.array(future_positions)

def kalman_predict(trajectory, n_steps_ahead, dt=0.1):
    x_data = trajectory['xloc'].values
    y_data = trajectory['yloc'].values
    
    if len(x_data) < 2:
        print("Not enough data points for Kalman filtering.")
        return np.array([])
    
    velocities = np.diff(np.column_stack((x_data, y_data)), axis=0)
    velocities = np.vstack((velocities, velocities[-1]))
    
    # Assuming cubic behavior, let's include acceleration and jerk (third derivative of position)
    accelerations = np.diff(velocities, axis=0)
    accelerations = np.vstack((accelerations, accelerations[-1]))
    
    initial_state_mean = [x_data[0], y_data[0], velocities[0, 0], velocities[0, 1], accelerations[0, 0], accelerations[0, 1]]

    transition_matrix = [
        [1, 0, dt, 0, 0.5 * dt**2, 0],  # x' = x + vx*dt + 0.5*ax*dt^2
        [0, 1, 0, dt, 0, 0.5 * dt**2],  # y' = y + vy*dt + 0.5*ay*dt^2
        [0, 0, 1, 0, dt, 0],            # vx' = vx + ax*dt
        [0, 0, 0, 1, 0, dt],            # vy' = vy + ay*dt
        [0, 0, 0, 0, 1, 0],             # ax' = ax
        [0, 0, 0, 0, 0, 1]              # ay' = ay
    ]

    observation_matrix = [
        [1, 0, 0, 0, 0, 0],  # observe x
        [0, 1, 0, 0, 0, 0]   # observe y
    ]

    transition_covariance = np.eye(6) * 0.8 # Adjusted value for better accuracy
    observation_covariance = np.eye(2) * 0.3  # Adjusted value for better accuracy
    initial_state_covariance = np.eye(6) * 0.6

    kf = KalmanFilter(
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance
    )

    observations = np.column_stack((x_data, y_data))
    state_means, _ = kf.filter(observations)

    future_state_means = state_means[-1]
    future_positions = []

    for _ in range(n_steps_ahead):
        future_state_means = np.dot(transition_matrix, future_state_means)
        future_positions.append([future_state_means[0], future_state_means[1]])

    return np.array(future_positions)

def plot_trajectories(predicted, actual, vehicle_id, overpass_start_time, delta, overpass_end_loc_x):
    predicted_x = predicted[:, 0]
    actual_x = actual['xloc'].values
    
    if len(actual_x) == 0:
        print(f"No actual data available for vehicle ID {vehicle_id}")
        return
    
    time_step_predicted = np.linspace(overpass_start_time, overpass_start_time + delta, len(predicted_x))
    time_step_actual = np.linspace(overpass_start_time, overpass_start_time + delta, len(actual_x))

    time_step_actual = actual['time'].values 
    
    plt.figure(figsize=(16, 10))
    plt.plot(time_step_actual, actual_x, marker='x', color='blue', label='Actual')
    plt.plot(time_step_predicted, predicted_x, marker='o', color='red', label='Kalman Predicted')
    
    plt.xlabel('Time (s)', fontsize=30)
    plt.ylabel('X Coordinates (m)', fontsize=30) 
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.savefig(f'kalman/{overpass_end_loc_x}_kalman_vehicle_{vehicle_id}.png')


def analyze_trajectories(overpass_start_loc_x, overpass_end_loc_x):
    input_data = pd.read_csv('I294_Cleaned.csv')
    lane = -2
    delta = 5
    input_data = input_data[input_data['lane'] == lane].reset_index(drop=True)  # Filter data for the given lane
    incoming_trajectories = input_data[input_data['xloc'] <= overpass_start_loc_x]  # Incoming trajectory before overpass  
    unique_ids = incoming_trajectories['ID'].unique()
   
    outgoing_trajectories = input_data[(input_data['xloc'] >= overpass_end_loc_x)]  # Groundtruth trajectory after the overpass  
    outgoing_ids = outgoing_trajectories['ID'].unique()

    correct_predictions = []
    traversed_data = []

    for temp_id in unique_ids:
        ground_truth_trajectory = outgoing_trajectories[outgoing_trajectories['ID'] == temp_id]
        ground_truth_x = ground_truth_trajectory['xloc'].values
        temp_incoming = incoming_trajectories[incoming_trajectories['ID'] == temp_id]
        overpass_start_time = temp_incoming['time'].values[-1]
        
        kalman_df = temp_incoming[temp_incoming['xloc'] <= overpass_start_loc_x]
        if len(kalman_df) < 2:
            print(f"Not enough points for vehicle ID {temp_id} after filtering.")
            continue

        # Set the number of steps to predict into the future
        steps = 50  # Adjust the number of steps to predict as needed
        kalman_predicted_positions = kalman_predict(kalman_df, steps) 

        kalman_predicted_x = kalman_predicted_positions[:, 0]
        kalman_predicted_x = kalman_predicted_x[kalman_predicted_x >= overpass_end_loc_x]
        
        error = float('inf')
        best_possible_x = None

        for second_id in outgoing_ids:
            possible_trajectory = outgoing_trajectories[
                (outgoing_trajectories['ID'] == second_id) & 
                (outgoing_trajectories['time'] >= overpass_start_time) 
            ]
            poss_x = possible_trajectory['xloc'].values[:10]

            if len(poss_x) == 0: 
                temp_error = float('inf')
            else:
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
            
            plot_trajectories(kalman_predicted_positions, ground_truth_trajectory, temp_id, overpass_start_time, delta, overpass_end_loc_x)

        # Save traversed data
        if best_possible_x is not None:
            for bx in best_possible_x:
                traversed_data.append([temp_id, bx])

    # Save traversed data to a CSV file
    traversed_df = pd.DataFrame(traversed_data, columns=['Vehicle_ID', 'xloc'])
    traversed_df.to_csv('traversed_data.csv', index=False)

    correct_predictions_results = sum(correct_predictions)
    if len(correct_predictions) > 0:
        accuracy = (correct_predictions_results / len(correct_predictions)) * 100
    else:
        accuracy = 0
    accuracy = np.round(accuracy, 2)
    print(f'Accuracy: {accuracy}%')
    return accuracy

def main(): 
    overpass_start = [1800,1800,1800,1800,1800,1800,1895,1930,1705,1755,2065,2111,1050,1120,1165]
    overpass_end  =  [1805,1810,1815,1820,1825,1830,1910,1945,1720,1770,2080,2126,1065,1135,1180]

   
    accuracies = []

    for start, end in zip(overpass_start, overpass_end):
        acc = analyze_trajectories(start, end)
        accuracies.append(acc) 
        #break 

    print(accuracies)

if __name__ == '__main__': 
    main()
 

