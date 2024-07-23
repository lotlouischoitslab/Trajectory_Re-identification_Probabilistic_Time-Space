import os
import pandas as pd
import numpy as np

def straight_line_method(overpass_start_loc_x, average_speed, delta, overpass_start_time):
    time_frame = np.arange(overpass_start_time, overpass_start_time + delta, 0.1)
    adjusted_time_frame = np.arange(0, delta, 0.1)
    x = [overpass_start_loc_x + average_speed * t for t in adjusted_time_frame]
    df = {
        'time': time_frame[:50],
        'adjusted_time': adjusted_time_frame,
        'xloc': x
    }
    print(len(time_frame),len(adjusted_time_frame),len(x))
    return pd.DataFrame(df)

def analyze_trajectories():  
    incoming_trajectories = pd.read_csv('before/incoming.csv')
    unique_ids = incoming_trajectories['ID'].unique() 
    outgoing_trajectories = pd.read_csv('before/outgoing.csv')
    overpass_start_loc_x = 1800
    overpass_end_loc_x = 1817 
    delta = 5  # Set the delta as needed for the time duration after the overpass

    correct_predictions = [] 

    for temp_id in unique_ids:
        ground_truth_trajectory = outgoing_trajectories[outgoing_trajectories['ID'] == temp_id]
        ground_truth_x = ground_truth_trajectory['xloc'].values
        temp_incoming = incoming_trajectories[incoming_trajectories['ID'] == temp_id]
        
        # Calculate average speed of the last two points before the overpass
        average_speed = np.mean(temp_incoming['speed'].values[-2:])
        print(f'average speed: {average_speed} m/s')
        
        overpass_start_time = temp_incoming['time'].values[-1]
        
        # Generate points using the straight line method
        linear_df = straight_line_method(overpass_start_loc_x, average_speed, delta, overpass_start_time)
        linear_df.to_csv('linear_plots/ID'+str(temp_id)+'_slope_possible.csv', index=False)
        
        error = float('inf')
        best_possible_x = None

        linear_x = linear_df[linear_df['xloc'] >= overpass_end_loc_x]['xloc'].values 
        linear_start_time = min(linear_df['time'].values)

        for second_id in unique_ids:
            possible_trajectory = outgoing_trajectories[(outgoing_trajectories['ID'] == second_id) & (outgoing_trajectories['time'] >= linear_start_time)]
            poss_x = possible_trajectory['xloc'].values 
            
            # Ensure the lengths match before calculating error
            min_len = min(len(linear_x), len(poss_x))
            temp_error = sum((linear_x[:min_len] - poss_x[:min_len]) ** 2)
            print(f'temp error: {temp_error}')
            
            if temp_error < error:
                error = temp_error
                best_possible_x = poss_x[:min_len]
        
        # Check if the best possible trajectory matches the ground truth within a tolerance
        min_len = min(len(best_possible_x), len(ground_truth_x))
        #if np.allclose(best_possible_x[:min_len], ground_truth_x[:min_len], atol=1e-2):
        if np.array_equal(best_possible_x[:min_len], ground_truth_x[:min_len]):
            correct_predictions.append(temp_id)

    accuracy = 100 * (len(correct_predictions) / len(unique_ids))
    print(f'Accuracy: {accuracy:.2f}%')

def main(): 
    analyze_trajectories()

if __name__ == '__main__': 
    main()
