import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


def analyze():
    best_trajectories_dir = "best_trajectories"
    best_trajectories_files = os.listdir(best_trajectories_dir)
    outgoing_trajectories = pd.read_csv('before/outgoing.csv')
    
    for predicted_trajectory_input_file in best_trajectories_files: 
        predicted_trajectory_input_path = os.path.join(best_trajectories_dir, predicted_trajectory_input_file)
        predicted_trajectory_input = pd.read_csv(predicted_trajectory_input_path)  
        ID_to_check = predicted_trajectory_input['Vehicle_ID'].values[0]
        ground_truth_trajectory = outgoing_trajectories[outgoing_trajectories['ID'] == ID_to_check]
        
        if len(predicted_trajectory_input['xloc']) != 0 and len(ground_truth_trajectory['time'] != 0):
            s_clean_x = predicted_trajectory_input['xloc'].values[0].strip("[]")
            list_str_x = s_clean_x.split()
            predicted_xlist = [round(float(x), 2) for x in list_str_x]

            s_clean_y = predicted_trajectory_input['yloc'].values[0].strip("[]")
            list_str_y = s_clean_y.split()
            predicted_ylist = [round(float(y), 2) for y in list_str_y]
 
            ground_truth_xlist = np.round(ground_truth_trajectory['xloc'].values, 2)
            ground_truth_ylist = np.round(ground_truth_trajectory['yloc'].values, 2)

            # Ensure the lists have the same length
            min_length = min(len(predicted_xlist), len(ground_truth_xlist))
            predicted_xlist = predicted_xlist[:min_length]
            predicted_ylist = predicted_ylist[:min_length]
            ground_truth_xlist = ground_truth_xlist[:min_length]
            ground_truth_ylist = ground_truth_ylist[:min_length]

            # Time axis for plotting  
            time_steps = np.linspace(0, 5, len(predicted_xlist))

            fig, ax = plt.subplots(figsize=(20, 10))
            ax.scatter(time_steps, predicted_xlist, label='Predicted X Coordinate', color='red', s=100)
            ax.plot(time_steps, ground_truth_xlist, label='Ground Truth X Coordinate', marker='x', color='blue', linewidth=3)
            ax.set_xlabel('Time (s)', fontsize=30)
            ax.set_ylabel('X coordinates (m)', fontsize=30) 
            ax.grid(True)
            ax.tick_params(axis='both', which='major', labelsize=30)
            #plt.legend(fontsize=20)
            plt.savefig(f'prediction_plots/predicted_{ID_to_check}.png')


def main():
    analyze()

if __name__ == '__main__':
    main()
