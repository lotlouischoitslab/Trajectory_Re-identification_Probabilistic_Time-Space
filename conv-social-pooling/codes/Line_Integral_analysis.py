# # import os
# # import numpy as np 
# # import pandas as pd 
# # import matplotlib.pyplot as plt


# # def analyze():
# #     best_trajectories_dir = "best_trajectories"
# #     best_trajectories_files = os.listdir(best_trajectories_dir)
# #     outgoing_trajectories = pd.read_csv('before/outgoing.csv')
    
# #     correct_cases = []
# #     wrong_cases = []

# #     for predicted_trajectory_input_file in best_trajectories_files: 
# #         predicted_trajectory_input_path = os.path.join(best_trajectories_dir, predicted_trajectory_input_file)
# #         predicted_trajectory_input = pd.read_csv(predicted_trajectory_input_path)  
# #         ID_to_check = predicted_trajectory_input['Vehicle_ID'].values[0]
# #         ground_truth_trajectory = outgoing_trajectories[outgoing_trajectories['ID'] == ID_to_check]
        
# #         if len(predicted_trajectory_input['xloc']) != 0 and len(ground_truth_trajectory['time']) != 0:
# #             s_clean_x = predicted_trajectory_input['xloc'].values[0].strip("[]")
# #             list_str_x = s_clean_x.split()
# #             predicted_xlist = [round(float(x), 2) for x in list_str_x]

# #             s_clean_y = predicted_trajectory_input['yloc'].values[0].strip("[]")
# #             list_str_y = s_clean_y.split()
# #             predicted_ylist = [round(float(y), 2) for y in list_str_y]
 
# #             ground_truth_xlist = np.round(ground_truth_trajectory['xloc'].values, 2)
# #             ground_truth_ylist = np.round(ground_truth_trajectory['yloc'].values, 2)

# #             # Ensure the lists have the same length
# #             min_length = min(len(predicted_xlist), len(ground_truth_xlist))
# #             predicted_xlist = predicted_xlist[:min_length]
# #             predicted_ylist = predicted_ylist[:min_length]
# #             ground_truth_xlist = ground_truth_xlist[:min_length]
# #             ground_truth_ylist = ground_truth_ylist[:min_length]

# #             # Retrieve line integral value
# #             line_integral_value = predicted_trajectory_input['line_integral_values'].values[0]

# #             # Check if the prediction is correct or not
# #             is_correct = all(px == gx for px, gx in zip(predicted_xlist, ground_truth_xlist))
# #             if is_correct:
# #                 correct_cases.append(line_integral_value)
# #             else:
# #                 wrong_cases.append(line_integral_value)

# #             # Time axis for plotting  
# #             time_steps = np.linspace(0, 5, len(predicted_xlist))

# #             fig, ax = plt.subplots(figsize=(20, 10))
# #             ax.plot(time_steps, predicted_xlist, label='Predicted X Coordinate', marker='o', color='red', linewidth=3)
# #             ax.plot(time_steps, ground_truth_xlist, label='Ground Truth X Coordinate', marker='x', color='blue', linewidth=3)
# #             ax.set_xlabel('Time (s)', fontsize=30)
# #             ax.set_ylabel('X coordinates (m)', fontsize=30) 
# #             ax.grid(True)
# #             ax.tick_params(axis='both', which='major', labelsize=30)
# #             plt.savefig(f'prediction_plots/predicted_{ID_to_check}.png')

# #     # Plot the distribution of line integral values
# #     fig, ax = plt.subplots(figsize=(12, 6))
# #     bins = np.linspace(0, max(correct_cases + wrong_cases), 10)  # Adjust number of bins as needed
# #     ax.hist(correct_cases, bins=bins, alpha=0.5, label='Correct Cases', density=True, edgecolor='black',color='blue')
# #     ax.hist(wrong_cases, bins=bins, alpha=0.5, label='Wrong Cases', density=True, edgecolor='black',color='blue')
# #     ax.set_xlabel('Line Integral Value', fontsize=16)
# #     ax.set_ylabel('Density', fontsize=16) 
# #     plt.grid(True)
# #     plt.savefig('line_integral_distribution.png')


# # def main():
# #     analyze()


# # if __name__ == '__main__':
# #     main()


# # import os
# # import numpy as np 
# # import pandas as pd 
# # import matplotlib.pyplot as plt
# # import seaborn as sns


# # def analyze():
# #     best_trajectories_dir = "best_trajectories"
# #     best_trajectories_files = os.listdir(best_trajectories_dir)
# #     outgoing_trajectories = pd.read_csv('before/outgoing.csv')
    
# #     line_integral_values = []

# #     for predicted_trajectory_input_file in best_trajectories_files: 
# #         predicted_trajectory_input_path = os.path.join(best_trajectories_dir, predicted_trajectory_input_file)
# #         predicted_trajectory_input = pd.read_csv(predicted_trajectory_input_path)  
# #         ID_to_check = predicted_trajectory_input['Vehicle_ID'].values[0]
# #         ground_truth_trajectory = outgoing_trajectories[outgoing_trajectories['ID'] == ID_to_check]
        
# #         if len(predicted_trajectory_input['xloc']) != 0 and len(ground_truth_trajectory['time']) != 0:
# #             s_clean_x = predicted_trajectory_input['xloc'].values[0].strip("[]")
# #             list_str_x = s_clean_x.split()
# #             predicted_xlist = [round(float(x), 2) for x in list_str_x]

# #             s_clean_y = predicted_trajectory_input['yloc'].values[0].strip("[]")
# #             list_str_y = s_clean_y.split()
# #             predicted_ylist = [round(float(y), 2) for y in list_str_y]
 
# #             ground_truth_xlist = np.round(ground_truth_trajectory['xloc'].values, 2)
# #             ground_truth_ylist = np.round(ground_truth_trajectory['yloc'].values, 2)

# #             # Ensure the lists have the same length
# #             min_length = min(len(predicted_xlist), len(ground_truth_xlist))
# #             predicted_xlist = predicted_xlist[:min_length]
# #             predicted_ylist = predicted_ylist[:min_length]
# #             ground_truth_xlist = ground_truth_xlist[:min_length]
# #             ground_truth_ylist = ground_truth_ylist[:min_length]

# #             # Retrieve line integral value
# #             line_integral_value = predicted_trajectory_input['line_integral_values'].values[0]
# #             line_integral_values.append(line_integral_value)

# #             # Time axis for plotting  
# #             time_steps = np.linspace(0, 5, len(predicted_xlist))

# #             fig, ax = plt.subplots(figsize=(20, 10))
# #             ax.plot(time_steps, predicted_xlist, label='Predicted X Coordinate', marker='o', color='red', linewidth=3)
# #             ax.plot(time_steps, ground_truth_xlist, label='Ground Truth X Coordinate', marker='x', color='blue', linewidth=3)
# #             ax.set_xlabel('Time (s)', fontsize=30)
# #             ax.set_ylabel('X coordinates (m)', fontsize=30) 
# #             ax.grid(True)
# #             ax.tick_params(axis='both', which='major', labelsize=30)
# #             plt.savefig(f'prediction_plots/predicted_{ID_to_check}.png')

# #     # Plot the distribution of line integral values using KDE
# #     fig, ax = plt.subplots(figsize=(12, 6))
# #     sns.kdeplot(line_integral_values, ax=ax, fill=True, color='blue')
# #     ax.set_xlabel('Line Integral Value', fontsize=16)
# #     ax.set_ylabel('Density', fontsize=16)
# #     plt.grid(True)
# #     plt.savefig('line_integral_distribution.png')

# # def main():
# #     analyze()

# # if __name__ == '__main__':
# #     main()

# import os
# import numpy as np 
# import pandas as pd 
# import matplotlib.pyplot as plt


# def analyze():
#     best_trajectories_dir = "best_trajectories"
#     best_trajectories_files = os.listdir(best_trajectories_dir)
#     outgoing_trajectories = pd.read_csv('before/outgoing.csv')
    
#     line_integral_values = []

#     for predicted_trajectory_input_file in best_trajectories_files: 
#         predicted_trajectory_input_path = os.path.join(best_trajectories_dir, predicted_trajectory_input_file)
#         predicted_trajectory_input = pd.read_csv(predicted_trajectory_input_path)  
#         ID_to_check = predicted_trajectory_input['Vehicle_ID'].values[0]
#         ground_truth_trajectory = outgoing_trajectories[outgoing_trajectories['ID'] == ID_to_check]
        
#         if len(predicted_trajectory_input['xloc']) != 0 and len(ground_truth_trajectory['time']) != 0:
#             s_clean_x = predicted_trajectory_input['xloc'].values[0].strip("[]")
#             list_str_x = s_clean_x.split()
#             predicted_xlist = [round(float(x), 2) for x in list_str_x]

#             s_clean_y = predicted_trajectory_input['yloc'].values[0].strip("[]")
#             list_str_y = s_clean_y.split()
#             predicted_ylist = [round(float(y), 2) for y in list_str_y]
 
#             ground_truth_xlist = np.round(ground_truth_trajectory['xloc'].values, 2)
#             ground_truth_ylist = np.round(ground_truth_trajectory['yloc'].values, 2)

#             # Ensure the lists have the same length
#             min_length = min(len(predicted_xlist), len(ground_truth_xlist))
#             predicted_xlist = predicted_xlist[:min_length]
#             predicted_ylist = predicted_ylist[:min_length]
#             ground_truth_xlist = ground_truth_xlist[:min_length]
#             ground_truth_ylist = ground_truth_ylist[:min_length]

#             # Retrieve line integral value
#             line_integral_value = predicted_trajectory_input['line_integral_values'].values[0]
#             line_integral_values.append(line_integral_value)

#             # Time axis for plotting  
#             time_steps = np.linspace(0, 5, len(predicted_xlist))

#             fig, ax = plt.subplots(figsize=(20, 10))
#             ax.plot(time_steps, predicted_xlist, label='Predicted X Coordinate', marker='o', color='red', linewidth=3)
#             ax.plot(time_steps, ground_truth_xlist, label='Ground Truth X Coordinate', marker='x', color='blue', linewidth=3)
#             ax.set_xlabel('Time (s)', fontsize=30)
#             ax.set_ylabel('X coordinates (m)', fontsize=30) 
#             ax.grid(True)
#             ax.tick_params(axis='both', which='major', labelsize=30)
#             plt.savefig(f'prediction_plots/predicted_{ID_to_check}.png')

#     # Plot the distribution of line integral values using a histogram with density normalization
#     fig, ax = plt.subplots(figsize=(12, 6))
#     bins = np.linspace(0, max(line_integral_values), 12)  # Adjust the number of bins as needed
#     ax.hist(line_integral_values, bins=bins, density=True, alpha=0.6, color='blue', edgecolor='black')
#     ax.set_xlabel('Line Integral Value', fontsize=14)
#     ax.set_ylabel('Density', fontsize=14)
#     plt.grid(True)
#     plt.savefig('line_integral_distribution_hist.png')

# def main():
#     analyze()

# if __name__ == '__main__':
#     main()

import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


def analyze():
    best_trajectories_dir = "best_trajectories"
    best_trajectories_files = os.listdir(best_trajectories_dir)
    outgoing_trajectories = pd.read_csv('before/outgoing.csv')
    
    correct_cases = []
    wrong_cases = []

    for predicted_trajectory_input_file in best_trajectories_files: 
        predicted_trajectory_input_path = os.path.join(best_trajectories_dir, predicted_trajectory_input_file)
        predicted_trajectory_input = pd.read_csv(predicted_trajectory_input_path)  
        ID_to_check = predicted_trajectory_input['Vehicle_ID'].values[0]
        ground_truth_trajectory = outgoing_trajectories[outgoing_trajectories['ID'] == ID_to_check]
        
        if len(predicted_trajectory_input['xloc']) != 0 and len(ground_truth_trajectory['time']) != 0:
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

            # Retrieve line integral value
            line_integral_value = predicted_trajectory_input['line_integral_values'].values[0]

            # Check if the prediction is correct or not
            is_correct = all(px == gx for px, gx in zip(predicted_xlist, ground_truth_xlist))
            if is_correct:
                correct_cases.append(line_integral_value)
            else:
                wrong_cases.append(line_integral_value)

            # Time axis for plotting  
            time_steps = np.linspace(0, 5, len(predicted_xlist))

            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(time_steps, predicted_xlist, label='Predicted X Coordinate', marker='o', color='red', linewidth=3)
            ax.plot(time_steps, ground_truth_xlist, label='Ground Truth X Coordinate', marker='x', color='blue', linewidth=3)
            ax.set_xlabel('Time (s)', fontsize=30)
            ax.set_ylabel('X coordinates (m)', fontsize=30) 
            ax.grid(True)
            ax.tick_params(axis='both', which='major', labelsize=30)
            plt.savefig(f'prediction_plots/predicted_{ID_to_check}.png')

    # Plot the distribution of line integral values using a histogram with density normalization
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.linspace(0, max(correct_cases + wrong_cases), 12)  # Adjust the number of bins as needed
    ax.hist(correct_cases, bins=bins, density=True, alpha=0.5, color='blue', edgecolor='black', label='Correct Cases')
    ax.hist(wrong_cases, bins=bins, density=True, alpha=0.5, color='red', edgecolor='black', label='Wrong Cases')
    ax.set_xlabel('Line Integral Value', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.legend(loc='upper right', fontsize=14)
    plt.grid(True)
    plt.savefig('line_integral_distribution.png')

def main():
    analyze()

if __name__ == '__main__':
    main()
