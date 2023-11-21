import pandas as pd

# Load the CSV file
file_to_split = pd.read_csv('Run_1_final_rounded.csv')

# Assuming 'lane_column' is the name of the column that contains lane information
lane_column = 'lane'  # Replace 'lane_column' with the actual column name

# Unique lane values to split
lanes = [2, 3, 4, 5]

for lane in lanes:
    # Filter data for the current lane
    lane_data = file_to_split[file_to_split[lane_column] == lane]
    
    # Save to a new CSV file
    lane_data.to_csv(f'lane_{lane}_data.csv', index=False)
