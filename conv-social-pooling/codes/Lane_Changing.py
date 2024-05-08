import pandas as pd

# Load the data
data = pd.read_csv('I294_L1_final.csv')

# Seconds to go back
delta = 10 # seconds 
  
# Prepare to capture segments of data around lane changes
lane_change_data = []

# Iterate over each unique vehicle ID
for key,ID in enumerate(data['ID'].unique()):
    print(f"index: {key}/{len(data['ID'].unique())}")
    vehicle_data = data[data['ID'] == ID]
    previous_lane = vehicle_data.iloc[0]['lane']  # Start with the first lane

    # Check for lane changes
    for index, row in vehicle_data.iterrows():
        # print(f'sub-index: {index}/{len(vehicle_data)}')
        current_lane = row['lane']
        if current_lane != previous_lane:
            # Lane change detected, find the index of this row
            change_index = vehicle_data.index.get_loc(index)

            # Calculate the range to extract around the lane change
            start_time = row['time'] - delta  # 10 seconds before
            end_time = row['time'] + delta   # 10 seconds after

            # Extract data from 10 seconds before to 10 seconds after the lane change
            mask = (vehicle_data['time'] >= start_time) & (vehicle_data['time'] <= end_time)
            segment = vehicle_data[mask]

            lane_change_data.append(segment)
            previous_lane = current_lane  # Update the lane tracker

# Concatenate all segments into a single DataFrame
resulting_data = pd.concat(lane_change_data, ignore_index=True)

# You can now save this to a CSV or analyze it further
resulting_data.to_csv('lane_changes_extracted.csv', index=False)
