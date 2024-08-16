import numpy as np 
import pandas as pd
import re
import matplotlib.pyplot as plt 

# Load the data
vehicle2 = pd.read_csv('possible_trajectories/batch_127_possible_trajectories.csv')
s_clean_x = vehicle2[vehicle2['Poss_ID']==2]['xloc'].values[0].strip("[]")
list_str_x = s_clean_x.split()
ground_xlist = [round(float(x),2) for x in list_str_x]
 
grt = 40.3
grt_time = []
for i in range(len(ground_xlist)):
    grt_time.append(grt)
    grt += 0.1 

# Plot the first vehicle's trajectory over time
plt.plot(grt_time,ground_xlist,label='ground')
plt.xlabel('Time (s)')
plt.ylabel('X')
plt.legend()
# plt.savefig('vehicle2_xloc.png')

# vehicle1_predicted = pd.read_csv('overall_results/overall_results.csv') 
# s_clean_x = vehicle1_predicted[(vehicle1_predicted['Vehicle_ID']==2) & (vehicle1_predicted['Predicted_ID']==8)]['muX'].values[0].strip("[]") 
# # print(s_clean_x)

# list_str_x = s_clean_x.split(', ') 

# # print(list_str_x)

# predicted_xlist = [round(float(x), 2) for x in list_str_x]





# plt.plot(grt_time,ground_xlist,label='ground')
# plt.plot(time,predicted_xlist,label='predicted')
# plt.xlabel('Time (s)')
# plt.ylabel('X')
# plt.legend()
# plt.savefig('vehicle1_xloc.png')


vehicle2 = pd.read_csv('details/trajectory_details.csv') # Vehicle_ID, ID
predicted = vehicle2[(vehicle2['Vehicle_ID']==2) & (vehicle2['Possible_ID']==2)]


for i in range(6):  
    muX = predicted['muX'].values[i].strip("[]").split(', ')
    # print(muX)
    # break
    muX = [float(x) for x in muX] 
    time = []
    val = 40.3
    for m in range(len(muX)):
        time.append(val)
        val += 0.2
    print(muX)

    
    plt.scatter(time, muX,label=f'predicted_maneuvers_{i+1}')  

plt.legend()
plt.savefig('vehicle2_xloc.png')

# plt.show()