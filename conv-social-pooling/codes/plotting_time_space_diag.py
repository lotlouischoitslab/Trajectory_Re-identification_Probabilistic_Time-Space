import pandas as pd
import matplotlib.pyplot as plt


smoothed_file = pd.read_csv('raw_trajectory.csv')

lanes = [2, 3, 4, 5]


for lane in lanes:
    lane_data = smoothed_file[smoothed_file['lane']==lane].reset_index(drop=True)


    IDs = []
    all_ts = []
    all_ys = []
    init_ID = -1

    # get all vehicle IDs
    for i in range(len(lane_data)):
        if lane_data['ID'][i] != init_ID:
            IDs.append(lane_data['ID'][i])
            init_ID = lane_data['ID'][i]

    # get xs and ts of each vehicle
    fig, ax = plt.subplots()
    for j in IDs:
        temp_data = lane_data[lane_data['ID']==j]
        # ys = temp_data['yloc-kf'].to_numpy()
        # ys = temp_data['yloc'].to_numpy()

        ys = temp_data['xloc'].to_numpy()
        # ys = temp_data['xloc-kf'].to_numpy()

        ts = temp_data.time.to_numpy()

        # lens = temp_data['length-smoothed'].to_numpy()

        # ys = (ys+0.5*lens)*3.2808399
        # ys = ys+0.5*lens
        ax.scatter(ts, ys,s=1)
        ax.text(ts[0], ys[0], str(j))

    ax.set_xlabel('Time (s)', fontsize = 20)
    # ax.set_ylabel('Location (ft)', fontsize = 20)
    ax.set_ylabel('Location (m)', fontsize = 20)

    # Increase the number of grid lines on the x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(100))

    # Increase the number of grid lines on the y-axis
    ax.yaxis.set_major_locator(plt.MaxNLocator(60))
    ax.grid()

    fig.set_size_inches(120,30)
    fig.savefig(f'Lane_{lane}-x.png')
