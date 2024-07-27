from __future__ import print_function
import torch
from model_six_maneuvers import highwayNet_six_maneuver
from TGSIM_utils import tgsimDataset, maskedNLL, maskedMSE, maskedNLLTest
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def scale_data(muX, muY, method='minmax'): 
    muX = np.array(muX).reshape(-1, 1)
    muY = np.array(muY).reshape(-1, 1)
    
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'zscore':
        scaler = StandardScaler()
    elif method == 'log':
        scaler = None
        muX = np.log1p(abs(muX))
        muY = np.log1p(abs(muY))
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid method. Choose from 'minmax', 'zscore', 'log', 'robust'.")

    if scaler:
        muX_scaled = scaler.fit_transform(muX).flatten()
        muY_scaled = scaler.fit_transform(muY).flatten()
    else:
        muX_scaled = muX.flatten()
        muY_scaled = muY.flatten()

    return muX_scaled.tolist(), muY_scaled.tolist()

def plot_trajectory_contour(predicted_trajectories_dir, overpass_start_loc_x, overpass_end_loc_x, lane, delta, fut_pred, alpha=12, batch_num=1024):
    predicted_trajectories_files = os.listdir(predicted_trajectories_dir)

    for predicted_trajectory_input_file in predicted_trajectories_files:
        predicted_trajectory_input_path = os.path.join(predicted_trajectories_dir, predicted_trajectory_input_file)
        predicted_trajectory_input = pd.read_csv(predicted_trajectory_input_path)
        ID_to_check = predicted_trajectory_input['Vehicle_ID'].values[0]

        if len(predicted_trajectory_input['xloc']) != 0:
            s_clean_x = predicted_trajectory_input['xloc'].values[0].strip("[]")
            list_str_x = s_clean_x.split()
            x = [round(float(x), 2) for x in list_str_x]

            s_clean_y = predicted_trajectory_input['yloc'].values[0].strip("[]")
            list_str_y = s_clean_y.split()
            y = [round(float(y), 2) for y in list_str_y]
            time = np.linspace(0, delta, len(x))
  
            maneuver_colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#8c564b']  # Colors for different maneuvers
            fig, ax = plt.subplots(figsize=(10, 6))

            for m in range(6): 
                muX_before = fut_pred[m][:, batch_num-1, 0].detach().numpy()
                muY_before = fut_pred[m][:, batch_num-1, 1].detach().numpy()

                gradient = np.max(x) - np.min(x)
                
                if gradient <= alpha:
                    gradient += alpha

                muX_scaled, muY_scaled = scale_data(muX_before, muY_before, method='minmax')
                muX = [(gradient * mx) + overpass_start_loc_x for mx in muX_scaled] 
                muY = [my + overpass_start_loc_x for my in muY_scaled]

                # Create a 2D histogram for the contour
                heatmap, xedges, yedges = np.histogram2d(muX, muY, bins=30, range=[[overpass_end_loc_x, overpass_end_loc_x + delta + 0.5], [min(muY), max(muY) + 0.05]])

                # Generate the mesh grid for the contour plot
                X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
                contour = ax.contourf(X, Y, heatmap.T, levels=60, cmap='viridis', alpha=0.5)
                fig.colorbar(contour)

                # Plot the predicted trajectory
                ax.plot(time[:len(muX)], muX[:len(time)], marker='o', linestyle='-', color=maneuver_colors[m], label=f'Trajectory for Maneuver {m+1}')

            ax.set_xlabel('Time (seconds)', fontsize=14)
            ax.set_ylabel('X - Longitudinal Coordinate (meters)', fontsize=14)
            ax.plot(time, x[:len(time)], marker='o', color='white', label=f'Trajectory for ID {ID_to_check}')  # Plot the actual trajectory
            
            ax.grid(True)
            fig.tight_layout()
            plt.savefig(f'contour_maps/vehicle_{ID_to_check}_contour.png')
            plt.close()


def main():
    predicted_trajectories_dir = "best_trajectories"  
    args = {}
    print('cuda available', torch.cuda.is_available())
    print('torch version', torch.__version__)
    print('cuda version', torch.version.cuda)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device == 'cuda':
        args['use_cuda'] = True 
    else:
        args['use_cuda'] = False 

    print(f'My device: {device}')
    args['encoder_size'] = 64
    args['decoder_size'] = 128
    args['in_length'] = 16
    args['out_length'] = 50
    args['grid_size'] = (13, 3)
    args['soc_conv_depth'] = 64
    args['conv_3x1_depth'] = 16
    args['dyn_embedding_size'] = 32
    args['input_embedding_size'] = 32
    args['num_lat_classes'] = 3
    args['num_lon_classes'] = 2
    args['use_maneuvers'] = True
    args['train_flag'] = False

    filepath_pred_Set = 'cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10_seconds/test'
    file_to_read = 'I294_Cleaned.csv'

    df = pd.read_csv(file_to_read)
    original_data = df.copy()
    temp_lane = -2 
    lanes_to_analyze = [temp_lane]

    batch_size = 1024
    overpass_start_loc_x, overpass_end_loc_x = 1800, 1815
    delta = 5

    net = highwayNet_six_maneuver(args)
    model_path = 'trained_model_TGSIM/cslstm_m.tar'
    net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    if args['use_cuda']: 
        net = net.to(device)

    predSet = tgsimDataset(filepath_pred_Set, t_h=30, t_f=100, d_s=2)
    predDataloader = DataLoader(predSet, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=predSet.collate_fn)

    for lane in lanes_to_analyze:  
        for i, data in enumerate(predDataloader):  
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, maneuver_enc = data

            if args['use_cuda']:
                hist = hist.cuda()
                nbrs = nbrs.cuda()
                mask = mask.cuda()
                lat_enc = lat_enc.cuda()
                lon_enc = lon_enc.cuda()
                fut = fut.cuda()
                op_mask = op_mask.cuda()
                maneuver_enc = maneuver_enc.cuda()

            fut_pred, maneuver_pred = net(hist, nbrs, mask, lat_enc, lon_enc) 
            plot_trajectory_contour(predicted_trajectories_dir, overpass_start_loc_x, overpass_end_loc_x, lane, delta, fut_pred, alpha=12, batch_num=min(batch_size, fut_pred[0].shape[1]))

            if i == 0:
                break

if __name__ == '__main__':
    main()
