from __future__ import print_function
import torch
from model_six_maneuvers import highwayNet_six_maneuver
from utils_works_with_101_80_cnn_modified_passes_history_too_six_maneuvers import ngsimDataset,maskedNLL,maskedMSE,maskedNLLTest, maskedMSETest
from torch.utils.data import DataLoader
import time
import math

import pickle
import numpy as np

###FOR MULTI-GPU system using a single gpu:
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="3"


## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
#args['out_length'] = 25
args['out_length'] = 100
args['grid_size'] = (13,3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3
args['num_lon_classes'] = 2
args['use_maneuvers'] = True
args['train_flag'] = False

directory = '/reza/projects/trajectory-prediction/codes/predicted_environment/'
model_directory = 'models/highwaynet-20-sec-101-80-speed-maneuver-for-GT-six-maneuvers/cslstm_m.tar'

# saving_directory = 'predicted_data/highwaynet-20-sec-101-80-speed-maneuver-for-GT-six-maneuvers/'
saving_directory = 'predicted_data/highwaynet-20-sec-101-80-speed-maneuver-for-GT-six-maneuvers/test/'
batch_size = 128

# Initialize network
net = highwayNet_six_maneuver(args)
net.load_state_dict(torch.load(directory+model_directory))
if args['use_cuda']:
    net = net.cuda()


## Initialize data loaders
# predSet = ngsimDataset('/reza/projects/trajectory-prediction/data/NGSIM/101-80-speed-maneuver-for-GT/20-seconds/train', t_h=30, t_f=200, d_s=2)
# predSet = ngsimDataset('/reza/projects/trajectory-prediction/data/NGSIM/101-80-speed-maneuver-for-GT/20-seconds/valid', t_h=30, t_f=200, d_s=2)
predSet = ngsimDataset('/reza/projects/trajectory-prediction/data/NGSIM/101-80-speed-maneuver-for-GT/20-seconds/test', t_h=30, t_f=200, d_s=2)

predDataloader = DataLoader(predSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=predSet.collate_fn)
lossVals = torch.zeros(100).cuda()
counts = torch.zeros(100).cuda()

## Variables holding train and validation loss values:
train_loss = []
val_loss = []
prev_val_loss = math.inf

net.train_flag = False

# Saving data
data_points = []
fut_predictions = []
lat_predictions = []
lon_predictions = []
maneuver_predictions = []
num_points = 0

for i, data  in enumerate(predDataloader):
    st_time = time.time()
    hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, points, maneuver_enc  = data

    if args['use_cuda']:
        hist = hist.cuda()
        nbrs = nbrs.cuda()
        mask = mask.cuda()
        lat_enc = lat_enc.cuda()
        lon_enc = lon_enc.cuda()
        fut = fut.cuda()
        op_mask = op_mask.cuda()
        maneuver_enc = maneuver_enc.cuda()

    # Forward pass

    fut_pred, maneuver_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
    fut_pred_max = torch.zeros_like(fut_pred[0])
    for k in range(maneuver_pred.shape[0]):
        indx = torch.argmax(maneuver_pred[k, :]).detach()
        fut_pred_max[:, k, :] = fut_pred[indx][:, k, :]
    l, c = maskedMSETest(fut_pred_max, fut, op_mask)

    lossVals += l.detach()
    counts += c.detach()

    # ##DEBUG
    # print("len(fut_pred), must be 6: ",len(fut_pred))
    # for m in range(len(fut_pred)):
    #     print("shape of fut_pred[m], must be (t_f//d_s,batch_size,5): ", fut_pred[m].shape)
    #     for n in range(batch_size):
    #         muX = fut_pred[m][:,n,0]
    #         muY = fut_pred[m][:,n,1]
    #         sigX = fut_pred[m][:,n,2]
    #         sigY = fut_pred[m][:,n,3]
    #         print('muX: ', muX)
    #         print('muY: ', muY)
    #         print('sigX: ', sigX)
    #         print('sigY: ', sigY)
    # ##END OF DEBUG

    points_np = points.numpy()
    fut_pred_np = []
    for k in range(6): #manuevers
        # fut_pred_np_point = fut_pred[k].clone().detach().cpu().numpy()
        fut_pred_np_point = fut_pred[k].detach().cpu().numpy()
        fut_pred_np.append(fut_pred_np_point)
    fut_pred_np = np.array(fut_pred_np)

    for j in range(points_np.shape[0]):
        point = points_np[j]
        # print('point.shape should be (49,): ', point.shape)
        data_points.append(point)
        fut_pred_point = fut_pred_np[:,:,j,:]
        # ###DEBUG
        # print('fut_pred_point.shape should be (6,t_f//d_s,5): ',fut_pred_point.shape) #6 is for different lon and lat maneuvers
        # print("check this: \n")
        # for i in range(6):
        #     muX = fut_pred_point[i, :, 0]
        #     muY = fut_pred_point[i, :, 1]
        #     sigX = fut_pred_point[i, :, 2]
        #     sigY = fut_pred_point[i, :, 3]
        #     print('muX: ', muX)
        #     print('muY: ', muY)
        #     print('sigX: ', sigX)
        #     print('sigY: ', sigY)
        # ###END OF DEBUG
        fut_predictions.append(fut_pred_point)


        maneuver_m = maneuver_pred[j].detach().cpu().numpy()
        maneuver_predictions.append(maneuver_m)

        num_points += 1
        if num_points%10000 == 0:
            print('point: ', num_points)
            print('point.shape should be (49,): ', point.shape)
            print('fut_pred_point.shape should be (6,t_f//d_s,5): ', fut_pred_point.shape)
            print('maneuver_m.shape should be (6,):', maneuver_m.shape)

print('MSE: ', lossVals / counts)

# Print test error
print('RMSE: ', torch.pow(lossVals / counts,0.5))   # Calculate RMSE, feet


print('number of data points: ', num_points)
with open(directory+saving_directory+"data_points.data", "wb") as filehandle:
	pickle.dump(np.array(data_points), filehandle, protocol=4)

with open(directory+saving_directory+"fut_predictions.data", "wb") as filehandle:
	pickle.dump(np.array(fut_predictions), filehandle, protocol=4)

with open(directory+saving_directory+"maneuver_predictions.data", "wb") as filehandle:
	pickle.dump(np.array(maneuver_predictions), filehandle, protocol=4)