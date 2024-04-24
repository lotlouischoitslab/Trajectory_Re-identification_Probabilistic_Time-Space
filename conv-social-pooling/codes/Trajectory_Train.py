from __future__ import print_function
import torch
# from model import highwayNet_six_maneuver
from model_six_maneuvers import highwayNet_six_maneuver
from TGSIM_utils import tgsimDataset,maskedNLL,maskedMSE,maskedNLLTest
from torch.utils.data import DataLoader
import time
import math
import pandas as pd

###FOR MULTI-GPU system using a single gpu:
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0" # this should be 0


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## Network Arguments
    args = {}
    args['use_cuda'] = True
    args['encoder_size'] = 64  
    args['decoder_size'] = 128  
    args['in_length'] = 16 
    args['out_length'] = 50
    args['grid_size'] = (13,3)
    args['soc_conv_depth'] = 64
    args['conv_3x1_depth'] = 16
    args['dyn_embedding_size'] = 32
    args['input_embedding_size'] = 32
    args['num_lat_classes'] = 3
    args['num_lon_classes'] = 2
    args['use_maneuvers'] = True
    args['train_flag'] = True


    # Initialize network
    net = highwayNet_six_maneuver(args)
    if args['use_cuda']:
        net = net.to(device)
    ## Initialize optimizer
    pretrainEpochs = 7
    trainEpochs = 3

  

    optimizer = torch.optim.Adam(net.parameters())
    batch_size = 128
    # crossEnt = torch.nn.BCELoss() 

    crossEnt = torch.nn.MSELoss() 


    ## Initialize data loaders
    # trSet = ngsimDataset('/reza/projects/trajectory-prediction/data/NGSIM/101-80-speed-maneuver-for-GT/10-seconds/train', t_h=30, t_f=100, d_s=2)
    # valSet = ngsimDataset('/reza/projects/trajectory-prediction/data/NGSIM/101-80-speed-maneuver-for-GT/10-seconds/valid', t_h=30, t_f=100, d_s=2)

    
    
    print('Loading Data')
    train_trajectories_directory = 'cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10_seconds/train' # HAL GPU Cluster
    valid_trajectories_directory = 'cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10_seconds/valid' # HAL GPU Cluster
  
   
    trSet = tgsimDataset(train_trajectories_directory, t_h=30, t_f=100, d_s=2)
    valSet = tgsimDataset(valid_trajectories_directory, t_h=30, t_f=100, d_s=2)


    print('TGSIM Data Loaded!')

    trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=True,num_workers=1,collate_fn=trSet.collate_fn)
    valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=True,num_workers=1,collate_fn=valSet.collate_fn)

    ################################# TRAINING PART ##################################################################################
    print('Start Training Now')
    ## Variables holding train and validation loss values:
    train_loss = []
    val_loss = []
    prev_val_loss = math.inf

    for epoch_num in range(pretrainEpochs+trainEpochs):
        if epoch_num == 0:
            print('Pre-training with MSE loss')
        elif epoch_num == pretrainEpochs:
            print('Training with NLL loss')
        
        net.train_flag = True

        # Variables to track training performance:
        avg_tr_loss = 0
        avg_tr_time = 0
        avg_lat_acc = 0
        avg_lon_acc = 0
        avg_maneuver_acc = 0


        for i, data in enumerate(trDataloader):
            st_time = time.time()
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

            # Forward pass
            if args['use_maneuvers']:
                fut_pred, maneuver_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                # Pre-train with MSE loss to speed up training
                if epoch_num < pretrainEpochs:
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                # Train with NLL loss
                    l = maskedNLL(fut_pred, fut, op_mask) + crossEnt(maneuver_pred, maneuver_enc)
                    avg_maneuver_acc += (torch.sum(torch.max(maneuver_pred.data, 1)[1] == torch.max(maneuver_enc, 1)[1])).item() / maneuver_enc.size()[0]
                    lat_pred = torch.remainder(torch.max(maneuver_pred.data, 1)[1], args['num_lat_classes'])
                    # lon_pred = torch.div(torch.max(maneuver_pred.data, 1)[1], args['num_lat_classes'], rounding_mode='floor')
                    lon_pred = torch.floor_divide(torch.max(maneuver_pred.data, 1)[1], args['num_lat_classes'])
                    # print('maneuver_enc.shape: ', maneuver_enc.shape)
                    # print('maneuver_pred.shape: ', maneuver_pred.shape)
                    # print('lat_enc.shape: ', lat_enc.shape)
                    # print('lon_enc.shape: ', lon_enc.shape)
                    # print('lat_pred.shape: ', lat_pred.shape)
                    # print('lon_pred.shape: ', lon_pred.shape)
                    # print((torch.sum(lat_pred.data == torch.max(lat_enc.data, 1)[1])).item())
                    # print((torch.sum(lon_pred.data == torch.max(lon_enc.data, 1)[1])).item())
                    avg_lat_acc += (torch.sum(lat_pred.data == torch.max(lat_enc.data, 1)[1])).item() / \
                                lat_enc.size()[0]
                    avg_lon_acc += (torch.sum(lon_pred.data == torch.max(lon_enc.data, 1)[1])).item() / \
                                lon_enc.size()[0]

                    # avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                    # avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
            else:
                fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                if epoch_num < pretrainEpochs:
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                    l = maskedNLL(fut_pred, fut, op_mask)

            # Backprop and update weights
            optimizer.zero_grad()
            l.backward()
            a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            optimizer.step()

            # Track average train loss and average train time:
            batch_time = time.time()-st_time
            avg_tr_loss += l.item()
            avg_tr_time += batch_time

            if i%100 == 99:
                eta = avg_tr_time/100*(len(trSet)/batch_size-i)
                print("Epoch no:",epoch_num+1,"| Epoch progress(%):",format(i/(len(trSet)/batch_size)*100,'0.2f'), "| Avg train loss:",format(avg_tr_loss/100,'0.4f'),"| Acc:",format(avg_lat_acc,'0.4f'),format(avg_lon_acc,'0.4f'), "| Validation loss prev epoch",format(prev_val_loss,'0.4f'), "| ETA(s):",int(eta))
                train_loss.append(avg_tr_loss/100)
                avg_tr_loss = 0
                avg_lat_acc = 0
                avg_lon_acc = 0
                avg_tr_time = 0
        #######################################################################################################################################################################################################


        ########################################### VALIDATE ######################################################################################################################################################################################################
        net.train_flag = False

        print("Epoch",epoch_num+1,'complete. Calculating validation loss...')
        avg_val_loss = 0
        avg_val_lat_acc = 0
        avg_val_lon_acc = 0
        val_batch_count = 0
        total_points = 0
        avg_maneuver_acc = 0

        for i, data  in enumerate(valDataloader):
            st_time = time.time()
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

            # Forward pass
            if args['use_maneuvers']:
                if epoch_num < pretrainEpochs:
                    # During pre-training with MSE loss, validate with MSE for true maneuver class trajectory
                    net.train_flag = True
                    fut_pred, _ = net(hist, nbrs, mask, lat_enc, lon_enc)
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                    # During training with NLL loss, validate with NLL over multi-modal distribution
                    fut_pred, maneuver_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    l = maskedNLLTest(fut_pred, maneuver_pred, fut, op_mask,avg_along_time = True)
                    avg_maneuver_acc += (torch.sum(torch.max(maneuver_pred.data, 1)[1] == torch.max(maneuver_enc, 1)[1])).item() / maneuver_enc.size()[0]
                    lat_pred = torch.remainder(torch.max(maneuver_pred.data, 1)[1], args['num_lat_classes'])
                    lon_pred = torch.floor_divide(torch.max(maneuver_pred.data, 1)[1], args['num_lat_classes'])
                    # lon_pred = torch.div(torch.max(maneuver_pred.data, 1)[1], args['num_lat_classes'], rounding_mode='floor')
                    avg_val_lat_acc += (torch.sum(lat_pred.data == torch.max(lat_enc.data, 1)[1])).item() / \
                                lat_enc.size()[0]
                    avg_val_lon_acc += (torch.sum(lon_pred.data == torch.max(lon_enc.data, 1)[1])).item() / \
                                lon_enc.size()[0]
                    # avg_val_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                    # avg_val_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
            else:
                fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                if epoch_num < pretrainEpochs:
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                    l = maskedNLL(fut_pred, fut, op_mask)

            avg_val_loss += l.item()
            val_batch_count += 1

        print(avg_val_loss/val_batch_count)

        # Print validation loss and update display variables
        print('Validation loss :',format(avg_val_loss/val_batch_count,'0.4f'),"| Val Maneuver Acc:",format(avg_maneuver_acc/val_batch_count*100,'0.4f'),"| Val Acc:",format(avg_val_lat_acc/val_batch_count*100,'0.4f'),format(avg_val_lon_acc/val_batch_count*100,'0.4f'))
        val_loss.append(avg_val_loss/val_batch_count)
        prev_val_loss = avg_val_loss/val_batch_count


    torch.save(net.state_dict(), 'trained_model_TGSIM/cslstm_m.tar')


if __name__ =='__main__':
    main()
