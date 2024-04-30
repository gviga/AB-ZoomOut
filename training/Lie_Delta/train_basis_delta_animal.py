from __future__ import print_function
import argparse
import os

import random
import torch
import torch.optim as optim
from deltaconvbasis import DeltaConvBasis
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from dataload_delta import AnimalDataLoader
from tqdm import tqdm
from torch_geometric.loader import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
manualSeed = 1  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)

b_size = 8

# Out Dir
outf = './models/deltaconv'
try:
    os.makedirs(outf)
except OSError:
    pass


DATA_PATH = 'data/'

TRAIN_DATASET = AnimalDataLoader(root=DATA_PATH, npoint=1000, split='train',
                                                    normal_channel=False, augm = True)
TEST_DATASET =AnimalDataLoader(root=DATA_PATH, npoint=1000, split='test',
                                                    normal_channel=False, augm = True)

dataset = DataLoader(TRAIN_DATASET, batch_size=b_size, shuffle=True, num_workers=0)
dataset_test = DataLoader(TEST_DATASET, batch_size=b_size, shuffle=True, num_workers=0)

# BasisNetwork with 30 basis
basisNet = DeltaConvBasis(in_channels=3, k=30, conv_channels=[64, 128, 256], mlp_depth=2, embedding_size=512, num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=0.5)
# Optimizer
optimizer = optim.Adam(basisNet.parameters(), lr=0.01, betas=(0.9, 0.999))
#checkpoint = torch.load(outf + '/basis_model_best.pth')
#basisNet.load_state_dict(checkpoint)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
#scheduler.load_state_dict(checkpoint['scheduler'])
basisNet.cuda()


train_losses = [] #np.load(outf + '/train_losses_basis.npy').tolist()
eval_losses = [] #np.load(outf + '/eval_losses_basis.npy').tolist()
best_eval_loss = np.inf; #np.min(eval_losses);

faust_losses = [];

start_epoch=0#np.load(outf + '/eval_losses_basis.npy').shape[0]
max_epoch=400
# Training Loop
for epoch in range(start_epoch, max_epoch):
    scheduler.step()
    train_loss = 0
    # Training single Epoch
    for data in tqdm(dataset, 0):
        points = data.pos
        points=points.reshape(-1,1000,3)
        points = points.transpose(2, 1)
        points = points.cuda().to(device)
        optimizer.zero_grad()
        basisNet = basisNet.train()
        # Obtaining predicted basis
        pred= basisNet(data)
     
        pred=pred.reshape(-1,1000,30)
        # Generating pairs
        basis_A = pred[1:,:,:]; basis_B = pred[:-1,:,:] 
        pc_A = points[1:,:,:]; pc_B = points[:-1,:,:]

        # Computing optimal transformation
        pseudo_inv_A = torch.pinverse(basis_A)
        C_opt = torch.matmul(pseudo_inv_A, basis_B)
        opt_A = torch.matmul(basis_A, C_opt)

        # SoftMap
        dist_matrix = torch.cdist(opt_A, basis_B)       
        s_max = torch.nn.Softmax(dim=1)
        s_max_matrix = s_max(-dist_matrix)

        # Basis Loss
        eucl_loss = torch.sum(torch.square(torch.matmul(s_max_matrix, torch.transpose(pc_B,1,2)) - torch.transpose(pc_B,1,2)))
        
        # Back Prop
        eucl_loss.backward()
        optimizer.step()
        train_loss += eucl_loss.item()
    
    # Validation
    with torch.no_grad():
        eval_loss = 0
        for data in tqdm(dataset_test, 0):
            points = data.pos
            points=points.reshape(-1,1000,3)
            points = points.transpose(2, 1)
            points = points.cuda()
            basisNet = basisNet.eval()
            pred = basisNet(data)
            pred=pred.reshape(-1,1000,30)
            basis_A = pred[1:,:,:]; basis_B = pred[:-1,:,:] 
            pc_A = points[1:,:,:]; pc_B = points[:-1,:,:]

            pseudo_inv_A = torch.pinverse(basis_A)
            C_opt = torch.matmul(pseudo_inv_A, basis_B)
            opt_A = torch.matmul(basis_A, C_opt)

            dist_matrix = torch.cdist(opt_A, basis_B)       
            s_max = torch.nn.Softmax(dim=1)
            s_max_matrix = s_max(-dist_matrix)
            eucl_loss = torch.sum(torch.square(torch.matmul(s_max_matrix, torch.transpose(pc_B,1,2)) - torch.transpose(pc_B,1,2)))
            eval_loss +=   eucl_loss.item()

        print('EPOCH ' + str(epoch) + ' - eva_loss: ' + str(eval_loss))

        # Saving if best model so far
        if eval_loss <  best_eval_loss:
            print('save model')
            best_eval_loss = eval_loss
            torch.save(basisNet.state_dict(), '%s/basis_model_best_animal.pth' % (outf))

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)

        # Logging losses
        np.save(outf+'/train_losses_basis_animal.npy',train_losses)
        np.save(outf+'/eval_losses_basis_animal.npy',eval_losses)