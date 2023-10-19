from __future__ import print_function
import argparse
import os
import random
import torch
import torch.optim as optim
from model import PointNetBasis
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from dataload import Surr12kModelNetDataLoader as DataLoader
from tqdm import tqdm


#stabilisce il device su cui fare le computazioni, in particolari  GPU con core cuda

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

manualSeed = 1  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)

b_size = 8

# Out Dir
outf = '../models_lie/trained'
try:
    os.makedirs(outf)
except OSError:
    pass


DATA_PATH = '../dataset/'


#data splitting
TRAIN_DATASET = DataLoader(root=DATA_PATH, npoint=1000, split='train',
                                                    normal_channel=False, augm = True)
TEST_DATASET = DataLoader(root=DATA_PATH, npoint=1000, split='test',
                                                    normal_channel=False, augm = True)

#load
dataset = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=b_size, shuffle=True, num_workers=0)
dataset_test = torch.utils.data.DataLoader(TEST_DATASET, batch_size=b_size, shuffle=True, num_workers=0)
# BasisNetwork with 20 basis
#richiamiamo l'architettura(la scelta qui è PointNet ----> da cambiare in DiffudsionNet(?))
basisNet = PointNetBasis(k=30, feature_transform=False)

# Optimizer
#scelta dell'ottimizzatore per la rete (otim.Adam)
#i parametri da ottimizzare sono indicati nel file del modello
#gli hyperparameters da vedere
optimizer = optim.Adam(basisNet.parameters(), lr=0.01, betas=(0.9, 0.999))

basisNet.cuda()

best_eval_loss = np.inf;

train_losses = [];
eval_losses = [];
faust_losses = [];

# Training Loop
for epoch in range(400):
    #scheduler.step()
    print("\n Epoch Number:",epoch)
    train_loss = 0
    # Training single Epoch
    #loop sui dati
    for data in tqdm(dataset, 0):
        #embedding naturale
        points = data[0]
        points = points.transpose(2, 1)
        points = points.cuda().to(device)
        optimizer.zero_grad()
        basisNet = basisNet.train()
        # Obtaining predicted basis
        pred, _, _ = basisNet(points)

        # Generating pairs
        #pairs di basi
        basis_A = pred[1:,:,:]; basis_B = pred[:-1,:,:] 
        #pairs di natural embedding
        pc_A = points[1:,:,:]; pc_B = points[:-1,:,:]

        # Computing optimal transformation
        #calcolo la trasformazione
        pseudo_inv_A = torch.pinverse(basis_A)
        C_opt = torch.matmul(pseudo_inv_A, basis_B)
        opt_A = torch.matmul(basis_A, C_opt)

        # SoftMap
        #calcolo la matrice di softmap
        dist_matrix = torch.cdist(opt_A, basis_B)       
        s_max = torch.nn.Softmax(dim=1)
        s_max_matrix = s_max(-dist_matrix)

        # Basis Loss
        #calcolo la loss function per le basi
        eucl_loss = torch.sum(torch.square(torch.matmul(s_max_matrix, torch.transpose(pc_B,1,2)) - torch.transpose(pc_B,1,2)))
        
        # Back Prop
        #passo di backpropagation
        eucl_loss.backward()
        optimizer.step()
        train_loss += eucl_loss.item()
    
    # Validation
    with torch.no_grad():
        eval_loss = 0
        for data in tqdm(dataset_test, 0):
            points = data[0]
            points = points.transpose(2, 1)
            points = points.cuda()
            basisNet = basisNet.eval()
            pred, _, _ = basisNet(points)        
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
            torch.save(basisNet.state_dict(), '%s/basis_model_best_30.pth' % (outf))

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)

        # Logging losses
        np.save(outf+'/train_losses_basis_30.npy',train_losses)
        np.save(outf+'/eval_losses_basis_30.npy',eval_losses)





