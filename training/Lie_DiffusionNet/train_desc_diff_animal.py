from __future__ import print_function
import argparse
import os
import random
import torch
import torch.optim as optim
from diffusionnet import DiffusionNetBasis,DiffusionNetDesc
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from dataload_diff import AnimalDataLoader as DataLoader
from tqdm import tqdm
from diffusion_net import geometry


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
manualSeed = 1  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)

b_size = 8

# Out Dir
outf = './models/diffusionnet/'
try:
    os.makedirs(outf)
except OSError:
    pass


DATA_PATH = 'data/'


TRAIN_DATASET = DataLoader(root=DATA_PATH, npoint=1000, split='train',
                                                    normal_channel=False, augm = True)
TEST_DATASET = DataLoader(root=DATA_PATH, npoint=1000, split='test',
                                                    normal_channel=False, augm = True)

dataset = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=b_size, shuffle=True, num_workers=0)
dataset_test = torch.utils.data.DataLoader(TEST_DATASET, batch_size=b_size, shuffle=True, num_workers=0)

basis = DiffusionNetBasis(k=30,n_block=12, feature_transform=False)
checkpoint = torch.load(outf + '/basis_model_best_animal.pth')
basis.load_state_dict(checkpoint)
basis.cuda()

classifier = DiffusionNetDesc(k=40,n_block=12, feature_transform=False)
checkpoint = torch.load(outf + '/desc_model_best.pth')
classifier.load_state_dict(checkpoint)
classifier.cuda()

optimizer = optim.Adam([{'params':classifier.parameters()}], lr=0.0001, betas=(0.9, 0.999))#
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
classifier.cuda()

best_eval_loss = np.inf;

train_losses = [];
eval_losses = [];


# Descriptors loss
def desc_loss(pc_A, pc_B, phi_A, phi_B, G_A, G_B):
    p_inv_phi_A = torch.pinverse(phi_A)
    p_inv_phi_B = torch.pinverse(phi_B)
    c_G_A = torch.matmul(p_inv_phi_A, G_A)
    c_G_B = torch.matmul(p_inv_phi_B, G_B)
    c_G_At = torch.transpose(c_G_A,2,1)
    c_G_Bt = torch.transpose(c_G_B,2,1)

    # Estimated C
    C_my = torch.matmul(c_G_A,torch.transpose(torch.pinverse(c_G_Bt),2,1))

    # Optimal C
    C_opt = torch.matmul(p_inv_phi_A, phi_B)

    # MSE
    eucl_loss = torch.mean(torch.square(C_opt - C_my))

    return eucl_loss


# Training Loop
for epoch in range(400):
    #scheduler.step()
    train_loss = 0
    eval_loss = 0
    # Training single Epoch
    for data in tqdm(dataset, 0):
        points = data[0]
        mass= data[1]
        evecs=data[4][:,:,:50]
        evals= data[3][:,:50]
        lap=data[2]
        gradx=data[5]
        grady=data[6]
        points = points.cuda().to(device)
        mass=mass.cuda().to(device)
        evecs=evecs.cuda().to(device)
        evals=evals.cuda().to(device)
        gradx=gradx.cuda().to(device)
        grady=grady.cuda().to(device)
        optimizer.zero_grad()
        classifier = classifier.train()

        with torch.no_grad():
            basis = basis.eval()
            pred = basis(points,mass,lap,evals,evecs,gradx,grady)
            basis_A = pred[1:,:,:20]; basis_B = pred[:-1,:,:20] 
            pc_A = points[1:,:,:]; pc_B = points[:-1,:,:]

        desc= classifier(points,mass,lap,evals,evecs,gradx,grady)
        desc_A = desc[1:,:,:]; desc_B = desc[:-1,:,:]
        eucl_loss = desc_loss(pc_A, pc_B, basis_A, basis_B, desc_A, desc_B)

        eucl_loss.backward()
        optimizer.step()
        train_loss += eucl_loss.item()
    
    # Validation
    for data in tqdm(dataset_test, 0):  
        points = data[0]
        mass= data[1]
        evecs=data[4][:,:,:50]
        evals= data[3][:,:50]
        lap=data[2]
        gradx=data[5]
        grady=data[6]
        points = points.cuda().to(device)
        mass=mass.cuda().to(device)
        evecs=evecs.cuda().to(device)
        evals=evals.cuda().to(device)
        gradx=gradx.cuda().to(device)
        grady=grady.cuda().to(device)
        with torch.no_grad():
            basis = basis.eval()
            classifier=classifier.eval()
            pred = basis(points,mass,lap,evals,evecs,gradx,grady)
            basis_A = pred[1:,:,:20]; basis_B = pred[:-1,:,:20] 
            pc_A = points[1:,:,:]; pc_B = points[:-1,:,:]

            desc= classifier(points,mass,lap,evals,evecs,gradx,grady)
            desc_A = desc[1:,:,:]; desc_B = desc[:-1,:,:]
            eucl_loss = desc_loss(pc_A, pc_B, basis_A, basis_B, desc_A, desc_B)

            eval_loss += eucl_loss.item()

    print('EPOCH ' + str(epoch) + ' - eva_loss: ' + str(eval_loss))

    if eval_loss <  best_eval_loss:
        print('save model')
        best_eval_loss = eval_loss
        torch.save(classifier.state_dict(), '%s/desc_model_best_animal.pth' % (outf))
        torch.save(basis.state_dict(), '%s/basis_model_best_animal.pth' % (outf))

    train_losses.append(train_loss)
    eval_losses.append(eval_loss)

    np.save(outf+'/train_losses_desc_animal.npy',train_losses)
    np.save(outf+'/eval_losses_desc_animal.npy',eval_losses)