
from pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation

import torch
#pacchetto torch.nn per creare la rete di convoluzione
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from time import time


############ BASIS NETWORK

#costruisco le architetture
#architettura per le basi
class PointNetBasis_feat(nn.Module):
    def __init__(self):
        super(PointNetBasis_feat, self).__init__()

        self.sa1 = PointNetSetAbstraction(512, 0.025, 32, 3 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.05, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.1, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.2, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128])

        self.dense1 = torch.nn.Linear(512,256)
        self.dense2 = torch.nn.Linear(256,128)
        #self.conv3 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(256) 
        self.bn2 = nn.BatchNorm1d(128)
        #self.bn3 = nn.BatchNorm1d(128)
        
        #self.drop1 = nn.Dropout(0.3)
        #self.conv4 = nn.Conv1d(128, k, 1)

    def forward(self, xyz):

        l0_points = xyz
        l0_xyz = xyz
        batchsize = xyz.size()[0]
        n_pts = xyz.size()[2]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        #pointwise features
        pointfeat = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        #global features
        # FC layers
        x = F.relu(self.bn1(self.conv1(l4_points)))
        x = torch.max(x, 2, keepdim=True)[0]
        
        x = x.view(-1, 1024)
        x = F.relu(self.bn6(self.dense1(x)))
        x = F.relu(self.bn7(self.dense2(x)))

        x = x.view(-1, 128, 1).repeat(1, 1, n_pts)
        #x = F.relu(self.bn2(self.conv2(x)))
        #x =  self.drop1(F.relu(self.bn3(self.conv3(x))))
        #x = self.conv4(x)
        return  torch.cat([x, pointfeat], 1)
        

#architettura per le basi
class PointNetBasis(nn.Module):
    def __init__(self, k = 20):
        super(PointNetBasis, self).__init__()
        self.k = k
        self.feat = PointNetBasis_feat()
        self.conv1 = torch.nn.Conv1d(512, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.k, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn2b = nn.BatchNorm1d(256)
        self.m   = nn.Dropout(p=0.3)
    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn2b(self.conv3(x)))
        x = self.m(x)
        #x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x

############ DESC NETWORK
class PointNetDesc(nn.Module):
    def __init__(self, k = 40, feature_transform=False):
        super(PointNetDesc, self).__init__()
        self.k=k
        self.sa1 = PointNetSetAbstraction(512, 0.1, 32, 3 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, k])

        #self.conv1 = nn.Conv1d(128, 128, 1)
        #self.conv2 = nn.Conv1d(128, 128, 1)
        #self.conv3 = nn.Conv1d(128, 128, 1)
        #self.bn1 = nn.BatchNorm1d(128)
        #self.bn2 = nn.BatchNorm1d(128)
        #self.bn3 = nn.BatchNorm1d(128)
        
        #self.drop1 = nn.Dropout(0.3)
        #self.conv4 = nn.Conv1d(128, k, 1)

    def forward(self, xyz):

        l0_points = xyz
        l0_xyz = xyz
        batchsize = xyz.size()[0]
        n_pts = xyz.size()[2]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        x = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        # FC layers
        x = F.relu(self.bn1(self.conv1(l0_points)))
        #x = F.relu(self.bn2(self.conv2(x)))
        #x =  self.drop1(F.relu(self.bn3(self.conv3(x))))
    
        #x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = x.view(batchsize, n_pts, self.k)
        return x
        
