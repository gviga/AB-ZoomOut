from __future__ import print_function
import torch
#pacchetto torch.nn per creare la rete di convoluzione
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import diffusion_net

############ BASIS NETWORK

#costruisco le architetture

class DiffusionNetBasis(nn.Module):
    def __init__(self, k = 30, n_block=12, feature_transform=False):
        super(DiffusionNetBasis, self).__init__()
        self.k = k
        self.diffusion=diffusion_net.layers.DiffusionNet(C_in=3,C_out=k,C_width=128,N_block=n_block, dropout=True)


    def forward(self,  x,mass,lap,evals,evecs,gradx,grady):
        batchsize = x.size()[0]
        n_pts = x.size()[1]
        x = self.diffusion(x,mass,lap,evals,evecs,gradx,grady)
        x = x.view(batchsize, n_pts, self.k)
    
        return x

############ DESC NETWORK
class DiffusionNetDesc(nn.Module):
    def __init__(self, k = 30, n_block=12, feature_transform=False):
        super(DiffusionNetDesc, self).__init__()
        self.k = k
        self.diffusion=diffusion_net.layers.DiffusionNet(C_in=3,C_out=k,C_width=128,N_block=n_block, dropout=True)


    def forward(self,  x,mass,lap,evals,evecs,gradx,grady):
        batchsize = x.size()[0]
        n_pts = x.size()[1]
        x = self.diffusion(x,mass,lap,evals,evecs,gradx,grady)
        x = x.view(batchsize, n_pts, self.k)
    
        return x




class DiffusionNetBasis(nn.Module):
    def __init__(self, k = 30, n_block=12, feature_transform=False):
        super(DiffusionNetBasis, self).__init__()
        self.k = k
        self.diffusion=diffusion_net.layers.DiffusionNet(C_in=3,C_out=k,C_width=128,N_block=n_block, dropout=True)


    def forward(self,  x,mass,lap,evals,evecs,gradx,grady):
        batchsize = x.size()[0]
        n_pts = x.size()[1]
        x = self.diffusion(x,mass,lap,evals,evecs,gradx,grady)
        x = x.view(batchsize, n_pts, self.k)
    
        return x

class Embedding_field(nn.Module):
    
    def __init__(self, hidden_layers=2, neurons_per_layer=128, input_dimension=3,output_dimension=3):
        super().__init__()

        self.input_layer = nn.Linear(input_dimension, neurons_per_layer)
        # Custom initialization for hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons_per_layer, neurons_per_layer) for i in range(hidden_layers)])
        self.output_layer = nn.Linear(neurons_per_layer, output_dimension)
        self.last_layer = nn.Linear(output_dimension,3)


    def forward(self, input):

        x = F.sigmoid(self.input_layer(input))
        for layer in self.hidden_layers:
            x = F.sigmoid(layer(x))
        
        feat=self.output_layer(x)

        return self.last_layer(F.sigmoid(feat)),feat