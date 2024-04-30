import torch
from torch.nn import Sequential as Seq, Dropout, LeakyReLU, Linear
from torch_geometric.nn import global_max_pool 
import torch.nn as nn
import torch.optim as optim

import deltaconv

#from deltaconv.models import DeltaNetBase
#from deltaconv.nn import MLP
import torch.nn.functional as F


class DeltaConvBasis(torch.nn.Module):
    def __init__(self, in_channels, k=30, conv_channels=[64, 128, 256], mlp_depth=2, embedding_size=1024, num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1):
        """Linearly-invariant-embedding for  Point Clouds with DeltaConv.
        The architecture is based on the architecture used by DGCNN (https://dl.acm.org/doi/10.1145/3326362.

        Args:
            in_channels (int): the number of channels provided as input.
            num_classes (int): the number of classes to segment.
            conv_channels (list[int]): the number of output channels of each convolution.
            mlp_depth (int): the depth of the MLPs of each convolution.
            embedding_size (int): the embedding size.
            categorical_vector(bool, optional): whether to use the categorical encoding in the model.
                Many authors use this in their models for ShapeNet.
            num_neighbors (int): the number of neighbors to use in estimating the gradient.
            grad_regularizer (float): the regularizer value used in the least-squares fitting procedure.
                In the paper, this value is referred to as \lambda.
                Larger grad_regularizer gives a smoother, but less accurate gradient.
                Lower grad_regularizer gives a more accurate, but more variable gradient.
                The grad_regularizer value should be >0 (e.g., 1e-4) to prevent exploding values.
            grad_kernel_width (float): the width of the gaussian kernel used to weight the
                least-squares problem to approximate the gradient.
                Larger kernel width means that more points are included, which is a 'smoother' gradient.
                Lower kernel width gives a more accurate, but possibly noisier gradient.
        """
        super().__init__()
        self.k=k
        self.deltanet_base = deltaconv.models.DeltaNetBase(in_channels, conv_channels, mlp_depth, num_neighbors, grad_regularizer, grad_kernel_width)
        # Global embedding
        self.lin_embedding = deltaconv.nn.MLP([sum(conv_channels), embedding_size])
        #basis
        self.segmentation_head = Seq(
            deltaconv.nn.MLP([embedding_size+sum(conv_channels), 256]), Dropout(0.3),
            Linear(256, 128), Dropout(0.3), LeakyReLU(negative_slope=0.2), Linear(128, self.k))


    def forward(self, data):

        conv_out = self.deltanet_base(data)
        x = torch.cat(conv_out, dim=1)
        x = self.lin_embedding(x)

        batch = data.batch
        x_max = global_max_pool(x, batch)[batch]

        x = torch.cat([x_max] + conv_out, dim=1)

        return self.segmentation_head(x)


class DeltaConvDesc(torch.nn.Module):
    def __init__(self, in_channels, k=40, conv_channels=[64, 128, 256], mlp_depth=2, embedding_size=1024, num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1):
        """Linearly-invariant-embedding for  Point Clouds with DeltaConv.
        The architecture is based on the architecture used by DGCNN (https://dl.acm.org/doi/10.1145/3326362.

        Args:
            in_channels (int): the number of channels provided as input.
            num_classes (int): the number of classes to segment.
            conv_channels (list[int]): the number of output channels of each convolution.
            mlp_depth (int): the depth of the MLPs of each convolution.
            embedding_size (int): the embedding size.
            categorical_vector(bool, optional): whether to use the categorical encoding in the model.
                Many authors use this in their models for ShapeNet.
            num_neighbors (int): the number of neighbors to use in estimating the gradient.
            grad_regularizer (float): the regularizer value used in the least-squares fitting procedure.
                In the paper, this value is referred to as \lambda.
                Larger grad_regularizer gives a smoother, but less accurate gradient.
                Lower grad_regularizer gives a more accurate, but more variable gradient.
                The grad_regularizer value should be >0 (e.g., 1e-4) to prevent exploding values.
            grad_kernel_width (float): the width of the gaussian kernel used to weight the
                least-squares problem to approximate the gradient.
                Larger kernel width means that more points are included, which is a 'smoother' gradient.
                Lower kernel width gives a more accurate, but possibly noisier gradient.
        """
        super().__init__()
        self.k=k
        self.deltanet_base = deltaconv.models.DeltaNetBase(in_channels, conv_channels, mlp_depth, num_neighbors, grad_regularizer, grad_kernel_width)
        # Global embedding
        self.lin_embedding = deltaconv.nn.MLP([sum(conv_channels), embedding_size])
        #basis
        self.segmentation_head = Seq(
            deltaconv.nn.MLP([embedding_size+sum(conv_channels), 256]), Dropout(0.3),  deltaconv.nn.MLP([256, 256]), Dropout(0.3),
            Linear(256, 128), LeakyReLU(negative_slope=0.2), Linear(128, self.k))


    def forward(self, data):

        conv_out = self.deltanet_base(data)

        x = torch.cat(conv_out, dim=1)
        x = self.lin_embedding(x)

        batch = data.batch
        x_max = global_max_pool(x, batch)[batch]

        x = torch.cat([x_max] + conv_out, dim=1)

        return self.segmentation_head(x)
