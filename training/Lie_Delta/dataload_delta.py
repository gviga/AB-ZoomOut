import numpy as np
import warnings
import os
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch

warnings.filterwarnings('ignore')

def data_augmentation(point_set):
        theta = np.random.uniform(0, np.pi * 2)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
        #point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter
        return point_set

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    #m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    #pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class Surr12kModelNetDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000, augm = False):
        self.uniform = uniform
        self.augm = augm
        np.random.seed(0)
        #splitta training set e test set
        if split =="train":
            self.pointclouds= np.load(os.path.join(root, "12k_shapes_train.npy")).astype(dtype=np.float32)
            #self.data = self.data
        if split =="test":
            self.pointclouds = np.load(os.path.join(root, "12k_shapes_test.npy")).astype(dtype=np.float32)
            #self.data = self.data
        #valuation set
        if split =="FAUST":
            self.pointclouds = np.load(os.path.join(root, "FAUST_noise.npy")).astype(dtype=np.float32)
      

    def __len__(self):
        return len(self.pointclouds)

    def _get_item(self, index):
        point_set = self.pointclouds[index,:,:]
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        if self.augm:
            point_set = data_augmentation(point_set)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        save = torch.tensor(point_set[:, 0:3], dtype=torch.float32,device='cuda')
        # Create a PyTorch Geometric Data object
        data = Data(pos=save)
        return data
        #return {'pos': point_set, 'x':[]}

    def __getitem__(self, index):
        return self._get_item(index)



class AnimalDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000, augm = False):
        self.uniform = uniform
        self.augm = augm
        np.random.seed(0)
        #splitta training set e test set
        if split =="train":
            self.pointclouds= np.load("./data/animal_train.npy").astype(dtype=np.float32)
            #self.data = self.data
        if split =="test":
            self.pointclouds = np.load("./data/animal_test.npy").astype(dtype=np.float32)
            #self.data = self.data

    def __len__(self):
        return len(self.pointclouds)

    def _get_item(self, index):
        point_set = self.pointclouds[index,:,:]
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        if self.augm:
            point_set = data_augmentation(point_set)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        save = torch.tensor(point_set[:, 0:3], dtype=torch.float32,device='cuda')
        # Create a PyTorch Geometric Data object
        data = Data(pos=save)
        return data
        #return {'pos': point_set, 'x':[]}

    def __getitem__(self, index):
        return self._get_item(index)



if __name__ == '__main__':
    import torch

    data = FAUSTModelNetDataLoader('data/',split='train', uniform=False, normal_channel=True,)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
        