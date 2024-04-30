#this code is the dataloader for a dataset whit the operators used to train feature using diffusionNet

import numpy as np
import warnings
import os
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
import robust_laplacian
from diffusion_net import geometry
import torch
from tqdm import tqdm


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


dir_path='./data/surreal_diffnet/'

class Surr12kModelNetDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, k_eig=128, split='train', uniform=False, normal_channel=False, cache_size=15000, augm = False):
        self.uniform = uniform
        self.augm = augm
        self.k_eig=k_eig
        np.random.seed(0)
        #splitta training set e test set
        if split =="train":
            self.split='train'
            self.data = np.load(os.path.join(root, "12k_shapes_train.npy")).astype(dtype=np.float32)
            self.indexes=np.arange(self.data.shape[0])
            #self.data = self.data
        if split =="test":
            self.data = np.load(os.path.join(root, "12k_shapes_test.npy")).astype(dtype=np.float32)
            self.indexes=np.arange(self.data.shape[0])

            self.split='test'
        #EDGES_PATH = os.path.join(root,"12ktemplate.ply")

    def __len__(self):
        return len(self.data)

    def _get_item(self, index):
        #load lazily
        if self.split=='train':
            point_set=np.load('./data/surreal_diffnet/surreal_train_'+str(self.indexes[index])+'_vert.npy').astype(dtype=np.float32)
            mass=np.load('./data/surreal_diffnet/surreal_train_'+str(self.indexes[index])+'_mass.npy').astype(dtype=np.float32)
            evecs=np.load('./data/surreal_diffnet/surreal_train_'+str(self.indexes[index])+'_evecs.npy').astype(dtype=np.float32)
            L=np.load('./data/surreal_diffnet/surreal_train_'+str(self.indexes[index])+'_lap.npy').astype(dtype=np.float32)
            evals=np.load('./data/surreal_diffnet/surreal_train_'+str(self.indexes[index])+'_evals.npy').astype(dtype=np.float32)
            gradx=torch.from_numpy(np.load('./data/surreal_diffnet/surreal_train_'+str(self.indexes[index])+'_gradx.npy').astype(dtype=np.float32)).to_sparse()
            grady=torch.from_numpy(np.load('./data/surreal_diffnet/surreal_train_'+str(self.indexes[index])+'_grady.npy').astype(dtype=np.float32)).to_sparse()
        else:
            point_set=np.load('./data/surreal_diffnet/surreal_test_'+str(self.indexes[index])+'_vert.npy').astype(dtype=np.float32)
            mass=np.load('./data/surreal_diffnet/surreal_test_'+str(self.indexes[index])+'_mass.npy').astype(dtype=np.float32)
            evecs=np.load('./data/surreal_diffnet/surreal_test_'+str(self.indexes[index])+'_evecs.npy').astype(dtype=np.float32)
            L=np.load('./data/surreal_diffnet/surreal_test_'+str(self.indexes[index])+'_lap.npy').astype(dtype=np.float32)
            evals=np.load('./data/surreal_diffnet/surreal_test_'+str(self.indexes[index])+'_evals.npy').astype(dtype=np.float32)
            gradx=torch.from_numpy(np.load('./data/surreal_diffnet/surreal_test_'+str(self.indexes[index])+'_gradx.npy').astype(dtype=np.float32)).to_sparse()
            grady=torch.from_numpy(np.load('./data/surreal_diffnet/surreal_test_'+str(self.indexes[index])+'_grady.npy').astype(dtype=np.float32)).to_sparse()            

        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        if self.augm:
            point_set = data_augmentation(point_set)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        return point_set, mass,L,evals,evecs,gradx,grady

    def __getitem__(self, index):
        return self._get_item(index)



dir_path='./data/animal_diffnet/'

class AnimalDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, k_eig=50, split='train', uniform=False, normal_channel=False, cache_size=15000, augm = False):
        self.uniform = uniform
        self.augm = augm
        self.k_eig=k_eig
        np.random.seed(0)

        if split =="train":
            self.split='train'
            self.data = np.load(os.path.join("./data/animal_train.npy"))
            self.indexes=np.arange(self.data.shape[0])
            #self.data = self.data
        if split =="test":
            self.split='test'
            self.data = np.load(os.path.join("./data/animal_test.npy"))
            self.indexes=np.arange(self.data.shape[0])
        #splitta training set e test set
        
            
        #EDGES_PATH = os.path.join(root,"12ktemplate.ply")

    def __len__(self):
        return len(self.data)

    def _get_item(self, index):
        #load lazily
        if self.split=='train':
            point_set=np.load('./data/animal_diffnet/animal_train_'+str(self.indexes[index])+'_vert.npy').astype(dtype=np.float32)
            mass=np.load('./data/animal_diffnet/animal_train_'+str(self.indexes[index])+'_mass.npy').astype(dtype=np.float32)
            evecs=np.load('./data/animal_diffnet/animal_train_'+str(self.indexes[index])+'_evecs.npy').astype(dtype=np.float32)
            L=np.load('./data/animal_diffnet/animal_train_'+str(self.indexes[index])+'_lap.npy').astype(dtype=np.float32)
            evals=np.load('./data/animal_diffnet/animal_train_'+str(self.indexes[index])+'_evals.npy').astype(dtype=np.float32)
            gradx=torch.from_numpy(np.load('./data/animal_diffnet/animal_train_'+str(self.indexes[index])+'_gradx.npy').astype(dtype=np.float32)).to_sparse()
            grady=torch.from_numpy(np.load('./data/animal_diffnet/animal_train_'+str(self.indexes[index])+'_grady.npy').astype(dtype=np.float32)).to_sparse()
        else:
            point_set=np.load('./data/animal_diffnet/animal_test_'+str(self.indexes[index])+'_vert.npy').astype(dtype=np.float32)
            mass=np.load('./data/animal_diffnet/animal_test_'+str(self.indexes[index])+'_mass.npy').astype(dtype=np.float32)
            evecs=np.load('./data/animal_diffnet/animal_test_'+str(self.indexes[index])+'_evecs.npy').astype(dtype=np.float32)
            L=np.load('./data/animal_diffnet/animal_test_'+str(self.indexes[index])+'_lap.npy').astype(dtype=np.float32)
            evals=np.load('./data/animal_diffnet/animal_test_'+str(self.indexes[index])+'_evals.npy').astype(dtype=np.float32)
            gradx=torch.from_numpy(np.load('./data/animal_diffnet/animal_test_'+str(self.indexes[index])+'_gradx.npy').astype(dtype=np.float32)).to_sparse()
            grady=torch.from_numpy(np.load('./data/animal_diffnet/animal_test_'+str(self.indexes[index])+'_grady.npy').astype(dtype=np.float32)).to_sparse()            

        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        if self.augm:
            point_set = data_augmentation(point_set)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        return point_set, mass,L,evals,evecs,gradx,grady

    def __getitem__(self, index):
        return self._get_item(index)



'''
#create the dataset for diffusion net

data = np.load(os.path.join("./data/animal_train.npy"))

data_list=torch.tensor(data)
for i in tqdm(range(data.shape[0])):
    
    face=torch.empty(0)
    vert=torch.tensor(data[i])
    frames, mass, L, evals, evecs, gradX, gradY = geometry.compute_operators(vert, face, 128)

    #saving
    np.save('./data/animal_diffnet/animal_train_'+str(i)+'_vert.npy',vert)
    np.save('./data/animal_diffnet/animal_train_'+str(i)+'_frames.npy',np.array(frames))
    np.save('./data/animal_diffnet/animal_train_'+str(i)+'_mass.npy',np.array(mass))
    np.save('./data/animal_diffnet/animal_train_'+str(i)+'_lap.npy',np.array(L.to_dense()))
    np.save('./data/animal_diffnet/animal_train_'+str(i)+'_evals.npy',np.array(evals))
    np.save('./data/animal_diffnet/animal_train_'+str(i)+'_evecs.npy',np.array(evecs))
    np.save('./data/animal_diffnet/animal_train_'+str(i)+'_gradx.npy',np.array(gradX.to_dense()))
    np.save('./data/animal_diffnet/animal_train_'+str(i)+'_grady.npy',np.array(gradY.to_dense()))

'''