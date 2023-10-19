import numpy as np


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
    return point,centroids.astype(np.int32)


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def fps_random_sampling(v , n=1000, ratio=0.5):
   
    _,fps_idx=farthest_point_sample(v,int(n*ratio))
    new_idx=np.delete(np.arange(v.shape[0]),fps_idx)
    random_idx=np.random.choice(new_idx,int(n*(1-ratio)), replace=False)
    random_idx.shape
    idx=np.concatenate((fps_idx, random_idx))
    idx.sort()
    return v[idx],idx