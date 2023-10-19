import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy


def knn_query(X, Y, k=1, return_distance=False, n_jobs=1):
    tree = NearestNeighbors(n_neighbors=k, leaf_size=40, algorithm="kd_tree", n_jobs=n_jobs)
    tree.fit(X)
    dists, matches = tree.kneighbors(Y)

    if k == 1:
        dists = dists.squeeze()
        matches = matches.squeeze()

    if return_distance:
        return dists, matches
    return matches

def map_fit(basis1,basis2,desc1, desc2):
    '''
    Given a couple of basis, and a couple of descriptors. This function computes the functional map in a closed form.
    '''
    F= np.matmul(np.linalg.pinv(basis1),desc1)
    G= np.matmul(np.linalg.pinv(basis2),desc2)
    return  np.linalg.lstsq(F.T,G.T,rcond=None)[0].T  #from 1 to 2


def FM_to_p2p(fmap12, basis1,basis2,adj=False,bijective=False): 
    '''
    given a functional map from 1 to 2 and a couple of basis, this funtion converts it to the point to point correspondence
    parameters: adj---- if TRUE the function uses the adjoint conversion method, if FALSE the standard one
                bijective--- if TRUE the function uses the bijective conversion method
    '''
    if adj:
        emb2=np.matmul(basis2,fmap12)
        emb1=basis1
    else:
        emb1=np.matmul(basis1,fmap12.T)
        emb2=basis2
    if bijective:
        p2p=knn_query(emb2,emb1)        #from 1 to 2
    else:
        p2p=knn_query(emb1,emb2)        #from 2 to 1
    return p2p   #(n1,)


def p2p_to_FM(p2p,basis1,basis2, bijective=False):

    '''
    given correspondence and a couple of basis, this funtion converts it to the functional map from 1 to 2
    parameters: bijective--- if TRUE p2p goes from 1 ro 2 and the function uses the bijective conversion method
                             if FALSE teh p2p goes from 2 to 1
    '''
    # Pulled back eigenvectors
    if bijective:
        basis2_pb = basis2[p2p, :]
        return scipy.linalg.lstsq(basis2_pb,basis1)[0]
    else:
        basis1_pb = basis1[p2p, :]
        return scipy.linalg.lstsq(basis2,basis1_pb)[0]

    # Solve with least square
    #return scipy.linalg.lstsq(basis2,evects1_pb)[0]#scipy.linalg.lstsq(evects1_pb,basis2)[0].T # (k2,k1)

