#Libraries
import numpy as np
import torch
import hdf5storage

import sys

from utils.matching_fun import *
from utils.data_preprocess import *
from utils.model import PointNetBasis,PointNetDesc
from utils.metrics import *



# Loading Data
Noise=False

DATA_PATH = '../data/FAUST_noise_0.01.mat'
dd = hdf5storage.loadmat(DATA_PATH)
if Noise is True:
    v = dd['vertices'].astype(np.float32)
else:
    v = dd['vertices_clean'].astype(np.float32)


#loading geodesic distance matrix (for evaluation)
geod_load = hdf5storage.loadmat('../data/N_out.mat')
D = geod_load['D'].astype(np.float32)
A_geod = D
p2p_gt = np.arange(1000)

#number of pairs considered
n_esp=50



#Loading Models
k=20
k_tilde=30
#BASIS and DESCRIPTORS for initialization
basis_model_init = PointNetBasis(k=k, feature_transform=False)
desc_model_init = PointNetDesc(k=40, feature_transform=False)
checkpoint = torch.load('../training/pretrained_models/basis_model_best_'+str(k)+'.pth',map_location=torch.device('cpu'))
basis_model_init.load_state_dict(checkpoint)
checkpoint = torch.load('../training/pretrained_models/desc_model_best_20.pth',map_location=torch.device('cpu'))
desc_model_init.load_state_dict(checkpoint)
basis_model_init = basis_model_init.eval()
desc_model_init = desc_model_init.eval()

#BASIS for Adjoint Bijective ZOOMOUT 
basis_model_zo = PointNetBasis(k=k_tilde, feature_transform=False)
checkpoint = torch.load('../training/pretrained_models/basis_model_best_'+str(k_tilde)+'.pth',map_location=torch.device('cpu'))
basis_model_zo.load_state_dict(checkpoint)
basis_model_zo = basis_model_zo.eval()

# with these parameter we can chose whic kind of conversion we want to use
bijective=True
adj=True

np.random.seed(2)
#saving estimated maps and metrics for initialized maps
match_lie=np.zeros((n_esp,p2p_gt.shape[0]))
err_lie_geod=np.zeros((n_esp,p2p_gt.shape[0]))
err_lie_eu=np.zeros((n_esp,p2p_gt.shape[0]))
ortho_lie=np.zeros((n_esp,))
bij_lie=np.zeros((n_esp,))
inj_lie=np.zeros((n_esp,))
conversion_err_lie=np.zeros((n_esp,p2p_gt.shape[0]))

#for refinemd maps
match_lie_OUR=np.zeros((n_esp,p2p_gt.shape[0],11))
err_lie_OUR_geod=np.zeros((n_esp,p2p_gt.shape[0],11))
err_lie_OUR_eu=np.zeros((n_esp,p2p_gt.shape[0],11))
ortho_lie_OUR=np.zeros((n_esp,11))
bij_lie_OUR=np.zeros((n_esp,11))
inj_lie_OUR=np.zeros((n_esp,11))

for i in range(n_esp):
    #shape selection
    vec = np.random.randint(100,size=2)
    if bijective:
        shapen1 = vec[1]
        shapen2 = vec[0]
    else:
        shapen1 = vec[0]
        shapen2 = vec[1]
    #save shapes
    v1 = v[shapen1,:,:].squeeze()
    v2 = v[shapen2,:,:].squeeze()

    ####lie###
    # Computing Basis and Descriptors
    pred_basis = basis_model_init(torch.transpose(torch.from_numpy(v[[shapen1,shapen2],:,:].astype(np.float32)),1,2))
    basis = pred_basis[0].detach().numpy()
    pred_desc = desc_model_init(torch.transpose(torch.from_numpy(v[[shapen1,shapen2],:,:].astype(np.float32)),1,2))
    desc = pred_desc[0].detach().numpy()
    # Saving basis amd descriptors
    basis1 = np.squeeze(basis[0])
    basis2 = np.squeeze(basis[1])
    desc1 = np.squeeze(desc[0])
    desc2 = np.squeeze(desc[1])

    #map optimization
    fmap = map_fit(basis1,basis2,desc1,desc2)   #from v1 to v2

    #map conversion (if bijective p2p from v1 to v2), (else p2p from v2 to v1)
    p2p = FM_to_p2p(fmap, basis1, basis2,adj, bijective)
    match_lie[i,:]=p2p
    
    #evaluation module
    err_lie_geod[i,:] =A_geod[(p2p, p2p_gt)]
    if bijective: 
        err_lie_eu[i,:] = np.sqrt(np.sum(np.square(v2[p2p_gt.astype(np.int32)] - v2[p2p,:]),1))    #euclidean error
    else:
        err_lie_eu[i,:] = np.sqrt(np.sum(np.square(v1[p2p_gt.astype(np.int32)] - v1[p2p,:]),1))     #euclidean error

    bij_lie[i]=eval_bijectivity(p2p,p2p_gt)
    ortho_lie[i]=eval_orthogonality(fmap)

    # Computing Basis and Descriptors for refinement
    pred_basis_zo = basis_model_zo(torch.transpose(torch.from_numpy(v[[shapen1,shapen2],:,:].astype(np.float32)),1,2))
    basis_zo = pred_basis_zo[0].detach().numpy()
    basis1_zo = np.squeeze(basis_zo[0])
    basis2_zo = np.squeeze(basis_zo[1])
    
    p2p=match_lie[i,:].astype(np.int32)
    
    fmap_zo = p2p_to_FM(p2p, basis1_zo[:,:k], basis2_zo[:,:k],bijective)
    ortho_lie_OUR[i,0]=eval_orthogonality(fmap_zo)

    #faccio l'iterazione
    for l in range(k,k_tilde):
        p2p_zo = FM_to_p2p(fmap_zo,basis1_zo[:,:l], basis2_zo[:,:l],adj,bijective)
        match_lie_OUR[i,:,l-k]=p2p_zo
        #evaluation module
        err_lie_OUR_geod[i,:,l-k] =A_geod[(p2p_zo, p2p_gt)]
        if bijective: 
            err_lie_OUR_eu[i,:,l-k] = np.sqrt(np.sum(np.square(v2[p2p_gt.astype(np.int32)] - v2[p2p_zo,:]),1))    #euclidean error
        else:
            err_lie_OUR_eu[i,:,l-k] = np.sqrt(np.sum(np.square(v1[p2p_gt.astype(np.int32)] - v1[p2p_zo,:]),1))     #euclidean error

        bij_lie_OUR[i,l-k]=eval_bijectivity(p2p_zo,p2p_gt)

        fmap_zo = p2p_to_FM(p2p_zo, basis1_zo[:,:l+1], basis2_zo[:,:l+1],bijective)
        ortho_lie_OUR[i,l+1-k]=eval_orthogonality(fmap_zo)

    p2p_zo = FM_to_p2p(fmap_zo,basis1_zo, basis2_zo,adj,bijective)
    match_lie_OUR[i,:,l+1-k]=p2p_zo
    #evaluation module
    err_lie_OUR_geod[i,:,l+1-k] =A_geod[(p2p_zo, p2p_gt)]
    if bijective: 
        err_lie_OUR_eu[i,:,l+1-k] = np.sqrt(np.sum(np.square(v2[p2p_gt.astype(np.int32)] - v2[p2p_zo,:]),1))    #euclidean error
    else:
        err_lie_OUR_eu[i,:,l+1-k] = np.sqrt(np.sum(np.square(v1[p2p_gt.astype(np.int32)] - v1[p2p_zo,:]),1))     #euclidean error

    bij_lie_OUR[i,l+1-k]=eval_bijectivity(p2p_zo,p2p_gt)


    
print(f'Geodesic mean error LieB+Ours: {np.mean(np.mean(err_lie_OUR_geod[:,:,10],1),0):4f} \n'
      f'Eucliden mean error LieB+Ours: {np.mean(np.mean(err_lie_OUR_eu[:,:,10],1),0):4f} \n'
      f'Bijectivity error LieB+Ours: {np.mean(bij_lie_OUR[:,10]):4f} \n'
      f'Orthogonality error LieB+Ours: {np.mean(ortho_lie_OUR[:,10]):4f} \n')

print(f'Geodesic mean error LieB: {np.mean(np.mean(err_lie_geod,1),0):4f} \n'
      f'Euclidean mean error LieB: {np.mean(np.mean(err_lie_eu,1),0):4f} \n'
      f'Bijectivity error LieB: {np.mean(bij_lie):4f} \n'
      f'Orthogonality error: {np.mean(ortho_lie):4f} \n')
