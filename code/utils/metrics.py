import numpy as np



def eval_bijectivity(p2p,p2p_gt):
    '''
    code for evaluate the bijectivity constraint: || Pi @ 1 - 1 ||
    '''
    aux=np.zeros((p2p_gt.shape[0],))

    for idx,j in enumerate(p2p_gt):
        aux[idx] = np.sum(p2p==j)-1

    return np.linalg.norm(aux)


def eval_injectivity2(p2p,p2p_gt):
    '''
    code for evaluate the bijectivity constraint: || max(Pi @ 1 - 1,0) ||
    '''
    aux=np.zeros((p2p.shape[0],))

    for idx,j in enumerate(p2p_gt):
        aux[idx] =np.max([ np.sum(p2p==j)-1,0])

    return np.linalg.norm(aux)

def eval_orthogonality(fmap):
    return np.linalg.norm(fmap@fmap.T-np.eye(fmap.shape[0]))
    