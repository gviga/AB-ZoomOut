# Adjoint Bijective ZoomOut: Efficient upsampling for learned linearly-invariant embedding
This repository is the official implementation of Adjoint Bijective ZoomOut, by G.Viganò and S. Melzi.
This code was written by Giulio Viganò (https://sites.google.com/view/gvigano/home-page).

The algorithm goal is to refine maps to solve the correspondence problem in the case of point clouds.
In particular, this algorithm is designed to refine maps in case of Learned lienraly invariant embedding.
For this reason we report here the code to compute these embeddings with the following backbones:

- PointNet, as in ( https://arxiv.org/abs/2010.13136, https://www.boa.unimib.it/handle/10281/456707).
- Pointnet++, as in (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4772787).
- DiffusionNet
- Deltaconv

For the original implementation of the code to train the networks, we refer to the repository of "Correspondence Learning via Lineraly-invariant Embedding" https://github.com/riccardomarin/Diff-FMAPs-PyTorch.
You can find a pre-trained version of some of the network in the training repository.

To replicate the Paper's results on the FAUST dataset, download the data from https://github.com/riccardomarin/Diff-FMAPs-PyTorch, put the FAUST file in the data folder, and run FAUST_test.py.

# Reference
If you use this code, please cite our works.

@inproceedings {10.2312:stag.20231293,
booktitle = {Smart Tools and Applications in Graphics - Eurographics Italian Chapter Conference},
editor = {Banterle, Francesco and Caggianese, Giuseppe and Capece, Nicola and Erra, Ugo and Lupinetti, Katia and Manfredi, Gilda},
title = {{Adjoint Bijective ZoomOut: Efficient Upsampling for Learned Linearly-invariant Embedding}},
author = {Viganò, Giulio and Melzi, Simone},
year = {2023},
publisher = {The Eurographics Association},
ISSN = {2617-4855},
ISBN = {978-3-03868-235-6},
DOI = {10.2312/stag.20231293}
}

For any issue, please contact the Authors. 
