

# Adjoint Bijective ZoomOut: Efficient upsampling for learned linearly-invariant embedding

![teaser github](https://github.com/gviga/AB-ZoomOut/assets/95035641/405f8e13-ff9c-4820-b235-ad6f156010fb)
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


@article{VIGANO2024103985,
title = {Bijective upsampling and learned embedding for point clouds correspondences},
journal = {Computers & Graphics},
pages = {103985},
year = {2024},
issn = {0097-8493},
doi = {https://doi.org/10.1016/j.cag.2024.103985},
url = {https://www.sciencedirect.com/science/article/pii/S0097849324001201},
author = {Giulio Viganò and Simone Melzi},
keywords = {Shape correspondence, Functional maps, Point clouds, Machine learning},
abstract = {In this paper, we present a novel pipeline to compute and refine a data-driven solution for estimating the correspondence between 3D point clouds. Our method is compatible with the functional map framework, so it relies on a functional representation of the correspondence, but, differently from other similar approaches, this method is specifically designed to exploit this functional scenario for point cloud matching. Our new method merges a data-driven approach to compute functional basis and descriptors on the shape’s surface and a new refinement method designed for the learned basis. This refinement algorithm arises from a different way of converting functional operators into point-to-point correspondence, which we prove to promote bijectivity between maps, exploiting a theoretical result. Iterating this procedure and performing basis upsampling in the same way as other similar methods, ours increases the accuracy of the correspondence, leading to more bijective correspondences. Different from other approaches, our method allows us to train a functional basis, considering the refinement stage. Combining our new pipeline with an improved feature extractor, our solution outperforms previous methods in various evaluations and settings. We test our method over different datasets, comprising near-isometric and non-isometric pairs.}
}


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
