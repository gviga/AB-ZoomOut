# Adjoint Bijective ZoomOut: Efficient upsampling for learned linearly-invariant embedding
This repository is the official implementation of Adjoint Bijective ZoomOu, by G.Viganò and S. Melzi.
This code was written by Giulio Viganò (https://sites.google.com/view/gvigano/home-page).

To train the networks, we refer to the repository of "Correspondence Learning via Lineraly-invariant Embedding" https://github.com/riccardomarin/Diff-FMAPs-PyTorch.
You can find a pre-trained version of the network in the training repository.

To replicate the Paper's results on the FAUST dataset, download the data from https://github.com/riccardomarin/Diff-FMAPs-PyTorch, put the FAUST file in the data folder, and run FAUST_test.py.
In the folder trained, you can find files and pre-trained models to train the models with Pointnet and Pointnet++ backbones. The details on the choice can be found in our latest work ().




# Reference
If you use this code, please cite our paper.

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
