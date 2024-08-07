# Lobular Coordinate Inference from scRNA-seq

## Overview
This repository is dedicated to the inference of lobular coordinates from single-cell RNA sequencing (scRNA-seq) data. Our focus is on implementing and comparing three distinct computational approaches to address this challenge. This repository contains the code used in the study:
[*A sexually dimorphic hepatic cycle of very low density lipoprotein uptake and assembly*](https://www.biorxiv.org/content/10.1101/2023.10.07.561324v2.abstract).

## Code for the article
1. `MLE.ipynb` it's the notebook that assigns the lobular coordinate to cells based on their transcriptome. Such coordinate is present in several figures of the paper.
2. `gene_selection.ipynb` This notebook expands the list of zonated genes, leveraging the coordinates found with the smaller gene list.


## Additional code
1. `other_methods/numpyro_SVI.ipynb`: Utilizing the NumPyro library, which is built on JAX, SVI offers a scalable Bayesian inference method, approximating posterior distributions of latent variables.
2. `other_methods/numpyro_MCMC.ipynb`: This approach, also using NumPyro, employs the No-U-Turn Sampler (NUTS), to sample the posterior distributions of all latent variables.

## Configuration
The `config.py` file is central to the setup, containing key hyperparameters such as:
- List of genes used for analysis
- Number of iterations
- Batch size

## Getting Started
In order to run the code the following packages are needed:

 - python
 - numpy
 - matplotlib
 - pip
 - scikit-learn
 - ipykernel
 - pandas
 - seaborn
 - anndata
 - scanpy 
 - python-igraph 
 - leidenalg
 - scipy

For PyTorch approaches:
 - torch 
 - torchvision 
 - torchaudio

For JAX-based approaches (SVI, MCMC):
 - jax 
 - jaxlib
 - numpyro




## Contact
Contact andrea.salati@epfl.ch
