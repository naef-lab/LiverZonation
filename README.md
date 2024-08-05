# Lobular Coordinate Inference from scRNA-seq

## Overview
This repository is dedicated to the inference of lobular coordinates from single-cell RNA sequencing (scRNA-seq) data. Our focus is on implementing and comparing three distinct computational approaches to address this challenge. This repository contains the code used in the study:
[*A sexually dimorphic hepatic cycle of very low density lipoprotein uptake and assembly*](https://www.biorxiv.org/content/10.1101/2023.10.07.561324v2.abstract).

## Approaches
1. **Maximum Likelihood Estimation (MLE)**: Leveraging PyTorch, this method focuses on optimizing the likelihood function directly. METHOD USED IN THE ARTICLE
2. **Stochastic Variational Inference (SVI)**: Utilizing the NumPyro library, which is built on Jax, SVI offers a scalable Bayesian inference method.
3. **Markov Chain Monte Carlo (MCMC) with NUTS Algorithm**: This approach, also using NumPyro, employs the No-U-Turn Sampler (NUTS), an advanced MCMC method.

## Configuration
The `config.py` file is central to the setup, containing key metaparameters such as:
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

For Jax approaches:
 - jax 
 - jaxlib
 - numpyro

For PyTorch approaches:
 - torch 
 - torchvision 
 - torchaudio


## Contact
Contact andrea.salati@epfl.ch
