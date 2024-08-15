#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import scanpy as sc
import anndata
from sklearn.decomposition import PCA
from .noise_model import Noise_Model


####################################################################################################
# big functions


def get_data_from_anndata(path, gene_list=None, cell_list=None):
    """
    This function loads data from an AnnData object, performs filtering and normalization, and returns several outputs.
    the function also removes the bad replicate, removes mitochondrial genes, and filters cells with an unusual number of reads + too low mito content.

    Parameters:
    path (str): The path to the AnnData object file.
    gene_list (list, optional): A list of gene names to be retrieved. If None, all genes are retrieved.
    cell_list (list, optional): A list of cell names to be retrieved. If None, all cells are retrieved.

    Returns:
    tuple: A tuple containing six elements:
        - data (anndata.AnnData): The filtered and normalized AnnData object.
        - data_zonated (anndata.AnnData): The AnnData object filtered to include only the genes specified in gene_list.
        - n_c (numpy.ndarray): The sum of counts per cell.
        - dm (torch.Tensor): The design matrix created from the sample identifiers.
        - sample_id (numpy.ndarray): An array of sample identifiers, where each unique sample name is mapped to an integer.
        - nn (numpy.ndarray): An array of unique sample names.

    """
    adatas = {"Tomaz": sc.read_h5ad(path)}
    data = anndata.concat(adatas, label="dataset", join="inner")
    sample_names = data.obs["orig.ident"]

    # removing the bad replicate
    remove = "scRNA_Seq_Tomaz_220531-01-M-ZT22-H-lib_1-129S"
    remove_mask = np.invert(sample_names == remove)

    # update data and sample list after the removal of the sample
    data = data[remove_mask, :]

    # filter mitochoncdrial genes
    genes_mito = [s.startswith("mt-") for s in data.var.index]
    genes_Mup = [s.startswith("Mup") for s in data.var.index]
    genes_keep = np.logical_not(np.logical_or(genes_mito, genes_Mup))
    # cut out cells with an unusual number of reads + too low mito content
    cells_keep = np.logical_and(
        data.obs["nCount_RNA"] < 15000, data.obs["percent.mt"] > 3
    )
    data = data[cells_keep, genes_keep]
    NC, NG = data.shape

    # adding fields to the data object
    data.layers["n_cg"] = data.X.toarray()
    data.obs["n_c"] = data.layers["n_cg"].sum(axis=1)
    data.layers["f_cg"] = data.layers["n_cg"] / np.array(data.obs["n_c"]).reshape(
        NC, 1
    )  # [:, None]

    sample_names = data.obs["orig.ident"]
    nn = sample_names.unique()

    sample_id = np.zeros(data.n_obs, dtype=np.int64)
    for i, s in enumerate(nn):
        sample_id[sample_names == s] = i
    # dm = make_design_matrix(torch.tensor(sample_id, dtype=float)) # change
    dm = make_design_matrix_numpy(sample_id)

    data_zonated = data[:, gene_list]
    return data, data_zonated, data.obs["n_c"].values, dm, sample_id, nn


def do_pca(data, pc=0):
    """
    in this fuynction we do pca on the data.
    In our datset the zonation source of variation is clrearly the zonation one,
    however in other datasets, could not be the same case.
    """

    pca = PCA(n_components=5, whiten=False)
    X_pca = data

    # normalize yourself
    X_pca = X_pca - X_pca.mean(axis=0)[None, :]
    X_pca = X_pca / np.sqrt(X_pca.var(axis=0))[None, :]
    PC = pca.fit_transform(X_pca)

    # normalization of the first PC
    x_unif = PC[:, pc]
    x_unif = x_unif - x_unif.mean()
    x_unif = x_unif / np.sqrt(x_unif.var())
    return x_unif


def fit_coeff(data, x_unif, genes, layer="n_cg"):
    """
    This function is used to fit the coefficients a0 and a1 of the model
    using a negative binomial regression.
    """
    noise = "NB"
    D = np.stack((np.repeat(1, data.n_obs), x_unif), axis=1)
    coef_pau = np.zeros((len(genes), D.shape[1]))

    alpha = np.zeros(len(genes))
    logN = np.log(data.obs["n_c"].values)

    for gi, g in enumerate(genes):
        yy = data[:, [g]].layers[layer].toarray().squeeze()
        model_n = Noise_Model(yy, D, logN, noise)

        iterations = 50
        mf = model_n.fit(iterations)
        if noise == "Poisson":
            coef_pau[gi, :] = mf
        else:
            coef_pau[gi, :] = mf[:-1]
            alpha[gi] = mf[-1]
    return coef_pau


def save_parameters(x, a0, a1, sample_names_uniq, name, data):
    """
    This function is used to save the parameters to a .txt file.
    """
    df_a0 = pd.DataFrame(a0, index=sample_names_uniq, columns=data.var.index)
    df_a1 = pd.Series(a1, index=data.var.index)
    df_x = pd.Series(x, index=data.obs.index)

    # Write the DataFrame to a .txt file
    df_a0.to_csv("coeff_values/" + name + "_a0.txt", sep=",")
    df_a1.to_csv("coeff_values/" + name + "_a1.txt", sep=",")
    df_x.to_csv("coeff_values/" + name + "_x.txt", sep=",")


def shift_samples_per_mouse(x, a0, a1, sample_id, central, data, layer="f_cg"):
    """
    This function is used to shift the samples per mouse.
    This is done because diferent sample (mice) have different
    are x values, and different mice look completely shifted to one another.
    As this is not desirable, we alligned our sumple such as the are aligned
    on the central (defualt) side.
    BE CAREFUL: Make sure that the data layer IS NOT sparse.
    """
    xs = []
    # computing the individual shifts
    for s in np.unique(sample_id):
        idx = sample_id == s
        central_score = data[idx, central].layers[layer].sum(1)
        xx = x[idx]
        rc = central_score.argsort().argsort() / idx.sum()
        col = np.logical_and(rc > 0.9, rc < 0.99)
        offset = np.median(xx[col])
        xs.append(offset)

    x_shifted, a0_shifted, a1_shifted = shift_parameters(x, a0, a1, xs, sample_id)

    return x_shifted, a0_shifted, a1_shifted, xs


def shift_parameters(x, a0, a1, shift, sample_id):
    """
    This function transforms all the model's parameters
    accordingly to the shift that were found beforehand.
    In particular it shifts the x, and trasform the intercept a0,
    accordingly to the shift.
    Inputs:
    x: np.array, the x values
    a0: np.array, the intercepts
    a1: np.array, the slopes
    shift: np.array, the shifts
    sample_id: np.array, the sample identifiers
    """

    x_scaled = np.zeros(len(x), dtype=float)
    a0_scaled = np.zeros(a0.shape, dtype=float)
    a1_scaled = a1
    for s in np.unique(sample_id):
        idx = sample_id == s
        xx = x[idx]
        x_range = 1.0
        x_scaled[idx] = (xx - shift[s]) / x_range
        a0_scaled[s, :] = a0[s, :] + a1[:] * shift[s]
        a1_scaled[:] = a1[:] * x_range
    return x_scaled, a0_scaled, a1_scaled


####################################################################
# smaller functions, called many times


def scale_parameters(x, a0, a1):
    x_range = x.clone().max() - x.clone().min()
    x_min = x.clone().min()
    x_scaled = (x.clone() - x_min) / x_range
    a0_scaled = a0.clone() + a1.clone() * x_min
    a1_scaled = a1.clone() * x_range
    return x_scaled, a0_scaled, a1_scaled


def scale_parameters2(x, a0, a1, x_min, x_max):
    x_range = x_max - x_min
    x_scaled = (x - x_min) / x_range
    a0_scaled = a0 + a1 * x_min
    a1_scaled = a1 * x_range
    return x_scaled, a0_scaled, a1_scaled


def gene_index(data, gene):
    return np.where(gene == data.var.index)[0][0]


def import_data_txt(location):
    df = pd.read_csv(location, delimiter="\t")
    z = df.values
    return z


def make_design_matrix_numpy(sample_id):
    design_matrix = np.hstack(
        [
            np.array(sample_id == i).astype(int).reshape(len(sample_id), 1)
            for i in np.unique(sample_id)
        ]
    )
    return design_matrix


# functions use for plotting results

# def f_norm(x, a0_pyro, a1_pyro, DM):
#     y = x[:, None] * a1_pyro[None, :]
#     y += np.matmul(DM[:, :], a0_pyro)
#     return np.exp(y)


# def f_single(x, a0_pyro, a1_pyro, DM):
#     y = x[:, None] * np.matmul(DM[:, :], a1_pyro)
#     y += np.matmul(DM[:, :], a0_pyro)
#     return np.exp(y)


# def f_norm_a0_mean(x, a0_pyro, a1_pyro):
#     y = x[:, None] * a1_pyro[None, :] + a0_pyro.mean(axis=0)
#     return np.exp(y)


# def split_vector_into_subsets(vector, num_subsets):
#     avg = len(vector) // num_subsets
#     remainder = len(vector) % num_subsets
#     subsets = []
#     i = 0
#     for _ in range(num_subsets):
#         subset_size = avg + (1 if remainder > 0 else 0)
#         subset = vector[i : i + subset_size]
#         subsets.append(subset)
#         i += subset_size
#         remainder -= 1
#     return subsets
