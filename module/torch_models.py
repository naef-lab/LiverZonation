import numpy as np
import torch
from torch import tensor as tt
from collections import namedtuple
from tqdm import tqdm


def training(data, x_unif, coef_pau, n_c, dm, clamp, n_iter, batch_size, dev):
    """
    Trains a model using PyTorch on the provided dataset.

    This function initializes and optimizes model parameters to fit the given data. It is the sole function where PyTorch is utilized for computational operations. The training involves gradient descent optimization using the Adam optimizer. The function also handles device setting for PyTorch operations.

    Parameters:
    data (array-like): The dataset used for training.
    x_unif (array-like): Vector to initialize values of latent variable x.
    coef_pau (array-like): Coefficients a1_g and a0_g of the model.
    n_c (array-like): The number of counts per cell.
    dm (array-like): The design matrix created from the sample identifiers.
    clamp (int): An index specifying which dimension to clamp during training. It gets rid of one of the likelihood symmetries.
    n_iter (int): Number of iterations for the training loop.
    batch_size (int): Size of the batch for training. If set to 0, it defaults to the size of the dataset.
    dev (str or torch.device): The device (e.g., 'cpu' or 'cuda') on which to perform the training.

    Returns:
    tuple: A tuple containing the final values of the model's parameters and the loss values:
        - x_final (numpy.ndarray): Final optimized values of x.
        - a0_final (numpy.ndarray): Final optimized values of a0, one of the PAU coefficients.
        - a1_final (numpy.ndarray): Final optimized values of a1, another PAU coefficient.
        - disp_final (numpy.ndarray): Final dispersion values after optimization.
        - losses (list): List of loss values recorded at each training iteration.

    """
    # set the device to use
    torch.set_default_device(dev)
    dm = tt(dm, dtype=torch.float32)
    DATA = torch.tensor(data)
    NC, NG = DATA.shape
    NS = dm.shape[1]
    if batch_size == 0:
        batch_size = NC

    # preparing the starting values for the optimization
    a0_pau = coef_pau[:, 0]
    a1_pau = coef_pau[:, 1]
    scale = a1_pau[clamp]
    a1_scaled = a1_pau / scale
    x_scaled = x_unif * scale

    mask = torch.eye(NG, dtype=float)
    mask[clamp, clamp] = 0
    fix = tt(1.0).detach()
    log_n_UMI = torch.log(tt(n_c))
    mp = dict(
        log_n_UMI=log_n_UMI,
        clamp=clamp,
        dm=dm,
        fix=fix,
        mask=mask,
        cutoff=50,
    )
    MyTuple = namedtuple("param", mp)
    MP = MyTuple(**mp)

    # initalizing the parameters (leafs)
    disp = tt(np.log(0.3), requires_grad=True)
    x = tt(x_scaled, requires_grad=True, dtype=float)
    a1 = tt(a1_scaled, requires_grad=True, dtype=float)
    a0 = tt(a0_pau, dtype=torch.float32)
    a0 = a0.repeat(NS, 1)
    a0.requires_grad = True

    # training the model
    losses = []
    optimizer = torch.optim.Adam([x, a0, a1, disp], lr=0.001)
    batch_size = NC
    # Optimize the latent variables to minimize the loss
    # for step in range(n_iter):
    for step in tqdm(range(n_iter), desc="Training progress"):
        optimizer.zero_grad()  # zero the gradients
        output = loss_clamp_batch(x, a0, a1, disp, batch_size, MP, DATA)
        output.backward()  # compute the gradients
        optimizer.step()  # update the variable
        losses.append(output.detach())

    x_final = x.clone().cpu().detach().numpy()
    disp_final = torch.exp(disp.clone()).cpu().detach().numpy()
    a0_final = a0.clone().cpu().detach().numpy()
    a1_final = a1.clone().cpu().detach().numpy()

    return x_final, a0_final, a1_final, disp_final, losses


def training_gene_selection(
    DATA, x_unif, coef_pau, n_c, dm, clamp, n_iter, batch_size, dev
):
    """
    This function is used to train the model on the data.

    """
    NC, NG = DATA.shape
    NS = dm.shape[1]
    if batch_size == 0:
        batch_size = NC

    # preparing the starting values for the optimization
    a0_pau = coef_pau[:, 0]
    a1_pau = coef_pau[:, 1]
    scale = a1_pau[clamp]
    a1_scaled = a1_pau / scale
    x_scaled = x_unif * scale

    mask = torch.eye(NG, device=dev, dtype=float)
    mask[clamp, clamp] = 0
    fix = tt(1.0, device=dev).detach()
    log_n_UMI = torch.log(tt(n_c, device=dev))
    mp = dict(
        log_n_UMI=log_n_UMI,
        clamp=clamp,
        dm=dm,
        fix=fix,
        mask=mask,
        cutoff=50,
    )
    MyTuple = namedtuple("param", mp)
    MP = MyTuple(**mp)

    # initalizing the parameters (leafs)
    disp = tt(np.log(0.3), requires_grad=True, device=dev)
    x = tt(x_scaled, requires_grad=True, dtype=float, device=dev)
    a1 = tt(a1_scaled, requires_grad=True, dtype=float, device=dev)
    a0 = tt(a0_pau, dtype=torch.float32, device=dev)
    a0 = a0.repeat(NS, 1)
    a0.requires_grad = True

    # training the model
    losses = []
    optimizer = torch.optim.Adam([x, a0, a1, disp], lr=0.001)
    batch_size = NC
    # Optimize the latent variables to minimize the loss
    for step in range(n_iter):
        optimizer.zero_grad()  # zero the gradients
        output = loss_clamp_batch(x, a0, a1, disp, batch_size, MP, DATA)
        output.backward()  # compute the gradients
        optimizer.step()  # update the variable
        losses.append(output.detach())

    x_final = x.clone().cpu().detach().numpy()
    disp_final = torch.exp(disp.clone()).cpu().detach().numpy()
    a0_final = a0.clone().cpu().detach().numpy()
    a1_final = a1.clone().cpu().detach().numpy()

    return x_final, a0_final, a1_final, disp_final, losses


def loss_clamp_batch(x, a0, a1, disp, batch_size, mp, DATA):
    """
    function that takes as input the leaf parameters and returns the loss.
    Beware disp will need to be exp() in the main function
    a0 is sample specific, disp and a1 only gene specific.
    If you want to use all datapoints, set batch_size = DATA.shape[0]
    The 'clamp' gene slope coefficient a1 is set to the fix value (1).

    """
    NC = DATA.shape[0]
    # killing the gradient
    a1_ = torch.matmul(mp.mask, a1)
    a1_[mp.clamp] = mp.fix

    idx = torch.randperm(DATA.shape[0])[:batch_size]
    y = x[idx, None] * a1_[None, :] + mp.log_n_UMI[idx, None]
    y += torch.matmul(mp.dm[idx, :], a0)
    alpha = torch.exp(disp)

    y = mp.cutoff * torch.tanh(y / mp.cutoff)
    lmbda = torch.exp(y)

    r = 1 / alpha
    p = alpha * lmbda / (1 + alpha * lmbda)
    NB = torch.distributions.NegativeBinomial(
        total_count=r, probs=p, validate_args=None
    )
    return -NB.log_prob(DATA[idx, :]).sum() * (
        NC / batch_size
    )  # correct the likelihood rescaling


def loss_clamp_batch_2(x, a0, a1, disp, batch_size, mp, DATA):
    """
    function that takes as input the leaf parameters and returns the loss.
    Beware disp will need to be exp() in the main function
    a0 is sample specific, disp and a1 only gene specific.
    If you want to use all datapoints, set batch_size = DATA.shape[0]
    The 'clamp' gene slope coefficient a1 is set to the fix value (1).
    This function clamps also the intercept a0 to the initial values.
    """
    NC = DATA.shape[0]
    # killing the gradient of a1
    a1_ = torch.matmul(mp.mask, a1)
    a1_[mp.clamp] = mp.fix

    # killing the gradient of columns number 1 of a0
    a0_ = torch.matmul(a0, mp.maskk)
    a0_[:, mp.clamp] = mp.fixx

    idx = torch.randperm(DATA.shape[0])[:batch_size]
    y = x[idx, None] * a1_[None, :] + mp.log_n_UMI[idx, None]
    y += torch.matmul(mp.dm[idx, :], a0_)
    alpha = torch.exp(disp)

    y = mp.cutoff * torch.tanh(y / mp.cutoff)
    lmbda = torch.exp(y)

    r = 1 / alpha
    p = alpha * lmbda / (1 + alpha * lmbda)
    NB = torch.distributions.NegativeBinomial(
        total_count=r, probs=p, validate_args=None
    )
    return -NB.log_prob(DATA[idx, :]).sum() * (
        NC / batch_size
    )  # correct the likelihood rescaling


def loss_simple(x, a0, a1, disp, mp, DATA):
    NC = DATA.shape[0]
    # killing the gradient
    a1_ = torch.matmul(mp.mask, a1)
    a1_[mp.clamp] = mp.fix

    # idx = torch.randperm(DATA.shape[0])[:batch_size]
    y = x[:, None] * a1_[None, :] + a0[None, :] + mp.log_n_UMI[:, None]
    alpha = torch.exp(disp)

    y = mp.cutoff * torch.tanh(y / mp.cutoff)
    lmbda = torch.exp(y)

    r = 1 / alpha
    p = alpha * lmbda / (1 + alpha * lmbda)
    NB = torch.distributions.NegativeBinomial(
        total_count=r, probs=p, validate_args=None
    )
    return -NB.log_prob(DATA[:, :]).sum()


# def loss_simple_batch(x, a0, a1, disp, batch_size, mp, DATA):

#     NC = DATA.shape[0]
#     # killing the gradient
#     a1_ = torch.matmul(mp.mask, a1)
#     a1_[mp.clamp] = mp.fix

#     idx = torch.randperm(DATA.shape[0])[:batch_size]
#     y = x[idx, None] * a1_[None, :] + a0[None, :] + mp.log_n_UMI[idx, None]
#     alpha = torch.exp(disp)

#     y = mp.cutoff * torch.tanh(y / mp.cutoff)
#     lmbda = torch.exp(y)

#     r = 1 / alpha
#     p = alpha * lmbda / (1 + alpha * lmbda)
#     NB = torch.distributions.NegativeBinomial(
#         total_count=r, probs=p, validate_args=None
#     )
#     return -NB.log_prob(DATA[idx, :]).sum() * (NC / batch_size)


def loss_gene_selection(x, a0, a1, disp, batch_size, mp, DATA):
    """
    This loss function does not optimize x, but only a0 and a1.
    It is used copute the likelihood of the data, and fits the
    parameters than will later be used for the optimization of x.
    """
    NC = DATA.shape[0]

    idx = torch.randperm(DATA.shape[0])[:batch_size]
    y = (
        mp.log_n_UMI[idx, None]
        + torch.matmul(mp.dm[idx, :], a0)
        + torch.matmul(mp.dm[idx, :], a1) * x[idx, None]
    )
    alpha = disp.exp()

    y = mp.cutoff * torch.tanh(y / mp.cutoff)
    lmbda = torch.exp(y)

    r = 1 / alpha
    p = alpha * lmbda / (1 + alpha * lmbda)
    NB = torch.distributions.NegativeBinomial(
        total_count=r, probs=p, validate_args=None
    )
    return -NB.log_prob(DATA[idx, :]).sum() * (
        NC / batch_size
    )  # correct the likelihood rescaling


def make_design_matrix(cell_identifiers):
    """A function to create a generic design matrix from cell identifiers

    For example:
    input, cell_identifiers = [1,1,1,2,2,2]
    output, design_matrix: [[1,1,1,0,0,0][0,0,0,1,1,1]]

    Arguments
    ---------
    cell_identifiers: torch tensor
        List of cells identifiers, should be of type torch.int

    Returns
    -------
    design_matrix: torch tensor
        Design matrix of shape (num_cells, num_unique_identifiers)

    """
    design_matrix = torch.hstack(
        [
            (cell_identifiers == v).type(torch.float).reshape(len(cell_identifiers), 1)
            for i, v in enumerate(torch.unique(cell_identifiers))
        ]
    )
    return design_matrix
