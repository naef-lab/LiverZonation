import numpy as np

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def easy_model(mp, y=None):
    """
    This model has a lower number of cells, coming all from the same sample,
    therefore there is no need to use the design matrix
    """
    Nc = y.shape[0]
    Ng = y.shape[1]

    cell_plate = numpyro.plate("cells", size=Nc, dim=-2)
    gene_plate = numpyro.plate("genes", size=Ng, dim=-1)

    disp = numpyro.sample("disp", dist.Uniform(0.0, 1.0))

    with gene_plate:
        a1 = numpyro.sample("a1", dist.Normal(mp.a1_loc, 3.0))
        a0 = numpyro.sample("a0", dist.Normal(mp.a0_loc, 6.0))

    with cell_plate:
        xx = numpyro.sample("x", dist.Normal(mp.x, 3.0))

    # log_mu = xx * a1 + mp.log_n_UMI[:, None] + a0
    log_mu = xx * a1 + mp.log_n_UMI + a0

    cutoff = 50.0
    log_mu = cutoff * jnp.tanh(log_mu / cutoff)
    mu = jnp.exp(log_mu)

    conc = 1 / disp
    rate = 1 / (disp * mu)

    with cell_plate:
        with gene_plate:
            numpyro.sample(
                "obs_", dist.GammaPoisson(concentration=conc, rate=rate), obs=y
            )


def model(mp, y=None):
    Nc = y.shape[0]
    Ng = y.shape[1]
    Ns = mp.dm.shape[1]

    sample_plate = numpyro.plate("samples", size=Ns, dim=-2)
    cell_plate = numpyro.plate("cells", size=Nc, dim=-2)
    gene_plate = numpyro.plate("genes", size=Ng, dim=-1)

    disp = numpyro.sample("disp", dist.Uniform(0.0, 1.0))

    with gene_plate:
        a1 = numpyro.sample("a1", dist.Normal(mp.a1_loc, 3.0))

    with gene_plate, sample_plate:
        a0 = numpyro.sample("a0", dist.Normal(mp.a0_loc, 6.0))

    with cell_plate:
        xx = numpyro.sample("x", dist.Normal(mp.x, 3.0))

    log_mu = xx * a1 + mp.log_n_UMI
    log_mu += jnp.matmul(mp.dm, a0)

    cutoff = 50.0
    log_mu = cutoff * jnp.tanh(log_mu / cutoff)

    mu = jnp.exp(log_mu)
    # Using the mean and dispersion parametrization
    conc = numpyro.deterministic("conc", 1 / disp)
    rate = numpyro.deterministic("rate", 1 / (disp * mu))

    with cell_plate, gene_plate:
        numpyro.sample("obs_", dist.GammaPoisson(concentration=conc, rate=rate), obs=y)


def guide(mp, y=None):
    Nc = y.shape[0]
    Ng = y.shape[1]
    Ns = mp.dm.shape[1]

    sample_plate = numpyro.plate("samples", size=Ns, dim=-2)
    cell_plate = numpyro.plate("cells", size=Nc, dim=-2)
    gene_plate = numpyro.plate("genes", size=Ng, dim=-1)

    # loc
    a1_loc = numpyro.param("a1_loc", mp.a1_loc)
    a0_loc = numpyro.param("a0_loc", mp.a0_loc)
    x_loc = numpyro.param("x_loc", mp.x)

    # scale
    a1_scale = numpyro.param("a1_scale", jnp.ones((1, Ng)))
    a0_scale = numpyro.param("a0_scale", jnp.ones((Ns, Ng)))
    x_scale = numpyro.param("x_scale", jnp.ones((Nc, 1)))
    disp_loc = numpyro.param("disp_loc", mp.disp)

    # for disp a delta function is used
    disp = numpyro.sample("disp", dist.Delta(disp_loc))

    with gene_plate:
        a1 = numpyro.sample("a1", dist.Normal(a1_loc, a1_scale))

    with gene_plate, sample_plate:
        a0 = numpyro.sample("a0", dist.Normal(a0_loc, a0_scale))

    with cell_plate:
        xx = numpyro.sample("x", dist.Normal(x_loc, x_scale))
