
from utils.utility import batch_matmul
from jax import numpy as jnp
from sde.sde import get_sde
from sde.subvp import SUBVPSDE
import jax

def get_yang_song(cfg):

    sde = get_sde(cfg)

    if isinstance(sde, SUBVPSDE):

        def yang_song(func, function_parameters, data, perturbed_data, time, z, text_embedding, key):
            # Todo lambda(t)
            score_model = func(perturbed_data, time, text_embedding, function_parameters, key)
            #_mu, cov = sde.parameters(time, jnp.zeros_like(data))
            #score = jax.vmap(lambda a, b: a * b)(-score_model, 1. / cov)
            score = score_model

            #mean = sde.mean(time, data)
            std = sde.diffusion.decomposition(time)

            if cfg.loss.likelihood_weighting and sde.diffusion.diagonal.form:
                L = sde.diffusion(time)
                lambda_scalar = L @ L.transpose()
                loss = jnp.square(score + z * sde.diffusion.inv_decomposition(time))
                loss = 0.5 * jnp.sum(loss.reshape((loss.shape[0], -1)), axis=-1) * lambda_scalar
            # This one is the default
            else:
                loss = jnp.square(score*std + z)
                loss = 0.5 * jnp.sum(loss.reshape((loss.shape[0], -1)), axis=-1)

            loss = jnp.mean(loss)

            return loss
    else:
        raise NotImplementedError("Not implemented for SDE other than SUBVPSDE")
    return yang_song