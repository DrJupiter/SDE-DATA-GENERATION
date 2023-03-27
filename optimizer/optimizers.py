import jax.numpy as jnp
import optax
from models.dummy import dummy_jax

# https://github.com/deepmind/optax


def get_optim(cfg,params=None):
    if cfg.model.name == "dummy_jax":
        return dummy_jax.get_optim_parameters(cfg), dummy_jax.optim_alg
    elif cfg.model.optimizer == "adam":
        return adam(cfg,params)
    raise ValueError(f"Model {cfg.model.name} not found")

def adam(cfg,params):
    optimizer = optax.adam(learning_rate=cfg.model.learningrate)
    assert params is not None, "pls give func get_optim the model parameters, for this optimiazation function"
    opt_param = optimizer.init(params)
    return optimizer, opt_param
