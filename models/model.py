from .dummy import dummy_jax


def get_model(cfg):
    if cfg.model.name == "dummy_jax":
        return dummy_jax.get_parameters(cfg), dummy_jax.model_call
    raise ValueError(f"Model {cfg.model.name} not found")

def get_optim(cfg):
    if cfg.model.name == "dummy_jax":
        return dummy_jax.get_optim_parameters(cfg), dummy_jax.optim_alg
    raise ValueError(f"Model {cfg.model.name} not found")

def get_loss(cfg):
    if cfg.model.name == "dummy_jax":
        return dummy_jax.loss_fn
    raise ValueError(f"Model {cfg.model.name} not found")
    

