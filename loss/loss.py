
from loss.implicit import implicit_score_matching
from loss.dsm import get_denosing_score_matching
from jax.numpy import zeros

def get_loss(cfg):
    if cfg.loss.name == "denoising_score_matching":
        return get_denosing_score_matching(cfg)
    elif cfg.loss.name == "implicit_score_matching":
        return implicit_score_matching
    elif cfg.loss.name == "none":
        return lambda *args: zeros(1) 
    raise NotImplementedError(f"The loss {cfg.loss.name} is not implemented")
