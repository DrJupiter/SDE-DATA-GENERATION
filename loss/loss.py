
from loss.implicit import get_implicit_score_matching
from loss.dsm import get_denosing_score_matching
from jax.numpy import array

def get_loss(cfg):
    if cfg.loss.name == "denoising_score_matching":
        return get_denosing_score_matching(cfg)
    elif cfg.loss.name == "implicit_score_matching":
        return get_implicit_score_matching(cfg)
    elif cfg.loss.name == "none":
        return lambda *args: array(0.)
    raise NotImplementedError(f"The loss {cfg.loss.name} is not implemented")
