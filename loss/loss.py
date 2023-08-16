
from loss.implicit import get_implicit_score_matching
from loss.dsm import get_denosing_score_matching
from loss.combined_sm import get_combined
from loss.yangsong import get_yang_song
from jax.numpy import array
from loss.classifier_losses.cross_entropy import get_cross_entropy
from loss.mean import get_mean_matching

def get_loss(cfg):
    if cfg.loss.name == "denoising_score_matching":
        return get_denosing_score_matching(cfg)
    elif cfg.loss.name == "implicit_score_matching":
        return get_implicit_score_matching(cfg)
    elif cfg.loss.name == "none":
        return lambda *args: array(0.)
    elif cfg.loss.name == "combined_score_matching":
        return get_combined(cfg)
    elif cfg.loss.name == "yangsong":
        return get_yang_song(cfg) 
    elif cfg.loss.name == "cross_entropy":
        return get_cross_entropy(cfg)
    elif cfg.loss.name == "mean_matching":
        return get_mean_matching(cfg)
    raise NotImplementedError(f"The loss {cfg.loss.name} is not implemented")
