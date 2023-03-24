
from loss.implicit import implicit_score_matching

def get_loss(cfg):
    if cfg.loss.name == "denoising_score_matching":
        raise NotImplementedError()
    elif cfg.loss.name == "implicit_score_matching":
        return implicit_score_matching
    raise NotImplementedError(f"The loss {cfg.loss.name} is not implemented")
