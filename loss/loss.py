
def get_loss(cfg):
    if cfg.loss.name == "denoising_score_matching":
        raise NotImplementedError()
    elif cfg.loss.name == "implicit_score_matching":
        raise NotImplementedError()
    raise NotImplementedError(f"The loss {cfg.loss.name} is not implemented")
