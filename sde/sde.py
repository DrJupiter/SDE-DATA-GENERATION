#from subvp import SUBVPSDE 
from sde.subvp import SUBVPSDE 
from sde.denoise import DENOISESDE


def get_sde(cfg):
    if cfg.sde.name == "paper_subvp":
        return SUBVPSDE(cfg)
    if cfg.sde.name == "denoise":
        return DENOISESDE(cfg)
    raise NameError(f"No SDE found with the name {cfg.sde.name}")



