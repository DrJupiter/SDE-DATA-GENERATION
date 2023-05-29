#from subvp import SUBVPSDE 
from sde.subvp import SUBVPSDE 


def get_sde(cfg):
    if cfg.sde.name == "paper_subvp":
        return SUBVPSDE(cfg)
    raise NameError(f"No SDE found with the name {cfg.sde.name}")



