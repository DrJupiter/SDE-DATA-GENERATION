import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sde.subvp import SUBVPSDE 
def get_sde(cfg):
    if cfg.sde.name == "paper_subvp":
        return SUBVPSDE
    raise NameError(f"No SDE found with the name {cfg.sde.name}")



