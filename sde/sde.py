#from subvp import SUBVPSDE 
from sde.subvp import SUBVPSDE 
from sde.denoise import DENOISESDE
from sde.sde_redefined import SDE
from sympy import Matrix, Symbol
import sympy

def get_sde(cfg):
    if cfg.sde.name == "paper_subvp":
        t = Symbol('t', real=True, nonnegative=True) 
        n = 1 # n = 1, because it is a Scalar sde
        F = Matrix.diag([[-0.5 * (cfg.sde.beta_min + t *(cfg.sde.beta_max-cfg.sde.beta_min))]*n])
        L = Matrix.diag([[sympy.sqrt(cfg.sde.beta_min+ t * (cfg.sde.beta_max-cfg.sde.beta_min))]*n])
        Q = Matrix.eye(n)
        return SDE(t, F.diagonal(), L.diagonal(), Q.diagonal(), drift_diagonal_form=True, diffusion_diagonal_form=True, diffusion_matrix_diagonal_form=True)
        #return SUBVPSDE(cfg)
    if cfg.sde.name == "denoise":
        return DENOISESDE(cfg)
    raise NameError(f"No SDE found with the name {cfg.sde.name}")



