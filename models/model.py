## Needed for Andreas to import in a subfolder from a subfolder
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
##

# SHARDING
import jax

from models import get_ddpm_unet_new, dummy_jax, get_ddpm_unet_classifier


def get_model(cfg, key):
    if cfg.model.name == "dummy_jax":
        return dummy_jax.get_parameters(cfg), dummy_jax.get_dummy_train(cfg), dummy_jax.get_dummy_train(cfg)
    elif cfg.model.name == "ddpm_unet":

        if cfg.model.type == "score":

            params, ddpm_unet, inf_ddpm_unet = get_ddpm_unet_new(cfg, key)
            return params, ddpm_unet, inf_ddpm_unet
        elif cfg.model.type == "classifier":

            params, ddpm_unet, inf_ddpm_unet = get_ddpm_unet_classifier(cfg, key)
            return params, ddpm_unet, inf_ddpm_unet

    elif cfg.model.name == "sde": 
        from sde.sde import get_sde
        #from sde import get_sde
        sde = get_sde(cfg)
        def model_call(xt, t, x0, key):

            if isinstance(x0, list):
                return xt
            else:
                t = t if len(t.shape) != 1 else t.reshape(len(t),-1)
                xt = xt if len(xt.shape) != 1 else xt.reshape(1,-1)
                x0 = x0 if len(x0.shape) != 1 else x0.reshape(1,-1)
                out = sde.score(x0, t, xt)
                if out.shape[0] == 1:
                    return out.reshape(-1)
                return out
        return [jax.numpy.ones(1)], model_call, model_call
    
    raise ValueError(f"Model {cfg.model.name} not found")