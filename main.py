## Jax

# Stop jax from taking up 90% of GPU vram
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jax

# Data
from data.dataload import dataload 


## Weights and biases
import wandb

#wandb.init(project="test-project", entity="ai-dtu")

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs/", config_name="defaults", version_base='1.3')
def run_experiment(cfg):

    print(cfg) 
    wandb.config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
            )

    run = wandb.init(entity=cfg.wandb.setup.entity, project=cfg.wandb.setup.project)

    dataset = dataload(cfg) 
    if cfg.wandb.log.loss:
        wandb.log({"loss": 0})
        

if __name__ == "__main__":
    run_experiment()
