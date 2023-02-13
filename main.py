## Jax

# Stop jax from taking up 90% of GPU vram
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jax

# Data
from data.dataload import dataload 

# TODO: Discuss this design choice in terms of the optimizer
# maybe make this a seperate module

# Model and optimizer
from models.model import get_model, get_optim, get_loss

# Visualization

from visualization.visualize import display_images

## Weights and biases
import wandb

#wandb.init(project="test-project", entity="ai-dtu")

import hydra
from omegaconf import DictConfig, OmegaConf

### TODO: REMOVE

from jax import numpy as jnp

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

@hydra.main(config_path="configs/", config_name="defaults", version_base='1.3')
def run_experiment(cfg):

#    print(cfg) 
#    wandb.config = OmegaConf.to_container(
#            cfg, resolve=True, throw_on_missing=True
#            )
    print(cfg)
    wandb.init(entity=cfg.wandb.setup.entity, project=cfg.wandb.setup.project)

    train_dataset, test_dataset = dataload(cfg) 

    model_parameters, model_call = get_model(cfg)

    optim_parameters, optim_alg = get_optim(cfg)

    loss_fn = get_loss(cfg)

    # TODO: GET TRAIN, TEST, AND VALIDATION SPLITS

    for epoch in range(cfg.train_and_test.train.epochs): 
        for t, (data, labels) in enumerate(train_dataset):

            # TODO: REMOVE
            t_data = one_hot(labels, 10).T 
            optim_parameters, model_parameters = optim_alg(optim_parameters, model_parameters, t_data, labels)
            
            if t % cfg.wandb.log.frequency == 0:
                if cfg.wandb.log.loss:
                    wandb.log({"loss": loss_fn(model_parameters, t_data, data.T)})
                if cfg.wandb.log.img:
                    display_images(cfg, model_call(t_data, model_parameters).T, labels)

        if wandb.log.epochfrequency % epoch == 0:
            None

        

if __name__ == "__main__":
    run_experiment()
