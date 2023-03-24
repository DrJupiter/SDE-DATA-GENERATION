## Adding path (for some it is needed to import packages)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# convret to torch for FID
import torch

## Jax

# Stop jax from taking up 90% of GPU vram
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jax
from jax import grad

# Data
from data.dataload import dataload 

# TODO: Discuss this design choice in terms of the optimizer
# maybe make this a seperate module

# Model and optimizer
from models.model import get_model, get_optim, get_loss

# Visualization
from visualization.visualize import display_images

# Validation (FID)
from validation.FID import FID_score

## Weights and biases
import wandb

#wandb.init(project="test-project", entity="ai-dtu")

import hydra

## Optimizer
import optax

### TODO: REMOVE

import jax.numpy as jnp

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

@hydra.main(config_path="configs/", config_name="defaults", version_base='1.3')
def run_experiment(cfg):

    print(cfg)
    wandb.init(entity=cfg.wandb.setup.entity, project=cfg.wandb.setup.project)

    train_dataset, test_dataset = dataload(cfg) 

    model_parameters, model_call = get_model(cfg) # model_call(x_in, timesteps, parameters)

    optimizer, optim_parameters = get_optim(cfg, model_parameters)

    loss_fn = get_loss(cfg) # loss_fn(parameters, model_pass, data_batch, target_batch, time_steps)
    grad_fn = grad(loss_fn,0)

    for epoch in range(cfg.train_and_test.train.epochs): 
        for t, (data, labels) in enumerate(train_dataset): # batches
			
            # TODO: REMOVE
            t_data = one_hot(labels, 10).T 

            timesteps = jnp.ones(data.shape[0])*t

            # get grad for this batch
            # loss_value, grads = jax.value_and_grad(loss_fn)(model_parameters, model_call, data, labels, t)
            grads = grad_fn(model_parameters, model_call, data, labels, timesteps)

            # optim_parameters, model_parameters = optim_alg(optim_parameters, model_parameters, t_data, labels)
            updates, optim_parameters = optimizer.update(grads, optim_parameters, model_parameters)
            model_parameters = optax.apply_updates(model_parameters, updates)

            if t % cfg.wandb.log.frequency == 0:
                  if cfg.wandb.log.loss:
                    wandb.log({"loss": loss_fn(model_parameters, model_call, data, labels, timesteps)})
                    # wandb.log({"loss": loss_value})
                  if cfg.wandb.log.img:
                     display_images(cfg, model_call(data, timesteps, model_parameters), labels)

        if epoch % cfg.wandb.log.epoch_frequency == 0:
            if cfg.wandb.log.FID: 
                # generate pictures before this can be run

                # extract imgs from dataset
                x_test = [torch.tensor(data.reshape(cfg.train_and_test.test.batch_size,3,32,32)) for (data,labels) in test_dataset][:1]
                
                x_test = torch.vstack(x_test)

                # get saved imgs
                path_to_imgs = f"{cfg.train_and_test.test.img_save_loc}*jpg" # or whatever extension they will end up with

                # calculate and log fid score
                wandb.log({"fid_score": FID_score(x1=x_test,x2=x_test)})

if __name__ == "__main__":
    run_experiment()
