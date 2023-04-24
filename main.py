## Adding path (for some it is needed to import packages)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Saving and loading
import pickle

# convret to torch for FID
import torch

## Jax

# Stop jax from taking up 90% of GPU vram
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.5'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'

import jax
import jax.numpy as jnp

# Data
from data.dataload import dataload 

# TODO: Discuss this design choice in terms of the optimizer
# maybe make this a seperate module

# Model and optimizer
from models.model import get_model 
from optimizer.optimizers import get_optim

# Loss
from loss.loss import get_loss
    
# Visualization
from visualization.visualize import display_images

# Validation (FID)
from validation.FID import FID_score

## Weights and biases
import wandb

#wandb.init(project="test-project", entity="ai-dtu")

# config mangement
import hydra

## Optimizer
import optax

## SDE
from sde.sde import get_sde
from sde.sample import sample

# sharding
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

# TEST SHARDING TODO: REMOVE
# import os
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

### Train loop:

# Gets config
@hydra.main(config_path="configs/", config_name="defaults", version_base='1.3')
def run_experiment(cfg):

    # initialize Weights and Biases
    print(cfg)
    wandb.init(entity=cfg.wandb.setup.entity, project=cfg.wandb.setup.project)

    # Get randomness key
    key = jax.random.PRNGKey(cfg.model.key)
    key, subkey = jax.random.split(key)

    # Load train and test sets
    train_dataset, test_dataset = dataload(cfg) 

    # Get model forward call and its parameters
    model_parameters, model_call = get_model(cfg, key = subkey) # model_call(x_in, timesteps, parameters)
   
    # Get optimizer and its parameters
    optimizer, optim_parameters = get_optim(cfg, model_parameters)

    # get sde
    SDE = get_sde(cfg)

    # get loss functions and convert to grad function
    loss_fn = get_loss(cfg) # loss_fn(func, function_parameters, data, perturbed_data, time, key)

    grad_fn = jax.grad(loss_fn,1) # TODO: try to JIT function partial(jax.jit,static_argnums=0)(jax.grad(loss_fn,1))


    # get shard
    sharding = PositionalSharding(mesh_utils.create_device_mesh((len(jax.devices()),1)))

    # start training for each epoch
    for epoch in range(cfg.train_and_test.train.epochs): 
        for i, (data, labels) in enumerate(train_dataset): # batch training

            # split key to keep randomness "random" for each training batch
            key, *subkey = jax.random.split(key, 4)

            # get timesteps given random key for this batch and data shape
            # TODO: Strictly this changes from sde to sde
            timesteps = jax.random.uniform(subkey[0], (data.shape[0],), minval=1e-5, maxval=1)

            # Perturb the data with the timesteps trhough sampling sde trick (for speed, see paper for explanation)
            perturbed_data = jax.device_put(SDE.sample(timesteps, data, subkey[1]),sharding.reshape((1,len(jax.devices()))))



            # scale timesteps for more significance
            scaled_timesteps = timesteps*999

            # get grad for this batch
              # loss_value, grads = jax.value_and_grad(loss_fn)(model_parameters, model_call, data, labels, t) # is this extra computation time

                
            grads = grad_fn(model_call, model_parameters, data, perturbed_data, scaled_timesteps, subkey[2])

            # get change in model_params and new optimizer params
            # optim_parameters, model_parameters = optim_alg(optim_parameters, model_parameters, t_data, labels)
            updates, optim_parameters = optimizer.update(grads, optim_parameters, model_parameters)

            # update model params
            model_parameters = optax.apply_updates(model_parameters, updates)

            # Logging loss and an image
            if i % cfg.wandb.log.frequency == 0:
                  if cfg.wandb.log.loss:
                    wandb.log({"loss": loss_fn(model_call, model_parameters, data, perturbed_data, scaled_timesteps, subkey[2])})
                    # wandb.log({"loss": loss_value})
                  if cfg.wandb.log.img:
                    # reverse sde sampling
                    drift = lambda t,y, args: SDE.reverse_drift(y, jnp.array([t]), args)
                    diffusion = lambda t,y, args: SDE.reverse_diffusion(y, jnp.array([t]), args)
                    get_sample = lambda t, key1, key0, xt: sample(0, 0, t.astype(float)[0], -1/1000, drift, diffusion, [model_call, model_parameters if cfg.model.name != "sde" else data[0], key0], xt, key1) 
                                     # dt0 = - 1/N

                    n = len(perturbed_data) 
                    if cfg.wandb.log.n_images < n:
                        n = cfg.wandb.log.n_images 

                    key, *subkey = jax.random.split(key, len(perturbed_data)*2 + 1)

                    args = (timesteps.reshape(-1,1)[:n], jnp.array(subkey[:len(subkey)//2])[:n], jnp.array(subkey[len(subkey)//2:])[:n], perturbed_data[:n])
                    images = jax.vmap(get_sample, (0, 0, 0, 0))(*args)
                    
                    Z = (jax.random.normal(subkey[2], data.shape)*255)[:n]
                    args = (jnp.ones_like(timesteps.reshape(-1,1))[:n], jnp.array(subkey[:len(subkey)//2])[:n], jnp.array(subkey[len(subkey)//2:])[:n], Z)
                    normal_distribution = jax.vmap(get_sample, (0, 0, 0, 0))(*args)

                    # Rescale images for plotting
                    mins, maxs=jnp.min(perturbed_data, axis=1).reshape(-1, 1)[:n], jnp.max(perturbed_data, axis=1)[:n].reshape(-1,1)
                    rescaled_images = (perturbed_data[:n]-mins)/(maxs-mins)*255

                    display_images(cfg, images, labels[:n], log_title="Reverse Sample x(t) -> x(0)")
                    display_images(cfg, perturbed_data[:n], labels[:n], log_title="Perturbed images")
                    display_images(cfg, rescaled_images, labels[:n], log_title="Min-Max Rescaled")
                    display_images(cfg, normal_distribution, labels[:n], log_title="N(0,I) -> x(0)")
                    display_images(cfg, Z, labels[:n], log_title="N(0,I)")
                    display_images(cfg, data[:n], labels[:n], log_title="Original Images: x(0)")
                  if cfg.wandb.log.parameters:
                          with open(os.path.join(wandb.run.dir, "paremeters.pickle"), 'wb') as f:
                            pickle.dump((epoch*len(train_dataset) + i, model_parameters, optim_parameters), f, pickle.HIGHEST_PROTOCOL)
                          wandb.save("paramters.pickle")
                    #image = get_sample(timesteps[0], subkey[2], subkey[2], perturbed_data[0])

                    #rescaled_perturbed = (perturbed_data[0]-jnp.min(perturbed_data[0]))/(jnp.max(perturbed_data[0])-jnp.min(perturbed_data[0]))*255

                    #random_noise = jax.random.normal(subkey[2], data[0].shape)*255
                    #image_from_random = get_sample(1, subkey[2], subkey[2], random_noise)

                    #random_noise_uniform = jax.random.uniform(subkey[2], data[0].shape)*255
                    #image_from_random_uniform = get_sample(1, subkey[2], subkey[2], random_noise_uniform)
                    #display_images(cfg, [image, perturbed_data[0], data[0], rescaled_perturbed, image_from_random, random_noise,image_from_random_uniform, random_noise_uniform], ["sample", "perturbed", "original", "rescaled", "image from random", "random image", "uniform sample image", "uniform image"])

                    #display_images(cfg, model_call(perturbed_data, scaled_timesteps, model_parameters, subkey[2]), labels)
        # Test loop
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
