## Adding path (for some it is needed to import packages)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Saving and loading
import pickle


## Jax

# Stop jax from taking up 90% of GPU vram
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.5'
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
#os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import jax
import jax.numpy as jnp
#jax.config.update('jax_platform_name', 'cpu')

# Data
from data.dataload import dataload, get_data_mean, get_all_test_data

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
from validation.fid_c import get_fid_model

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

# time the process so we cant stop before termination with the goal of allowing WANDB to save our weights
from time import time


# Paramter loading
from utils.utils import load_model_paramters, load_optimizer_paramters, get_wandb_input, min_max_rescale, get_save_path_names, get_classifier

### Train loop:

# Gets config
@hydra.main(config_path="configs/", config_name="defaults", version_base='1.3')
def run_experiment(cfg):
    START_TIME = time()
    TIME_EXCEEDED = False
    # initialize Weights and Biases
    print(cfg)
    print(jax.devices())
    wandb.init(**get_wandb_input(cfg))

    # Get randomness key
    key = jax.random.PRNGKey(cfg.model.key)
    key, subkey = jax.random.split(key)

    # Load train and test sets
    train_dataset, test_dataset = dataload(cfg) 

    TRAIN_MEAN = get_data_mean(cfg, train_dataset)
    # Get model forward call and its parameters
    model_parameters, model_call, inference_model = get_model(cfg, key = subkey) 
    model_parameters = load_model_paramters(cfg, model_parameters)

    # get sde
    SDE = get_sde(cfg)

    # get loss functions and convert to grad function
    loss_fn = get_loss(cfg) # loss_fn(func, function_parameters, data, perturbed_data, time, key)

    if cfg.train_and_test.mode == "train":
    # Get optimizer and its parameters
      optimizer, optim_parameters = get_optim(cfg, model_parameters)
      optim_parameters = load_optimizer_paramters(cfg, optim_parameters)
      grad_fn = jax.grad(loss_fn,1) # TODO: try to JIT function partial(jax.jit,static_argnums=0)(jax.grad(loss_fn,1))
      grad_fn = jax.jit(grad_fn, static_argnums=0)
    elif cfg.train_and_test.mode == "validation":
        if cfg.model.type == "score":
          fid_model = get_fid_model(cfg) 




    # get shard
    sharding = PositionalSharding(mesh_utils.create_device_mesh((len(jax.devices()),1)))

    # start training for each epoch
    if cfg.train_and_test.mode == "train":

      for epoch in range(cfg.train_and_test.train.epochs): 
          for i, (data, (labels, text_embeddings)) in enumerate(train_dataset): # batch training
              # Check if we should terminate early, so we can properly log Wandb before being killed.
              if cfg.time.time_termination and (time()-START_TIME >= cfg.time.time_of_termination_h*60*60): # convert hours into seconds.
                  TIME_EXCEEDED = True
                  #break
              
              # split key to keep randomness "random" for each training batch
              key, *subkey = jax.random.split(key, 4)

              data = jax.device_put(data ,sharding.reshape((1,len(jax.devices()))))

              # get timesteps given random key for this batch and data shape
              # TODO: Strictly this changes from sde to sde
              timesteps = jax.random.uniform(subkey[0], (data.shape[0],), minval=1e-5, maxval=1)
       
              # TODO: Potentially not memory efficient in terms of how this replication is done
              #timesteps = jax.device_put(timesteps, sharding.reshape(-1).replicate(0))

              # Perturb the data with the timesteps through sampling sde trick (for speed, see paper for explanation)
              perturbed_data, z = SDE.sample(timesteps, data, subkey[1])
            
              #perturbed_data = jax.device_put(perturbed_data,sharding.reshape((1,len(jax.devices()))))


              # scale timesteps for more significance
              #scaled_timesteps = timesteps*999
              scaled_timesteps = timesteps

              # get grad for this batch
                # loss_value, grads = jax.value_and_grad(loss_fn)(model_parameters, model_call, data, labels, t) # is this extra computation time

              grads = grad_fn(model_call, model_parameters, data, perturbed_data, scaled_timesteps, z, text_embeddings,subkey[2])

              # get change in model_params and new optimizer params
              # optim_parameters, model_parameters = optim_alg(optim_parameters, model_parameters, t_data, labels)
              updates, optim_parameters = optimizer.update(grads, optim_parameters, model_parameters)

              # update model params
              model_parameters = optax.apply_updates(model_parameters, updates)

              # Logging loss and an image
              if i % cfg.wandb.log.frequency == 0:
                    if cfg.wandb.log.loss:
                      loss = loss_fn(model_call, model_parameters, data, perturbed_data, scaled_timesteps, z, text_embeddings, subkey[2])
                      wandb.log({"loss": loss})

                    if (cfg.wandb.log.accuracy and i % 1000) and cfg.model.type == "classifier":
                        predicted_classes = jnp.argmax(inference_model(data, scaled_timesteps, text_embeddings, model_parameters, key), axis=1)
                        correct_classes = jnp.argmax(text_embeddings, axis=1)
                        wandb.log({"accuracy": jnp.mean(predicted_classes == correct_classes)})
                      # wandb.log({"loss": loss_value})
                    if cfg.wandb.log.img and i % 100 == 0 and cfg.model.type == "score":
                      # reverse sde sampling
                      drift = lambda t,y, args: SDE.reverse_drift(y, jnp.array([t]), args)
                      diffusion = lambda t,y, args: SDE.reverse_diffusion(y, jnp.array([t]), args)
                      get_sample = lambda t, key1, key0, xt, text_embedding: sample(1e-5, 0, t.astype(float)[0], -1/1000, drift, diffusion, [inference_model, text_embedding,model_parameters if cfg.model.name != "sde" else data[0], key0], xt, key1) 
                                      # dt0 = - 1/N

                      n = len(perturbed_data) 
                      if cfg.wandb.log.n_images < n:
                          n = cfg.wandb.log.n_images 

                      key, *subkey = jax.random.split(key, len(perturbed_data)*2 + 1)

                      args = (timesteps.reshape(-1,1)[:n], jnp.array(subkey[:len(subkey)//2])[:n], jnp.array(subkey[len(subkey)//2:])[:n], perturbed_data[:n], text_embeddings[:n])
                      images = jax.vmap(get_sample, (0, 0, 0, 0, 0))(*args)
                 
                    
                      Z = (jax.random.normal(key, data.shape)*255)[:n]
                      args = (jnp.ones_like(timesteps.reshape(-1,1))[:n], jnp.array(subkey[:len(subkey)//2])[:n], jnp.array(subkey[len(subkey)//2:])[:n], Z, text_embeddings[:n])
                      normal_distribution = jax.vmap(get_sample, (0, 0, 0, 0, 0))(*args)

                      inference_out = inference_model(perturbed_data[:n], timesteps[:n], text_embeddings[:n],model_parameters if cfg.model.name != "sde" else data[:n], key)
                     
                      Z_T, _ = SDE.sample(jnp.ones_like(timesteps[:n]), jnp.zeros_like(data[:n]) + TRAIN_MEAN, subkey[0])
                    
                      args = (jnp.ones_like(timesteps.reshape(-1,1))[:n], jnp.array(subkey[:len(subkey)//2])[:n], jnp.array(subkey[len(subkey)//2:])[:n], Z_T, text_embeddings[:n])
                      mean_normal_distribution = jax.vmap(get_sample, (0, 0, 0, 0, 0))(*args)

                      Z_0, _ = SDE.sample(jnp.ones_like(timesteps[:n]), jnp.zeros_like(data[:n]), subkey[0])
                      args = (jnp.ones_like(timesteps.reshape(-1,1))[:n], jnp.array(subkey[:len(subkey)//2])[:n], jnp.array(subkey[len(subkey)//2:])[:n], Z_0, text_embeddings[:n])
                      zero_normal_distribution = jax.vmap(get_sample, (0, 0, 0, 0, 0))(*args)

                      display_images(cfg, images, labels[:n], log_title="Reverse Sample x(t) -> x(0)")
                      display_images(cfg, perturbed_data[:n], labels[:n], log_title="Perturbed images")
                      display_images(cfg, min_max_rescale(perturbed_data[:n]), labels[:n], log_title="Min-Max Rescaled")
                      display_images(cfg, normal_distribution, labels[:n], log_title="N(0,I) -> x(0)")
                      display_images(cfg, min_max_rescale(normal_distribution), labels[:n], log_title="N(0,I) -> x(0), min-max rescaled")
                      display_images(cfg, Z_0, labels[:n], log_title="Pertrubed 0")
                      display_images(cfg, zero_normal_distribution, labels[:n], log_title="Perturbed 0 -> x(0)")
                      display_images(cfg, Z_T, labels[:n], log_title="Pertrubed TRAIN MEAN + N(0,I)")
                      display_images(cfg, mean_normal_distribution, labels[:n], log_title="Perturbed TRAIN MEAN + N(0,I) -> x(0)")
                      display_images(cfg, Z, labels[:n], log_title="N(0,I)")
                      display_images(cfg, data[:n], labels[:n], log_title="Original Images: x(0)")
                      display_images(cfg, min_max_rescale(inference_out), labels[:n], log_title="Model output, min-max rescaled")

                    if (cfg.wandb.log.parameters and i % 1000 == 0):
                            file_name = get_save_path_names(cfg)
                            with open(os.path.join(wandb.run.dir, file_name["model"]), 'wb') as f:
                              pickle.dump((epoch*len(train_dataset) + i, model_parameters), f, pickle.HIGHEST_PROTOCOL)
                              f.close()
                            #wandb.save(file_name["model"])
                            with open(os.path.join(wandb.run.dir, file_name["optimizer"]), 'wb') as f:
                              pickle.dump((epoch*len(train_dataset) + i, optim_parameters), f, pickle.HIGHEST_PROTOCOL)
                              f.close()
                            #wandb.save(file_name["optimizer"])
    elif cfg.train_and_test.mode == "validation":

        # TODO SHARD TEST DATA
        all_data, all_labels, all_embeddings = get_all_test_data(cfg, test_dataset)
        split_factor = cfg.train_and_test.test.split_factor 
        assert len(all_data) % split_factor == 0, f"split factor {split_factor} doesn't divide the length of the data {len(all_data)}"

    
           
        if cfg.model.type == "score":
          

          drift = lambda t,y, args: SDE.reverse_drift(y, jnp.array([t]), args)
          diffusion = lambda t,y, args: SDE.reverse_diffusion(y, jnp.array([t]), args)
          get_sample = lambda t, key1, key0, xt, text_embedding: sample(1e-5, 0, t.astype(float)[0], -1/1000, drift, diffusion, [inference_model, text_embedding,model_parameters if cfg.model.name != "sde" else all_data[0], key0], xt, key1) 

          key, *subkey = jax.random.split(key, len(all_data) * 2 + 1)

          timesteps = jnp.ones((all_data.shape[0],))
          Z_0, _ = SDE.sample(timesteps, jnp.zeros_like(all_data), subkey[0])


          args = (timesteps.reshape(-1, 1), jnp.array(subkey[:len(subkey)//2]), jnp.array(subkey[len(subkey)//2:]), Z_0, all_embeddings)

          all_generated_imgs = []
          for i in range(len(all_data)//split_factor):
            arg = [x[i*split_factor:(i+1)*split_factor] for x in args]
            generated_imgs = jax.vmap(get_sample, (0, 0, 0, 0, 0))(*arg)
            all_generated_imgs += list(generated_imgs)
          all_generated_imgs = jnp.array(all_generated_imgs)

          display_images(cfg, all_generated_imgs[:10], all_labels.reshape(-1)[:10], log_title="Perturbed 0 -> x(0)")
          fid = fid_model(all_generated_imgs, all_data[:len(all_generated_imgs)])
          wandb.log({"FID GEN x DATA": fid})
          # sanity check

          if cfg.wandb.log.accuracy:
            classifier_parameters, classifier = get_classifier(cfg)
            all_predicted_classes = []
            for i in range(len(all_data)//split_factor):
              all_predicted_classes += jnp.argmax(classifier(all_generated_imgs[i*split_factor:(i+1)*split_factor], None, None, classifier_parameters, key), axis=1)

            all_predicted_classes = jnp.array(all_predicted_classes) 
            
            wandb.log({"accuracy on test": jnp.mean(all_labels == all_predicted_classes)})



          fid_data = fid_model(jax.random.permutation(key, all_data[:1000], axis=0, independent=False), all_data[:1000], force_recompute=True)
          wandb.log({"FID DATA x DATA": fid_data})

        elif cfg.model.type == "classifier":
          all_correct_classes = jnp.argmax(all_embeddings, axis=1)
          all_predicted_classes = []
          for i in range(len(all_data)//split_factor):
            all_predicted_classes += jnp.argmax(inference_model(all_data[i*split_factor:(i+1)*split_factor], None, None, model_parameters, key), axis=1)
          all_predicted_classes = jnp.array(all_predicted_classes) 
          wandb.log({"accuracy on test": jnp.mean(all_correct_classes == all_predicted_classes)})
        

if __name__ == "__main__":
    run_experiment()
