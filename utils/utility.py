
from hydra import compose, initialize
import hydra


import models 

import jax.numpy as jnp
from jax import vmap
import jax

import pickle
import os

def get_hydra_config(config_path='../configs', job_name='test', version_base='1.3', config='defaults.yaml', overrides=[], reload=True):
    """
    !!! This function is meant for TESTING PURPOSES, not to be used
    in production. !!!

    Load the hydra config manually.
    
    (The parameters are the same as loading a hydraconfig normally)
    """

    if reload:
       hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path=config_path, job_name=job_name, version_base=version_base) 
    cfg = compose(config, overrides=overrides)
    return cfg


batch_matmul = vmap(lambda a,b: jnp.matmul(a.T, b), (0, 0) , 0)
"""
    dim(a): B x W x L1 | B x L1 
    dim(b): B x W x L2 | B x L1 

    Performs matmull across the batch dimension B.

    return: B x L1 x L2 | B  
"""


def get_save_path_names(cfg):
    file_name = {}
    file_name['model'] = f"{cfg.dataset.name}-{cfg.model.name}-parameters.pickle"
    file_name['optimizer'] = f"{cfg.dataset.name}-{cfg.model.name}-{cfg.optimizer.name}-parameters.pickle"
    file_name["test_data"] = f"{cfg.dataset.name}-{cfg.model.type}-test.npz"
    file_name["test_data_statistics"] = f"{cfg.train_and_test.test.fid_model_type}-{cfg.dataset.name}-{cfg.model.name}-{cfg.loss.name}-test-statistics.npz"
    return file_name 


def get_classifier(cfg):

    # ! DON'T DO THIS UNLESS YOU KNOW WHAT YOU'RE DOING, IT WILL MESS THINGS UP OTHERWISE
        # Modify the config
    cfg = get_hydra_config(overrides=[f'model={cfg.parameter_loading.classifier_name}', "model.type=classifier", "parameter_loading.model=True", f"parameter_loading.model_path={cfg.parameter_loading.classifier_path}"])

    # Generate key as in main
    key = jax.random.PRNGKey(cfg.model.key)
    key, subkey = jax.random.split(key)

    # get model
    parameters, _, classifier = models.get_model(cfg, subkey)
    # load parameters
    parameters = load_model_paramters(cfg, parameters)

    # Reset the config
    get_hydra_config()

    return parameters, classifier

    



def load_model_paramters(cfg, model_paramters):
    file_name = get_save_path_names(cfg) 
    if cfg.parameter_loading.model:
        if os.path.isdir(cfg.parameter_loading.model_path):
            with open(os.path.join(cfg.parameter_loading.model_path, file_name['model']), "rb") as mp:
                iteration, model_paramters = pickle.load(mp)
        
        elif os.path.isfile(cfg.parameter_loading.model_path):
            with open(cfg.parameter_loading.model_path, "rb") as mp:
                iteration, model_paramters = pickle.load(mp)
        else:
            raise FileNotFoundError(f"Unable to find {cfg.parameter_loading.model_path}")

        if cfg.model.sharding:
            model_paramters = get_model_sharding(cfg)(model_paramters)
        print(f"Loaded model: {cfg.model.name}-{cfg.model.type} paramters @ checkpoint iteration {iteration}")
        print(f"\t Model path: {file_name['model']}")
    return model_paramters

def load_optimizer_paramters(cfg, optimizer_paramters):

    file_name = get_save_path_names(cfg) 

    if cfg.parameter_loading.optimizer:
        if os.path.isdir(cfg.parameter_loading.optimizer_path):

            with open(os.path.join(cfg.parameter_loading.optimizer_path, file_name["optimizer"]), "rb") as op:
                iteration, optimizer_paramters = pickle.load(op)

        elif os.path.isfile(cfg.parameter_loading.optimizer_path):
            with open(cfg.parameter_loading.optimizer_path, "rb") as op:
                iteration, optimizer_paramters = pickle.load(op)
        else:
            raise FileNotFoundError(f"Unable to find {cfg.parameter_loading.optimizer_path}")

        print(f"Loaded optimizer: {cfg.optimizer.name} paramters @ checkpoint iteration {iteration}")

    return optimizer_paramters


def get_model_sharding(cfg):
    if cfg.model.name == "dummy_jax":
        return models.shard_dummy 
    elif cfg.model.name == "ddpm_unet":
        if cfg.model.type == "score":

            return models.shard_score_ddpm_unet
        elif cfg.model.type == "classifier":
            return models.shard_classifier_ddpm_unet 
        
    raise NotImplementedError(f"Sharding not implemented for {cfg.model.name}")

def get_wandb_input(cfg):
    args = {}
    args["entity"] = cfg.wandb.setup.entity
    args["project"] = cfg.wandb.setup.project

    # Sanity check for type of the model
    assert cfg.model.type == cfg.loss.type, f"The model type {cfg.model.type} != {cfg.loss.type}, make sure to choose ones which match or change the model type"

    if cfg.model.sharding:
        n = jax.device_count()        

        if cfg.train_and_test.mode == "train":
            batch_remainder = cfg.train_and_test.train.batch_size % n
            assert batch_remainder == 0, f"Train Batch Size {cfg.train_and_test.train.batch_size} mod {n} = {batch_remainder} !=0, Thus sharding will fail" 

        if cfg.wandb.log.img:
            assert cfg.wand.log.n_images % n == 0, f"Producing Images will fail due to incombatible sharding and image amounts: {cfg.wandb.log.n_images} mod {n} = {cfg.wandb.log.n_images % n} != 0"
        

    tags = [cfg.wandb.setup.experiment, cfg.loss.name, cfg.model.name, cfg.sde.name, cfg.dataset.name]
    if cfg.wandb.setup.experiment == "train":
        pass
    elif cfg.wandb.setup.experiment == "model_size":
        additional_tags = []
        if cfg.model.name == "ddpm_unet":
            sizes = cfg.model.parameters.Channel_sizes
            additional_tags += [f"channel{i}-{size}" for i,size in enumerate(sizes)]
            additional_tags += [f"scaling{i}{factor}" for i, factor in enumerate(cfg.model.hyperparameters.scaling_factors)]
            additional_tags += [f"Time, Text: {cfg.model.hyperparameters.time_embedding_dims}", f"Time inner: {cfg.model.hyperparameters.time_embedding_inner_dim}"]
        tags += additional_tags
    elif cfg.wandb.setup.experiment == "batch_size":
        additional_tags = [f"{cfg.train_and_test.train.batch_size}"]
        tags += additional_tags
    
    args["tags"] = tags
    return args 

def min_max_rescale(img):
    mins, maxs = jnp.min(img, axis=1).reshape(-1,1), jnp.max(img, axis=1).reshape(-1,1)

    return (img-mins)/(maxs-mins) * 255 