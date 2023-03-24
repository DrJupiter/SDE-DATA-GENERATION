
from sde import SDE
import jax
from jax import numpy as jnp

class SUBVPSDE(SDE):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.beta_min = cfg.sde.beta_min
        self.beta_max = cfg.sde.beta_max
        self.description = cfg.sde.description

    def parameters(self, t, x0):

        exponents = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min 

        # Get means for batch
        mean = jnp.exp(exponents) * x0
        covar = (1-jnp.exp( 2. * exponents)) 
        return mean, covar
    
    def p_xt_x0(self, xt, x0, t):
        """
        The conditional distribution p(x(t)|x(0))
        """
        mean, covar = self.parameters(t, x0)
        return jax.scipy.stats.norm(xt, mean,jnp.sqrt(covar))
    
    def sample(self, t, x0, key):
        # Split rng

        # Get paramters for the normal distribution
        mean, covar = self.parameters(t, x0)

        # We take the sqrt to get the matrix A in AA^T = COVARIANCE 
        std = jnp.sqrt(covar)

        key, subkey = jax.random.split(key) 
        z = jax.random.normal(subkey, x0.shape)
        # bacth mul
        return mean + std * z

    def score(self, x0, t, xt):
        mu, covariance = self.parameters(t, x0)

        
        #inv_covariance = jnp.linalg.inv(covariance)
        #-0.5 (inv_covariance + inv_covariance.T)(xt-x0)

        # As we have isotropic covariance, then we have
        return -1/covariance * (xt-mu)

    def __repr__(self) -> str:
        return self.description 

if __name__ == "__main__":
    from utils.utils import get_hydra_config
    config = get_hydra_config(overrides=["visualization.visualize_img=true","wandb.log.img=false"]) 
    subvpsde = SUBVPSDE(config)
    print(subvpsde)
    from data.dataload import dataload
    train, _ = dataload(config)
    iter_train = iter(train)
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    data, label = next(iter_train) 
    t = jax.random.uniform(subkey, data.shape, minval=0, maxval=1)
    print(key)
    xt = subvpsde.sample(t, data, key)
    from visualization.visualize import display_images
    if config.dataset.name == 'cifar10':
        label = [config.dataset.classes[int(idx)] for idx in label]
    display_images(config, xt, label)
    display_images(config, data, label)
    print(key)