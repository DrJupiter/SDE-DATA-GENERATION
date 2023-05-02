import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


#from sde_class import SDE
from sde.sde_class import SDE
import jax
from jax import numpy as jnp

# TODO: CHECK IF TIME AND NOISE IS APPLIED CORRECTLY, GO BACK FROM -INF;INF domain to color values

class SUBVPSDE(SDE):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.beta_min = cfg.sde.beta_min
        self.beta_max = cfg.sde.beta_max
        self.description = cfg.sde.description

    def parameters(self, t, x0):

        exponents = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min 

        # Get means for batch
        mean = jax.vmap(lambda a, b: a * b)(jnp.exp(exponents) , x0)
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
        return mean + jax.vmap(lambda a,b: a*b)(std , z), z

    def score(self, x0, t, xt):
        mu, covariance = self.parameters(t, x0)

        
        #inv_covariance = jnp.linalg.inv(covariance)
        #-0.5 (inv_covariance + inv_covariance.T)(xt-x0)

        # As we have isotropic covariance, then we have
        
        return jax.vmap(lambda a,b: a*b)(-1/covariance , (xt-mu))

    def drift(self, x, t):
        return -0.5 * (self.beta_min + t * (self.beta_max-self.beta_min)) * x 
    
    def diffusion(self, _x, t):
        return (self.beta_min + t * (self.beta_max-self.beta_min))**(1/2)

    def reverse_drift(self, x, t, args):
        sm = args[0] 
        return self.drift(x,t) - sm(x,t, *args[1:]) * self.diffusion(x,t)**2 
        #return self.drift(x,t) - sm(x,t, *args[1:]) * self.diffusion(x,t)**2 
    
    def reverse_diffusion(self, x, t, args):
        return jnp.identity(x.shape[0]) * self.diffusion(x, t)

    def __repr__(self) -> str:
        return self.description 

if __name__ == "__main__":
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    from utils.utils import get_hydra_config
    from visualization.visualize import display_images
    import sys
    config = get_hydra_config(overrides=["visualization.visualize_img=true","wandb.log.img=false"]) 
    subvpsde = SUBVPSDE(config)
    print(subvpsde)
    from data.dataload import dataload
    train, _ = dataload(config)
    iter_train = iter(train)
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    data, label = next(iter_train) 
    if config.dataset.name == 'cifar10':
        label = [config.dataset.classes[int(idx)] for idx in label]
    print(data)
    data = jnp.array(data, dtype=jnp.float32)
    print(data.shape)

    t = jax.random.uniform(subkey, (data.shape[0],), minval=0, maxval=1)

    xt = subvpsde.sample(t, data, key)
    subvpsde.score(data, t, xt)
    
    display_images(config, xt, label)
    display_images(config, data, label)
    from models.model import get_model
    key, subkey = jax.random.split(key)
    model = get_model(config, subkey)
