import jax.numpy as jnp
import optax

# https://github.com/deepmind/optax


def get_optim(cfg,params=None):
    if cfg.optimizer.name == "adam":
        return adam(cfg,params)
    raise ValueError(f"Model {cfg.model.name} not found")

def adam(cfg,params):
    #exponential_decay = optax.warmup_exponential_decay_schedule(1/5000, 1, 0, 5000, 1e-8, 0, False, 2e-4)
    decay=optax.warmup_cosine_decay_schedule(
  init_value=0.0,
  peak_value=1.0,
  warmup_steps=50,
  decay_steps=1_000,
  end_value=0.0,
    )
    clip = optax.clip_by_global_norm(1.0)
    adam = optax.adamw(learning_rate=decay)
    optimizer = optax.chain(clip, adam) 
    assert params is not None, "pls give func get_optim the model parameters, for this optimiazation function"
    opt_param = optimizer.init(params)
    return optimizer, opt_param
