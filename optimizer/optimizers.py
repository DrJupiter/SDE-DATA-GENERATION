import jax.numpy as jnp
import optax

# https://github.com/deepmind/optax


def get_optim(cfg,params=None):
    if cfg.optimizer.name == "adam":
        return adam(cfg,params)
    raise ValueError(f"Model {cfg.model.name} not found")

def adam(cfg,params):
    #exponential_decay = optax.warmup_exponential_decay_schedule(1/5000, 1, 0, 5000, 1e-8, 0, False, 2e-4)
    clip = optax.clip_by_global_norm(1.0)
    lr = cfg.optimizer.learning_rate*10
    scale_term = 0.0001
    sgd_with_warm_restarts_decay_scheduler = optax.sgdr_schedule(
    [{"init_value":lr, "peak_value":lr+scale_term, "decay_steps":1000, "warmup_steps":100, "end_value":lr - scale_term},
     {"init_value":lr - scale_term, "peak_value":lr, "decay_steps":1000, "warmup_steps":100, "end_value":max(lr - 2 * scale_term, scale_term)},
     {"init_value":max(scale_term, lr - 2*scale_term), "peak_value":lr-scale_term, "decay_steps":1000, "warmup_steps":100, "end_value":lr/10},
    ]
    )
    #adam = optax.adam(learning_rate=cfg.optimizer.learning_rate)
    adam = optax.adam(learning_rate=sgd_with_warm_restarts_decay_scheduler)
    optimizer = optax.chain(clip, adam) 
    assert params is not None, "pls give func get_optim the model parameters, for this optimiazation function"
    opt_param = optimizer.init(params)
    return optimizer, opt_param
