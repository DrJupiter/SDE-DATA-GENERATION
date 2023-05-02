import jax.numpy as jnp
def sum_diff_loss(parameters, model_pass, data_batch, target_batch, time_steps, _z, key = None):
    predictions = model_pass(data_batch, time_steps, parameters) # DDPM = f(x_in, timesteps, parameters)
    loss = jnp.sum(predictions)-jnp.sum(target_batch)
    return loss