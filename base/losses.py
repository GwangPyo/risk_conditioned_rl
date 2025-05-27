import jax
import jax.numpy as jnp


@jax.jit
def quanitle_regression_loss(target, predict, taus):
    pairwise_delta = target[..., None, :] - predict[..., None]
    abs_pairwise_delta = jnp.abs(pairwise_delta)
    taus = taus[..., None]
    loss = jnp.where(pairwise_delta < 0, (1 - taus) * abs_pairwise_delta, taus * abs_pairwise_delta)
    return loss
