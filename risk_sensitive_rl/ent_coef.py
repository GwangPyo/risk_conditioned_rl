from flax import nnx
import jax.numpy as jnp


class EntCoef(nnx.Module):
    def __init__(self,
                 init: float = 1e-2,
                 *, rngs):
        # no need for rngs, but just for the consistency
        self.lag_param = nnx.Param(self.denormalize(jnp.ones(shape=(), dtype=jnp.float32) * init))

    @staticmethod
    def normalize(x):
        return jnp.exp(x)

    @staticmethod
    def denormalize(x):
        return jnp.log(x)

    def __call__(self):
        return self.normalize(self.lag_param.value)
