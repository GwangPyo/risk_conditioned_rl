import jax.numpy as jnp
from flax import nnx
import jax
from functools import partial


class Sampler(object):
    def __init__(self,
                 batch_size: int = 512,
                 sample_size: int = 100,
                 *,
                 seed):
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.seed = nnx.Rngs(seed)

    @partial(jax.jit, static_argnums=(0,))
    def _sample(self, key):
        keys = jax.random.split(key, 5)
        # m = jax.random.uniform(keys[0], shape=(self.batch_size, 1))
        # M = (1 - m) * jax.random.uniform(keys[1], shape=(self.batch_size, 1)) + m

        mM = jax.random.beta(keys[0], shape=(self.batch_size, 2), a=0.1, b=0.1).sort(axis=-1)
        m, M = mM[..., 0], mM[..., 1]
        m = m[..., None]
        M = M[..., None]

        x = jax.random.uniform(keys[2], shape=(self.batch_size, self.sample_size - 2))
        x = jnp.concatenate([jnp.zeros_like(x[..., [0]]), x, jnp.ones_like(x[..., [1]])], axis=-1)

        alpha = jnp.ones(shape=(self.batch_size, self.sample_size))
        pmf = jax.random.dirichlet(keys[2], shape=(self.batch_size, ), alpha=alpha)
        x_bar = m + (M - m) * jax.random.uniform(keys[3], shape=(self.batch_size, self.sample_size)).sort(axis=-1)

        cdf = jnp.cumsum(pmf, axis=-1)
        y = jax.vmap(jnp.interp, in_axes=(0, 0, 0), out_axes=0)(x, cdf, x_bar)
        # y = (M - m) * y + m

        return x.sort(axis=-1), y.sort(axis=-1)

    def sample(self):
        return self._sample(self.seed())


