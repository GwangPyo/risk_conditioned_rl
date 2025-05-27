import jax.numpy as jnp
from flax import nnx
import distrax
from typing import Callable, Sequence
from base.jax_layers import create_mlp


class Actor(nnx.Module):

    def __init__(self,
                 features_dim: int,
                 actions_dim: int,
                 net_arch: Sequence[int] = (256, 256),
                 activation_fn: Callable = nnx.relu,
                 layer_norm: bool = True,
                 *,
                 rngs: nnx.Rngs
                 ):
        self.actions_dim = actions_dim
        self.features_dim = features_dim
        self.net_arch = list(net_arch)
        self.activation_fn = activation_fn
        self.layer_norm = layer_norm
        self.rng_collection: nnx.Rngs = rngs
        self.layers = self.build_layer()

    def build_layer(self):
        return nnx.Sequential(
            #  FourierFeatureNet(128, input_dim=self.features_dim, rngs=self.rng_collection),
            *create_mlp(self.features_dim, self.actions_dim * 2, net_arch=self.net_arch,
                        activation_fn=self.activation_fn,
                        rngs=self.rng_collection)
        )

    def __call__(self, observations):
        mu_log_sigma = self.layers(observations)
        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1, )
        log_sigma = log_sigma.clip(-20., 1.)
        distr = distrax.Normal(loc=mu, scale=jnp.exp(log_sigma))
        return distrax.Transformed(distr, bijector=distrax.Tanh())

    def sample_and_log_prob(self, observation, *, sample_shape=(), seed=None, ):
        if seed is None:
            seed = self.rng_collection()
        sample, log_prob = self(observation).sample_and_log_prob(seed=seed, sample_shape=sample_shape)
        return sample, log_prob.sum(axis=-1, keepdims=True)

    def mode(self, observations):
        mu_log_sigma = self.layers(observations)
        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1, )
        return jnp.tanh(mu)
