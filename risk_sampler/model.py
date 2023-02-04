import haiku as hk
import jax.numpy as jnp
import jax.random
import optax
import numpy as np

from common_model.commons import MLP, PowerEmbeddingNet, MonotoneMLP
from utils.misc import unroll, Progbar
from utils import optimize
from jax import nn
from functools import partial
from risk_sampler.sampler import SampleY
from risk_sampler.plot import plot_all


class Decoder(hk.Module):
    def __init__(self):
        super().__init__(name='decoder')
        self.mlp = MLP(output_dim=64,
                       net_arch=[256, 256])
        self.embedding = PowerEmbeddingNet(n_pow=256, features_dim=64)
        self.monotone_mlp = MonotoneMLP(output_dim=1, net_arch=(128, 128))
        self.bias = MLP(output_dim=1, net_arch=(256, ))

    def __call__(self, z, condition, taus):
        z = self.mlp(jnp.concatenate((z, condition), axis=-1))
        pow_t = self.embedding(taus)
        t1 = self.monotone_mlp(nn.relu(z[:, None, :]) * pow_t).squeeze(axis=-1)
        return nn.sigmoid(jnp.log(t1 + 1e-8) + self.bias(z))

    @classmethod
    def factory(cls,
                key: jax.random.PRNGKey,
                z_placeholder: jnp.ndarray,
                condition_placeholder: jnp.ndarray,
                taus_placeholder: jnp.ndarray):

        def inner(z, condition, taus):
            return cls()(z, condition, taus)

        func = hk.without_apply_rng(inner)
        params = func.init(key, z_placeholder, condition_placeholder, taus_placeholder)
        return func, params


class Encoder(hk.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.mlp = MLP(output_dim=z_dim, net_arch=[256, 256])
        self.linear = hk.Linear(1)

    def __call__(self, taus, y):
        encoded = self.mlp(jnp.concatenate((taus, y), axis=-1))
        condition = self.linear(encoded)
        return nn.sigmoid(encoded), condition

    @classmethod
    def factory(cls,
                key: jax.random.PRNGKey,
                taus_placeholder: jnp.ndarray,
                ):

        def inner(x, y):
            return cls()(x, y)

        func = hk.without_apply_rng(inner)
        params = func.init(key, taus_placeholder, taus_placeholder)
        return func, params


class Critic(hk.Module):
    """
    WGAN-Critic
    """
    def __init__(self):
        super().__init__()

    def __call__(self, z):
        return  MLP(output_dim=1, net_arch=[256, 256])(z)


class AAE(hk.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim

    def __call__(self):
        encoder = Encoder(self.z_dim)
        decoder = Decoder()

        def init(taus, y, taus_prime):
            z, cond = encoder(taus, y)
            y_hat = decoder(z, cond, taus_prime)
            return y_hat
        return init, (encoder, decoder)

    @classmethod
    def factory(cls,
                key: jax.random.PRNGKey,
                z_dim: int,
                placeholder: jnp.ndarray
                ):
        def inner():
            return cls(z_dim)()
        f = hk.without_apply_rng(hk.multi_transform(inner))
        encoder, decoder = f.apply
        params = f.init(key, placeholder, placeholder, placeholder)
        return (encoder, decoder), params


class RiskProposalNet(object):
    def __init__(self, z_dim=256, n_quantiles=32, seed=0):
        super().__init__()
        self.z_dim = z_dim
        self.n_quantiles = n_quantiles
        self.rng = hk.PRNGSequence(seed)

        (self.encoder, self.decoder), self.param_generator = self.build()

        opt_init, self.optim = optax.adabelief(learning_rate=3e-4)
        self.opt_state = opt_init(self.param_generator)

    def build(self):
        placeholder = jnp.ones((1, self.n_quantiles))
        (encoder, decoder), param = AAE.factory(next(self.rng), self.z_dim, placeholder)
        return (encoder, decoder), param

    @partial(jax.jit, static_argnums=0)
    def _encode(self, param, taus, y):
        return self.encoder(param, taus, y)

    def encode(self, taus, y):
        return self._encode(self.param_generator, taus, y)

    @partial(jax.jit, static_argnums=0)
    def _decode(self, param, z, c, taus):
        return self.decoder(param, z, c, taus)

    def decode(self, z, c, taus):
        return self._decode(self.param_generator, z, c, taus)

    def save(self, path):
        np.savez(path, **self.param_generator)

    def load_param(self, path):
        data = dict(np.load(path, allow_pickle=True))
        data = unroll(data)
        self.param_generator = hk.data_structures.to_immutable_dict(data)
        return self


class Trainer(object):
    def __init__(self, z_dim=256, n_quantiles=32, seed=0):
        self.z_dim = z_dim
        self.n_quantiles = n_quantiles
        self.risk_proposal_net = RiskProposalNet(
            z_dim=z_dim,
            n_quantiles=n_quantiles,
            seed=seed
        )
        self.rng = hk.PRNGSequence(seed)

        def fn_disc(z):
            return Critic()(z)

        self.discriminator = hk.without_apply_rng(hk.transform(fn_disc))
        self.param_disc = self.discriminator.init(next(self.rng), jnp.ones((1, self.z_dim)))

        opt_init, self.d_opt = optax.adabelief(learning_rate=3e-4)
        self.d_opt_state = opt_init(self.param_disc)

        self.sampler = SampleY(n_quantiles=n_quantiles)
        self.batch_size = 1024

    @staticmethod
    @jax.jit
    def quantile_loss(y: jnp.ndarray,
                      x: jnp.ndarray,
                      taus: jnp.ndarray,
                      delta: jnp.ndarray = 0.01
                      ) -> jnp.ndarray:
        pairwise_delta = y[:, None, :] - x[:, :, None]
        abs_pairwise_delta = jnp.abs(pairwise_delta)
        huber = jnp.where(abs_pairwise_delta > delta,
                          delta * (abs_pairwise_delta - delta * 0.5),
                          pairwise_delta ** 2 * 0.5)/delta
        loss = jnp.abs(taus[..., None] - jax.lax.stop_gradient(pairwise_delta < 0)) * huber
        return loss

    @partial(jax.jit, static_argnums=0)
    def generator_loss(self,
                       param_generator: hk.Params,
                       param_discriminator: hk.Params,
                       taus: jnp.ndarray,
                       y: jnp.ndarray,
                       taus_prime: jnp.ndarray,
                       key: jax.random.PRNGKey
                       ):

        z_distribution, condition = self.risk_proposal_net.encoder(param_generator, taus, y)

        decoded_distribution = self.risk_proposal_net.decoder(param_generator,
                                                              z_distribution,
                                                              condition,
                                                              taus_prime)
        reconstruction_loss = self.quantile_loss(y,
                                                 decoded_distribution,
                                                 taus_prime).sum(axis=(-1, -2))

        z_target = jax.random.uniform(key=key, shape=z_distribution.shape)
        # z_target = z_target.at[:, 0].set(y_mean)

        pred_gen = self.discriminator.apply(param_discriminator, z_distribution)
        # pred_true = self.discriminator.apply(param_discriminator, z_target)

        w_loss = -pred_gen.mean(axis=-1)
        taus_prime = jnp.linspace(0, 1, self.n_quantiles)[None]
        taus_prime = jnp.repeat(taus_prime, axis=0, repeats=taus.shape[0])
        rand_condition = jax.random.uniform(key=key, shape=condition.shape)
        decoded_target = self.risk_proposal_net.decoder(param_generator,
                                                        z_target,
                                                        rand_condition,
                                                        taus_prime
                                                        )
        # To be able to conditional generate

        condition_loss = jnp.power(100 * (rand_condition - decoded_target.mean(axis=-1, keepdims=True)), 2) \
                         + jnp.power(100* (condition - y.mean(axis=-1, keepdims=True)), 2)

        return (10 * reconstruction_loss.mean() + w_loss.mean() + condition_loss.mean()).mean(), \
            (reconstruction_loss.mean(), w_loss.mean(), z_distribution, z_target, condition_loss)

    @partial(jax.jit, static_argnums=0)
    def discriminator_loss(self,
                           param_discriminator: hk.Params,
                           z_true: hk.Params,
                           z_false: hk.Params,
                           key: jax.random.PRNGKey
                           ):
        pred_gen = self.discriminator.apply(param_discriminator, z_false)
        pred_true = self.discriminator.apply(param_discriminator, z_true)
        w = pred_gen.mean() - pred_true.mean()

        alpha = jax.random.uniform(key=key, shape=(pred_gen.shape[0], 1))
        interpolated = alpha * z_false + (1. - alpha) * z_true

        def f_disc(x):
            return self.discriminator.apply(param_discriminator, x).mean()

        grad = jax.vmap(jax.grad(f_disc))(interpolated)
        grad_norm = jnp.linalg.norm(grad + 1e-12, axis=-1)
        grad_penalty = 10 * ((grad_norm - 1.) ** 2).mean()
        return (w.mean() + grad_penalty), (w, grad_penalty)

    def train_step(self):
        taus, y = self.sampler.sample(batch_size=self.batch_size)
        taus_prime = jax.random.uniform(key=next(self.rng), shape=taus.shape)
        taus_prime = jnp.sort(taus_prime, axis=-1)
        # optimize generator
        self.risk_proposal_net.opt_state, self.risk_proposal_net.param_generator, \
            loss, aux = optimize(self.generator_loss,
                                 self.risk_proposal_net.optim,
                                 self.risk_proposal_net.opt_state,
                                 params_to_update=self.risk_proposal_net.param_generator,
                                 param_discriminator=self.param_disc,
                                 taus=taus, y=y, taus_prime=taus_prime,
                                 key=next(self.rng))

        (reconstruction_loss, g_loss, z_false, z_true, conditional) = aux
        for _ in range(5):
            self.d_opt_state, self.param_disc, d_loss, (w, grad_penalty) = optimize(
                self.discriminator_loss,
                self.d_opt,
                self.d_opt_state,
                self.param_disc,
                z_true, z_false, key=next(self.rng))

        return [("g_loss", g_loss.item()),
                ("recon_loss", reconstruction_loss.mean().item()),
                ("d_loss", w.mean().item()),
                ("conditional", conditional.mean().item()),
                ("grad_penalty", grad_penalty.mean().item())
                ]

    def train_epoch(self, n_epoch=100000):
        progbar = Progbar(n_epoch)
        z = jax.random.uniform(key=next(self.rng), shape=(10, self.z_dim))
        arange = jnp.arange(0, 10, dtype=jnp.float32).reshape(-1, 1) / 10.
        taus= jnp.arange(0, 1, 1/self.n_quantiles)[None]
        taus = jnp.repeat(taus, axis=0, repeats=10)

        print(self.risk_proposal_net.decode(z, arange, taus).mean(axis=-1))
        for _ in range(n_epoch):
            losses = self.train_step()
            progbar.add(1, losses)

    def save(self, path):
        self.risk_proposal_net.save(path)



