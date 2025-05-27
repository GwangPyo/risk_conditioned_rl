import jax.random

from base.jax_layers import create_mlp, FourierFeatureNetwork
from flax import nnx
import jax.numpy as jnp
from base.cond_nf import CondFlow
from base import quanitle_regression_loss
from functools import partial


class Encoder(nnx.Module):
    def __init__(self,
                 input_dims: int,
                 latent_dims: int,
                 *,
                 rngs
                 ):
        self.inputs = nnx.Sequential(FourierFeatureNetwork(input_dims, 128, rngs=rngs),
                                     *create_mlp(128, latent_dims, rngs=rngs),
                                     nnx.sigmoid
                                     )

    def __call__(self, x):
        return self.inputs(x)


class Decoder(nnx.Module):
    def __init__(self, latent_dim, *, rngs):
        self.latents = FourierFeatureNetwork(latent_dim - 1, latent_dim, rngs=rngs)
        self.kappa = FourierFeatureNetwork(1, latent_dim, rngs=rngs)
        self.outputs = CondFlow(latent_dim * 2, 4, 4, rngs=rngs)

    def __call__(self, z, taus):
        z_ = self.latents(z[..., :-1])
        kappa = self.kappa(z[..., -1][..., None])
        z_ = jnp.concatenate([z_, kappa], axis=-1)
        return self.outputs(z_, taus)

    def forward_and_log_det(self, z, taus):
        z_ = self.latents(z[..., :-1])
        kappa = self.kappa(z[..., -1][..., None])
        z_ = jnp.concatenate([z_, kappa], axis=-1)
        return self.outputs.forward_and_log_det(z_, taus)


class RiskProposalNetwork(nnx.Module):
    def __init__(self, inputs_dim: int, latent_dim: int,
                 max_cutoff: float = 0.75,
                 disc_coef: float = 3,
                 quantile_coef: float = 10,
                 cut_off_coef: float = 5,
                 *, rngs):
        self.n_x = inputs_dim // 2
        self.encoder = Encoder(inputs_dim, latent_dim, rngs=rngs)
        self.latent_dim = latent_dim
        self.decoder = Decoder(latent_dim, rngs=rngs)
        self.max_cutoff = max_cutoff
        self.rngs = rngs

        self.quantile_coef = quantile_coef
        self.disc_coef = disc_coef
        self.cut_off_coef = cut_off_coef

    def loss_fn(self, x, y, taus):
        xy = jnp.concatenate([x, y], axis=-1)
        area = jnp.trapezoid(y, x, axis=-1)
        z = self.encoder(xy)
        z_real = jax.random.uniform(self.rngs(), shape=z[..., :-1].shape)
        aae_loss = imq_kernel(self.disc_coef * z_real, self.disc_coef * z[..., :-1],
                              h_dim=z_real.shape[-1]) / z.shape[0]

        kappa = z[..., -1]
        z_label = jnp.concatenate([z[..., :-1], area[..., None]], axis=-1)
        decoded = self.decoder(z_label, taus)
        aae_loss = aae_loss
        q_loss = quanitle_regression_loss(y, decoded, taus).sum(axis=(-1)).mean()

        cut_off_loss = jnp.mean((kappa - area) ** 2, axis=0)
        cut_off_loss = cut_off_loss + ((area - decoded.mean()) ** 2).mean()
        return (self.quantile_coef * q_loss + self.cut_off_coef * cut_off_loss + aae_loss,
                {"q_loss": q_loss, "cut_off": cut_off_loss, "aae_loss": aae_loss, "z_fake": z[..., :-1]})

    def __call__(self, ph):
        key = self.rngs()
        z = jax.random.uniform(key, (ph.shape[0], self.latent_dim))
        z = z.at[..., -1].set(z[..., -1] * self.max_cutoff)
        return z

    def generate(self, taus):
        z = self.__call__(taus)
        taus = jax.random.uniform(self.rngs(), shape=(taus.shape[0], 64))
        taus = taus.sort(axis=-1)
        y = self.decoder(z, taus)
        return z, y

    def plot_generate(self, ph):
        z = self.__call__(ph)
        taus = jax.random.uniform(self.rngs(), shape=(ph.shape[0], 64))
        taus = taus.sort(axis=-1)
        y = self.decoder(z, taus)
        return taus, y, z[..., -1]

    @staticmethod
    def save(path, state):
        with open(path, 'wb') as f:
            import pickle
            pickle.dump(state, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            import pickle
            state = pickle.load(f)
        return state


@partial(jax.jit, static_argnums=2)
def imq_kernel(X: jax.Array,
               Y: jax.Array,
               h_dim: int):
    batch_size = X.shape[0]

    norms_x = (X ** 2).sum(1, keepdims=True)  # batch_size x 1
    prods_x = jnp.matmul(X, X.transpose())  # batch_size x batch_size
    dists_x = norms_x + norms_x.transpose() - 2 * prods_x

    norms_y = (Y ** 2).sum(1, keepdims=True)  # batch_size x 1
    prods_y = jnp.matmul(Y, Y.transpose())  # batch_size x batch_size
    dists_y = norms_y + norms_y.transpose() - 2 * prods_y

    dot_prd = jnp.matmul(X, Y.transpose())
    dists_c = norms_x + norms_y.transpose() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10., 30.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        res1 = (1 - jnp.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats += res1 - res2

    return stats
