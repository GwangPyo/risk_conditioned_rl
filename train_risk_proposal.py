from risk_sampler.sampler import Sampler
from risk_sampler.risk_proposal_network import RiskProposalNetwork

from flax.nnx import Rngs
from flax import nnx
import optax
import jax

from tqdm import trange
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from typing import Callable
from fire import Fire


@jax.jit
def update_fn(risk_graph, risk_state,
              x, y, taus):
    model, opt, metric = nnx.merge(risk_graph, risk_state)

    def loss_fn(m):
        loss, aux = m.loss_fn(x, y, taus)
        return loss, aux

    grad, aux = nnx.grad(loss_fn, has_aux=True)(model)
    opt.update(grad)
    z_fake = aux.pop('z_fake')
    metric.update(**aux)
    _, new_state = nnx.split((model, opt, metric))
    return new_state, z_fake


def plot_generate(model, n_plot: int = 100):
    ph = jnp.ones(shape=(n_plot))
    x, y, kappa = model.plot_generate(ph)
    cmap = cm.get_cmap('inferno')
    norm = mcolors.Normalize(vmin=0, vmax=model.max_cutoff)

    for i in range(x.shape[0]):
        color = cmap(norm(kappa[i].item()))
        plt.plot(np.asarray(x[i]), np.asarray(y[i]), color=color)
    plt.show()


def plot_cvar(model, n_plot: int = 200):
    ph = jnp.ones(shape=(n_plot))
    cmap = cm.get_cmap('inferno')
    norm = mcolors.Normalize(vmin=0, vmax=2)

    x = np.linspace(0, 1, model.n_x)[None]
    x = np.repeat(x, axis=0, repeats=n_plot // 2)
    alpha = np.linspace(0, 1, n_plot // 2)[..., None]
    y = x * alpha
    y_seek = alpha + (1 - alpha) * x

    y = jnp.concatenate([y, y_seek], axis=0)
    x = jnp.concatenate([x, x], axis=0)
    z = model.encoder(jnp.concatenate([x, y], axis=-1))
    alpha = jnp.concatenate([alpha, 1 + alpha], axis=0)
    y_hat = model.decoder(z, x)
    for i in range(x.shape[0]):
        color = cmap(norm(alpha[i, 0].item()))
        plt.plot(x[i], y_hat[i], color=color)
    plt.show()


def plot(model, n_plot):
    plot_generate(model, n_plot)
    plot_cvar(model, )


def adabelief_w(learning_rate: float = 3e-4,
                weight_decay: float = 1e-4,
                **kwargs
                ):
    return optax.chain(optax.scale_by_belief(**kwargs),
                       optax.add_decayed_weights(weight_decay),
                       optax.scale_by_learning_rate(learning_rate=learning_rate, )
                       )


def train(save_path: str,
          inputs_dim: int = 200,
          latent_dim: int = 256,
          batch_size: int = 256,
          verbose: int = 1,
          optimizer_class: Callable = adabelief_w,
          optimizer_kwargs: dict | None = None,
          epoch: int = 100,
          len_epoch: int = 10_000,
          *,
          seed: int = 42,
          ):
    """
    :param save_path: saving path of risk proposal network
    :param inputs_dim: size of the samples
    :param latent_dim: embedding dimension
    :param batch_size: batch size of training
    :param verbose: training status verbosity. 2: show plot and print results. 1: print results, 0: silence
    :param optimizer_class: type of optimizer e.g., adam, adabelief, adamw. default is adablief, with
    lr 3e-4, 1e-4 decay
    :param optimizer_kwargs: kwargs for optimizer, e.g., learning rate
    :param epoch: the number of epoch
    :param len_epoch: length of the epoch
    :param seed: random seed for risk sampler
    :return:
    """
    assert inputs_dim % 2 == 0
    sample_size = inputs_dim // 2
    if optimizer_kwargs is None:
        optimizer_kwargs = {"learning_rate": 3e-4}
    sampler = Sampler(batch_size=batch_size, sample_size=sample_size, seed=seed + 1)
    rngs = Rngs(seed)

    model = RiskProposalNetwork(inputs_dim, latent_dim, rngs=rngs)
    opt = nnx.Optimizer(model, optimizer_class(**optimizer_kwargs))

    metric = nnx.MultiMetric(q_loss=nnx.metrics.Average('q_loss'),
                             aae_loss=nnx.metrics.Average("aae_loss"),
                             cut_off=nnx.metrics.Average('cut_off'))

    graph, state = nnx.split((model, opt, metric))
    range_cls = trange if verbose >= 1 else range

    for _ in range(epoch):
        for _ in range_cls(len_epoch):
            x, y = sampler.sample()

            taus = jax.random.uniform(rngs(), shape=(x.shape[0], x.shape[-1]))
            state, z_fake = update_fn(graph, state, x, y, taus)

        model, opt, metric = nnx.merge(graph, state)
        if verbose >= 1:
            print(metric.compute())
        metric.reset()
        _, state = nnx.split((model, opt, metric))
        if verbose == 2:
            plot(model, 50)
        model.save(save_path, state)


if __name__ == '__main__':
    Fire(train)



