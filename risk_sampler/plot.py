import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
import utils

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

plt.rcParams.update({
    "text.usetex": True,
})


CMAP = 'inferno'


class Wang(object):

    def __call__(self, taus):
        batch_size = taus.shape[0]

        eta = jnp.arange(-1, 1, 2/100)
        eta = eta.reshape(taus.shape[0], 1)

        return jax.scipy.stats.norm.cdf(
            (jax.scipy.stats.norm.ppf(taus).reshape(batch_size, -1) + eta)).reshape(batch_size, -1), eta

    @staticmethod
    def plot(model, save=False):
        wang = Wang()
        taus = jnp.arange(0, 1, 1/ model.n_quantiles).reshape(1, -1)
        taus = jnp.repeat(taus, axis=0, repeats=100)
        y, eta = wang(taus)

        y_hat = jax.lax.stop_gradient(model.decode(*model.encode(taus, y), taus))

        color_map = cm.get_cmap(CMAP)
        max_ = np.max(eta)
        min_ = np.min(eta)
        norm = mpl.colors.Normalize(vmin=min_ - 0.5, vmax=max_ + 0.6)

        color_mapping = cm.ScalarMappable(norm=norm, cmap=color_map)

        fig, axes = plt.subplots(nrows=1, ncols=2)
        plt.subplots_adjust(wspace=0.3)
        fig.text(0.77, 0.25, "probs")
        fig.text(0.01, 0.70, "icdf", rotation='vertical')
        for i in range(100):
            axes[0].plot(taus[i], y[i], c=color_mapping.to_rgba(eta[i]))
        axes[0].set_title('Wang')
        axes[0].set_ylim((0, 1))

        for i in range(100):
            axes[1].plot(taus[i], y_hat[i], c=color_mapping.to_rgba(eta[i]))
        axes[1].set_ylim((0, 1))

        plt.subplots_adjust(wspace=0.3)
        cbar = fig.colorbar(color_mapping, ax=axes.ravel().tolist(), location='bottom')
        fig.text(0.48, 0.05, "neutral", )
        fig.text(0.8, 0.05, "seeking", )
        fig.text(0.1, 0.05, "sensitive", )
        axes[1].set_title("Reconstructed")

        if save:
            plt.savefig("Wang.pdf", format='pdf')

        plt.show()


class CPW(object):

    def __call__(self, taus):
        batch_size = taus.shape[0]
        eta = jnp.arange(1/100, 1+1/100, 1/100)
        eta = eta.reshape(taus.shape[0], 1)
        return jnp.power(taus, eta)/jnp.power(jnp.power(taus, eta) + jnp.power(1. - taus, eta), (1/eta)), eta

    @staticmethod
    def plot(model, save=False):
        cpw = CPW()
        taus = jnp.arange(0, 1, 1 / model.n_quantiles).reshape(1, -1)
        taus = jnp.repeat(taus, axis=0, repeats=100)
        y, eta = cpw(taus)

        y_hat = jax.lax.stop_gradient(model.decode(* model.encode(taus, y), taus))
        color_map = cm.get_cmap(CMAP)
        max_ = np.max(eta)
        min_ = np.min(eta)
        norm = mpl.colors.Normalize(vmin=min_ - 0.01, vmax=max_ + 0.01)
        color_mapping = cm.ScalarMappable(norm=norm, cmap=color_map)

        fig, axes = plt.subplots(nrows=1, ncols=2)

        fig.text(0.77, 0.25, "probs")
        fig.text(0.02, 0.77, "icdf", rotation='vertical')
        for i in range(100):
            axes[0].plot(taus[i], y[i], c=color_mapping.to_rgba(eta[i]))
        axes[0].set_title('CPW')
        axes[0].set_ylim((0, 1))
        for i in range(100):
            axes[1].plot(taus[i], y_hat[i], c=color_mapping.to_rgba(eta[i]))
        axes[1].set_ylim((0, 1))
        cbar = fig.colorbar(color_mapping, ax=axes.ravel().tolist(), location='bottom')
        fig.text(0.48, 0.05, "CPW-$\eta$", )

        axes[1].set_title("Reconstructed")
        if save:
            plt.savefig("CPW.pdf", format='pdf')

        plt.show()


class Power(object):
    def __call__(self, taus):
        batch_size = taus.shape[0]
        eta = jnp.arange(-10, 10, 2/10)

        eta = eta.reshape(taus.shape[0], 1)

        return jnp.where(eta <= 0,
                         1. - jnp.power((1. - taus), 1/(1 + jnp.abs(eta))),
               jnp.power(taus, (1/(1 + jnp.abs(eta))))), eta

    @staticmethod
    def plot(model, save=False):
        pow = Power()
        taus = jnp.arange(0, 1, 1/model.n_quantiles).reshape(1, -1)
        taus = jnp.repeat(taus, axis=0, repeats=100)
        y, eta = pow(taus)

        y_hat = jax.lax.stop_gradient(model.decode(* model.encode(taus, y), taus))
        color_map = cm.get_cmap(CMAP)
        max_ = np.max(eta)
        min_ = np.min(eta)
        norm = mpl.colors.Normalize(vmin=min_ - 0.5, vmax=max_ + 0.5)

        color_mapping = cm.ScalarMappable(norm=norm, cmap=color_map)

        fig, axes = plt.subplots(nrows=1, ncols=2)
        plt.subplots_adjust(wspace=0.3)
        fig.text(0.77, 0.25, "probs")

        fig.text(0.02, 0.77, "icdf", rotation='vertical')
        for i in range(100):
            axes[0].plot(taus[i], y[i], c=color_mapping.to_rgba(eta[i]))
        axes[0].set_title('Power')
        axes[0].set_ylim((0, 1))

        for i in range(100):
            axes[1].plot(taus[i], y_hat[i], c=color_mapping.to_rgba(eta[i]))
        axes[1].set_ylim((0, 1))
        cbar = fig.colorbar(color_mapping, ax=axes.ravel().tolist(), location='bottom')
        fig.text(0.48, 0.05, "neutral", )
        fig.text(0.8, 0.05, "seeking", )
        fig.text(0.1, 0.05, "sensitive", )
        axes[1].set_title("Reconstructed")

        if save:
            plt.savefig("power.pdf", format='pdf')

        plt.show()

class CVaR(object):
    def __call__(self, taus, min_eta=-1, max_eta=1):
        batch_size = taus.shape[0]
        eta = jnp.arange(min_eta, max_eta, (max_eta - min_eta)/100)
        eta = eta.reshape(taus.shape[0], 1)

        return jnp.where(eta <= 0, (1. - jnp.abs(eta)) * taus, eta + (1. - eta) * taus), eta

    @staticmethod
    def plot(model, min_eta=-1, max_eta=1, save=False):
        cvar = CVaR()
        taus = jnp.arange(0, 1, 1/model.n_quantiles).reshape(1, -1)
        taus = jnp.repeat(taus, axis=0, repeats=100)
        y, eta = cvar(taus, min_eta, max_eta)

        y_hat = jax.lax.stop_gradient(model.decode(* model.encode(taus, y), taus))
        color_map = cm.get_cmap(CMAP)
        max_ = np.max(eta)
        min_ = np.min(eta)
        norm = mpl.colors.Normalize(vmin=min_ - 0.03, vmax=max_ + 0.03)

        color_mapping = cm.ScalarMappable(norm=norm, cmap=color_map)

        fig, axes = plt.subplots(nrows=1, ncols=2)
        plt.subplots_adjust(wspace=0.3)
        fig.text(0.77, 0.25, "probs")

        fig.text(0.02, 0.77, "icdf", rotation='vertical')
        for i in range(100):
            axes[0].plot(taus[i], y[i], c=color_mapping.to_rgba(eta[i]))
        axes[0].set_title('CVaR')
        for i in range(100):
            axes[1].plot(taus[i], y_hat[i], c=color_mapping.to_rgba(eta[i]))

        cbar = fig.colorbar(color_mapping, ax=axes.ravel().tolist(), location='bottom')
        fig.text(0.48, 0.05, "neutral", )
        fig.text(0.8, 0.05, "seeking", )
        fig.text(0.1, 0.05, "sensitive", )
        axes[1].set_title("Reconstructed")
        if save:
            plt.savefig("CVaR.pdf", format='pdf')

        plt.show()

def plot_random(model):
    cvar = CVaR()
    taus = jnp.arange(0, 1, 1/model.n_quantiles).reshape(1, -1)
    taus = jnp.repeat(taus, axis=0, repeats=100)
    y, eta = cvar(taus)

    z_sample, c = model.encode(taus, y)
    key = jax.random.PRNGKey(42)
    key2 = jax.random.PRNGKey(43)
    rand_z = utils.rand_like(key, z_sample)
    seq = jax.random.uniform(key2, shape=(rand_z.shape[0], 1))
    seq = seq.sort(axis=-1)
    y_hat = model.decode(rand_z, seq, taus)

    color_map = cm.get_cmap(CMAP)
    max_ = np.max(eta)
    min_ = np.min(eta)
    norm = mpl.colors.Normalize(vmin=min_ - 0.03, vmax=max_ + 0.03)

    color_mapping = cm.ScalarMappable(norm=norm, cmap=color_map)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    plt.subplots_adjust(wspace=0.3)
    fig.text(0.77, 0.25, "probs")

    fig.text(0.02, 0.77, "icdf", rotation='vertical')
    for i in range(100):
        axes[0].plot(taus[i], y[i], c=color_mapping.to_rgba(eta[i]))
    axes[0].set_title('Random')
    for i in range(100):
        axes[1].plot(taus[i], y_hat[i], c=color_mapping.to_rgba(seq[i]))

    cbar = fig.colorbar(color_mapping, ax=axes.ravel().tolist(), location='bottom')

    axes[1].set_title("Reconstructed")

    plt.show()


def plot_all(model, save=False):
    Wang.plot(model, save=save)
    CPW.plot(model, save=save)
    Power.plot(model, save=save)
    CVaR.plot(model, save=save)
    plot_random(model)

if __name__ == '__main__':
    from risk_sampler.model import RiskProposalNet

    # for i in range(7, 10):
    model = RiskProposalNet().load_param('/home/yoo/risk_rl/risk_sampler/risk_proposal_seed_{}.npz'.format(0))
    plot_all(model, save=True)
    """
    cvar = CVaR()
    taus = jnp.arange(0, 1, 1 / model.n_quantiles).reshape(1, -1)
    taus = jnp.repeat(taus, axis=0, repeats=100)
    y, eta = cvar(taus, -1, 0)
    z = model.encode(taus, y)
    print(z.mean())
    tsne = TSNE()
    key = jax.random.PRNGKey(0)
    rand = jax.random.uniform(key=key, shape=z.shape)
    transforms = tsne.fit_transform(np.concatenate((rand, z), axis=0))
    uniform = transforms[:len(rand), :]
    z = transforms[len(rand):, :]
    plt.scatter(uniform[:, 0], uniform[:, 1], color='red')
    plt.scatter(z[:, 0], z[:, 1], color='blue')
    plt.show()
    """
    # CVaR.plot(model, max_eta=-0.9)
