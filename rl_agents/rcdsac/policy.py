import haiku as hk
import jax.numpy as jnp
from typing import Sequence
from common_model import MLP


class RawCosines(hk.Module):
    def __init__(self, features_dim: int, n_cos: int = 128):
        super().__init__()
        self.features_dim = features_dim
        self.n_cos = n_cos

        self.cosines = jnp.arange(1, self.n_cos + 1, dtype=jnp.float32) * jnp.pi
        self.cosines = self.cosines.reshape(1, 1, -1)

    def __call__(self, taus):
        taus = jnp.expand_dims(taus, axis=-1)
        cosines = jnp.cos(taus * self.cosines)
        return cosines


class RCDSACQf(hk.Module):
    def __init__(self,
                 z_dim: int = 256,
                 net_arch: Sequence[int] = (256, 256),
                 n_cos: int = 256
                 ):
        super().__init__()
        net_arch = list(net_arch)
        self.linear = hk.Linear(z_dim)
        self.feature_embedding = MLP(output_dim=z_dim,
                                     net_arch=net_arch)

        self.cosine_embedding = RawCosines(
            features_dim=z_dim,
            n_cos=n_cos,
        )

        self.outputs = MLP(
            net_arch=net_arch,
            output_dim=1,
        )

    def __call__(self, risk_param, obs, actions, taus):
        feature = self.feature_embedding(jnp.concatenate((obs, actions), axis=-1))
        # (batch, n_taus, n_cosines)
        taus = self.cosine_embedding(taus)
        # (batch, 1, n_cosines)
        risk_param = self.cosine_embedding(risk_param)
        # (batch, n_taus, n_cosines)
        risk_param = jnp.repeat(risk_param, axis=1, repeats=taus.shape[1])
        # (batch, n_taus, 2 * n_cosines)
        taus = jnp.concatenate((taus, risk_param), axis=-1)
        # (batch, n_taus, features_dim)
        taus = self.linear(taus)
        qfs = self.outputs(feature[:, None, :] * taus).squeeze(-1)
        return qfs


class RCDSACCritic(hk.Module):
    def __init__(self,
                 z_dim: int = 256,
                 net_arch: Sequence[int] = (256, 256),
                 n_cos: int = 256
                 ):
        super().__init__()
        self.kwargs = {
            "z_dim": z_dim,
            "net_arch": net_arch,
            "n_cos": n_cos
        }

    def __call__(self, risk_params, obs, actions, taus):
        return jnp.stack([RCDSACQf(**self.kwargs)(risk_params, obs, actions, taus)], axis=1)


class RCDSACActor(hk.Module):
    def __init__(self,
                 actions_dim,
                 z_dim: int = 256,
                 net_arch: Sequence[int] = (256, 256),
                 n_cos: int = 256
                 ):
        super().__init__()
        self.actions_dim = actions_dim
        self.feature_embedding = MLP(output_dim=z_dim,
                                     net_arch=net_arch)

        self.cosine_embedding = RawCosines(
            features_dim=z_dim,
            n_cos=n_cos,
        )

        self.outputs = MLP(
            net_arch=net_arch,
            output_dim= actions_dim,
        )
        self.linear = hk.Linear(z_dim)

    def __call__(self, risk_param, obs):
        feature = self.feature_embedding(obs)
        # (batch size, 1, n_cosines)
        cosines = self.cosine_embedding(risk_param)
        # (batch_size, 1, features_dim)
        # (batch_size, features_dim)
        cosine_feature = self.linear(cosines).squeeze(1)
        outs = self.outputs(jnp.concatenate((cosine_feature, feature), axis=-1))
        mu = hk.Linear(self.actions_dim)(outs)
        logstd = hk.Linear(self.actions_dim)(outs)
        logstd = jnp.clip(logstd, -20, 5)
        return mu, logstd

