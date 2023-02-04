import haiku as hk
import jax
import jax.numpy as jnp
import jax.nn as nn
from typing import Sequence
from common_model.commons import MLP, PowerIQNHead, IQNHead


class RiskConditionedActor(hk.Module):
    def __init__(self,
                 actions_dim: int,
                 net_arch: Sequence[int] = (256, 256)
                 ):
        super().__init__()
        self.layers = MLP(output_dim=actions_dim,
                          net_arch=net_arch,
                          activation_fn=nn.relu,
                          squashed_output=True,
                          )
        self.action_dim = actions_dim

    def __call__(self, risk_emb, obs):

        action = self.layers(jnp.concatenate((risk_emb, obs), axis=-1))
        return action


class RiskConditionedCritic(hk.Module):
    def __init__(self,
                 actions_dim: int,
                 net_arch: Sequence[int] = (256, 256),
                 n_critics: int = 2,
                 ablation: bool = False
                 ):
        super().__init__()
        self.actions_dim = actions_dim
        self.net_arch = net_arch
        self.n_critics = n_critics
        self.ablation = ablation

    def __call__(self, risk_emb, obs, actions, taus):
        def qf(risk_emb, obs, actions, taus):
            cat = MLP(256, self.net_arch)(jnp.concatenate((risk_emb, obs, actions), axis=-1))
            if self.ablation:
                print("cosine head")
                head = IQNHead(z_dim=256)
            else:
                print("monotone head")
                head = PowerIQNHead(z_dim=256)
            return head(cat, taus)
        return jnp.stack([qf(risk_emb, obs, actions, taus) for _ in range(self.n_critics)], axis=-2)

