from functools import partial
import gym

from utils.optimize import soft_update

from rl_agents.sac import SAC
from rl_agents.rcdsac.policy import RCDSACCritic, RCDSACActor
from common_model import tanh_normal_reparamterization, get_actions_logprob


import numpy as np

import haiku as hk
import optax
from utils.optimize import optimize
from typing import Optional, Callable
from rl_agents.risk_models import *


class RCDSAC(SAC):
    name = "SAC"
    risk_types = {"cvar": sample_cvar,
                  "general_cvar": sample_cvar_general,
                  "general_pow": sample_power_general,
                  "cpw": cpw,
                  "wang": wang,
                  "power": sample_power}

    def __init__(self,
                 env: gym.Env,
                 buffer_size: int = 1000_000,
                 gamma: float = 0.99,
                 batch_size: int = 256,
                 warmup_steps: int = 2000,
                 seed: int = 0,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 lr_ent: float = 3e-4,
                 soft_update_coef: float = 5e-3,
                 target_entropy: Optional[float] = None,
                 actor_fn: Callable = None,
                 critic_fn: Callable = None,
                 drop_per_net: int = 2,
                 wandb: bool = False,
                 risk_type: str = 'cvar',
                 min_risk_param: float = 0.,
                 max_risk_param: float = 1.,
                 ):

        self.rng = hk.PRNGSequence(seed)
        self.env = env
        n_quantiles = 32
        self.n_quantiles = n_quantiles
        if actor_fn is None:
            def actor_fn(risk_param, obs):
                return RCDSACActor(self.env.action_space.shape[0])(risk_param, obs)

            obs_placeholder, a_placeholder = self.make_placeholder()
            risk_param_placeholder = jnp.ones((1, 1))
            self.actor = hk.without_apply_rng(hk.transform(actor_fn))
            self.param_actor = self.actor.init(next(self.rng), risk_param_placeholder, obs_placeholder)

        if critic_fn is None:

            def critic_fn(risk_param, obs, actions, taus):
                return RCDSACCritic()(risk_param, obs, actions, taus)

            obs_placeholder, a_placeholder = self.make_placeholder()
            risk_param_placeholder = jnp.ones((1, 1))
            quantile_placeholder = jnp.ones((1, self.n_quantiles))
            self.critic = hk.without_apply_rng(hk.transform(critic_fn))
            self.param_critic = self.param_critic_target = self.critic.init(next(self.rng),
                                                                            risk_param_placeholder,
                                                                            obs_placeholder,
                                                                            a_placeholder,
                                                                            quantile_placeholder)

        super().__init__(env,
                         buffer_size=buffer_size,
                         gamma=gamma,
                         batch_size=batch_size,
                         warmup_steps=warmup_steps,
                         seed=seed,
                         target_entropy=target_entropy,
                         actor_fn=actor_fn,
                         critic_fn=critic_fn,
                         wandb=wandb)

        obs_placeholder, a_placeholder = self.make_placeholder()
        risk_param_placeholder = jnp.ones((1, 1))
        quantile_placeholder = jnp.ones((1, self.n_quantiles))

        self.actor = hk.without_apply_rng(hk.transform(actor_fn))
        self.param_actor = self.actor.init(next(self.rng), risk_param_placeholder, obs_placeholder)

        self.critic = hk.without_apply_rng(hk.transform(critic_fn))
        self.param_critic = self.param_critic_target = self.critic.init(next(self.rng),
                                                                        risk_param_placeholder,
                                                                        obs_placeholder,
                                                                        a_placeholder,
                                                                        quantile_placeholder)

        opt_init, self.opt_actor = optax.adabelief(learning_rate=lr_actor)
        self.opt_actor_state = opt_init(self.param_actor)

        opt_init, self.opt_critic = optax.adabelief(learning_rate=lr_critic)
        self.opt_critic_state = opt_init(self.param_critic)

        opt_init, self.opt_ent = optax.adabelief(learning_rate=lr_ent)
        self.opt_ent_state = opt_init(self.log_ent_coef)
        self._n_updates = 0
        self.soft_update_coef = soft_update_coef
        self.drop_per_net = drop_per_net

        self.min_risk_param, self.max_risk_param = min_risk_param, max_risk_param
        try:
            self.risk_model = RCDSAC.risk_types[risk_type]
        except KeyError:
            raise NotImplementedError
        self.current_risk = self.set_alpha()

    def set_alpha(self):
        return self.sample_alpha(key=next(self.rng), batch_size=1)

    @partial(jax.jit, static_argnums=0)
    def normalize(self, alpha):
        scale = self.max_risk_param - self.min_risk_param
        return (alpha - self.min_risk_param)/scale

    def sample_alpha(self,
                     key: jax.random.PRNGKey,
                     batch_size: int):
        scale = self.max_risk_param - self.min_risk_param
        uniform = jax.random.uniform(key=key, shape=(batch_size, 1))
        alpha = scale * uniform + self.min_risk_param
        return alpha

    def predict(self, observations: jnp.ndarray, state=None, *args, **kwargs) -> np.ndarray:
        # policy predict and post process to be numpy
        observations = observations[None]
        if state is None:
            state = self.sample_alpha(key=next(self.rng), batch_size=1)
        else:
            state = jnp.asarray([state], dtype=jnp.float32)[None]
        state = self.normalize(state)
        return np.asarray(self._predict(self.param_actor, state, observations, next(self.rng)))[0]

    def explore(self, observations, state=None, *args, **kwargs) -> np.ndarray:
        return self.predict(observations, state, *args, **kwargs)

    @partial(jax.jit, static_argnums=0)
    def _predict(self, param_actor, risk_param, observations, key):
        return tanh_normal_reparamterization(*self.actor.apply(param_actor, risk_param, observations), key)

    @partial(jax.jit, static_argnums=0)
    def actor_loss(self,
                   param_actor: hk.Params,
                   param_critic: hk.Params,
                   obs: jnp.ndarray,
                   taus: jnp.ndarray,
                   ent_coef: jnp.ndarray,
                   alpha: jnp.ndarray,
                   key: jax.random.PRNGKey
                   ):

        mu, logstd = self.actor.apply(param_actor, risk_param=alpha, obs=obs)
        actions, log_pi = get_actions_logprob(mu, logstd, key)
        qf = self.critic.apply(param_critic,
                               risk_param=alpha, obs=obs,
                               actions=actions, taus=taus).mean(axis=-1)
        qf = jnp.min(qf, axis=-1, keepdims=True)
        return (ent_coef * log_pi - qf).mean(), log_pi

    @staticmethod
    @jax.jit
    def quantile_loss(y: jnp.ndarray,
                      x: jnp.ndarray,
                      taus: jnp.ndarray) -> jnp.ndarray:
        pairwise_delta = y[:, None, :] - x[:, :, None]
        abs_pairwise_delta = jnp.abs(pairwise_delta)
        huber = jnp.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta ** 2 * 0.5)
        loss = jnp.abs(taus[..., None] - jax.lax.stop_gradient(pairwise_delta < 0)) * huber
        return loss

    @partial(jax.jit, static_argnums=0)
    def critic_loss(self,
                    param_critic: hk.Params,
                    param_critic_target: hk.Params,
                    param_actor: hk.Params,
                    obs: jnp.ndarray,
                    actions: jnp.ndarray,
                    rewards: jnp.ndarray,
                    dones: jnp.ndarray,
                    next_obs: jnp.ndarray,
                    taus: jnp.ndarray,
                    next_taus: jnp.ndarray,
                    ent_coef: jnp.ndarray,
                    alpha: jnp.ndarray,
                    key: jax.random.PRNGKey
                    ):

        key1, key2 = jax.random.split(key)
        placeholder = jnp.zeros(shape=(rewards.shape[0], self.n_quantiles))
        _, current_taus, _ = self.sample_taus(key1, placeholder)
        next_presume_taus, next_taus, _ = self.sample_taus(key2, placeholder)

        target_qf = self.compute_target_qf(param_critic_target,
                                           param_actor,
                                           next_obs=next_obs,
                                           next_taus=next_taus,
                                           rewards=rewards,
                                           dones=dones,
                                           ent_coef=ent_coef,
                                           alpha=alpha,
                                           key=key)

        current_qf = self.critic.apply(param_critic, alpha, obs, actions, taus)
        loss = jnp.stack([self.quantile_loss(target_qf, current_qf[:, i, :],  taus).mean(axis=(-1, -2))
                         for i in range(2)], axis=1).mean()
        return loss, None

    @partial(jax.jit, static_argnums=0)
    def compute_target_qf(self,
                          param_critic_target: hk.Params,
                          param_actor: hk.Params,
                          next_obs: jnp.ndarray,
                          next_taus: jnp.ndarray,
                          rewards: jnp.ndarray,
                          dones: jnp.ndarray,
                          ent_coef: jnp.ndarray,
                          alpha: jnp.ndarray,
                          key: jax.random.PRNGKey,
                          ):
        mu, logstd = self.actor.apply(param_actor, risk_param=alpha, obs=next_obs)
        next_actions, next_log_pi = get_actions_logprob(mu, logstd, key)
        next_qf = self.critic.apply(param_critic_target, risk_param=alpha,
                                    obs=next_obs, actions=next_actions, taus=next_taus)

        next_qf = next_qf.reshape(next_qf.shape[0], -1)
        next_qf = jnp.sort(next_qf, axis=-1)
        if self.drop_per_net > 0:
            next_qf = next_qf[:, self.drop_per_net:-self.drop_per_net]
        next_qf = next_qf - ent_coef * next_log_pi
        return jax.lax.stop_gradient(rewards + self.gamma * (1. - dones) * next_qf)

    @partial(jax.jit, static_argnums=0)
    def sample_taus(self, key, placeholder):
        presume_tau = jax.random.uniform(key, placeholder.shape) + 0.1
        presume_tau = presume_tau / presume_tau.sum(axis=-1, keepdims=True)
        tau = jnp.cumsum(presume_tau, axis=-1)
        tau_hat = jnp.zeros_like(tau)
        tau_hat = tau_hat.at[:, 0:1].set(tau[:, 0:1] / 2)
        tau_hat = tau_hat.at[:, 1:].set( (tau[:, 1:] + tau[:, :-1])/2)
        return jax.lax.stop_gradient(tau), jax.lax.stop_gradient(tau_hat), jax.lax.stop_gradient(tau_hat)

    @partial(jax.jit, static_argnums=0)
    def ent_coef_loss(self,
                      log_ent_coef,
                      current_log_pi
                      ):
        return (-log_ent_coef * jax.lax.stop_gradient(current_log_pi + self.target_entropy)).mean(), None

    def train_step(self):
        obs, actions, rewards, dones, next_obs = self.buffer.sample(self.batch_size)
        alpha = self.sample_alpha(key=next(self.rng), batch_size=self.batch_size)

        ent_coef = jnp.exp(self.log_ent_coef)
        taus = jax.random.uniform(next(self.rng), shape=(self.batch_size, self.n_quantiles))
        next_taus = jax.random.uniform(next(self.rng), shape=(self.batch_size, self.n_quantiles))
        normalized_alpha = self.normalize(alpha)
        self.opt_critic_state, self.param_critic, qf_loss, _ = optimize(
            self.critic_loss,
            self.opt_critic,
            self.opt_critic_state,
            params_to_update=self.param_critic,
            param_critic_target=self.param_critic_target,
            param_actor=self.param_actor,
            obs=obs, actions=actions,
            rewards=rewards,
            dones=dones,
            next_obs=next_obs,
            taus=taus,
            next_taus=next_taus,
            ent_coef=ent_coef,
            alpha=normalized_alpha, key=next(self.rng))

        placeholder = jnp.zeros(shape=(rewards.shape[0], self.n_quantiles))
        taus, _, _ = self.sample_taus(key=next(self.rng), placeholder=placeholder)

        self.opt_actor_state, self.param_actor, actor_loss, log_pi = optimize(
            self.actor_loss,
            self.opt_actor,
            self.opt_actor_state,
            self.param_actor,
            param_critic=self.param_critic,
            obs=obs, taus=taus, ent_coef=ent_coef, alpha=normalized_alpha, key=next(self.rng))

        self.opt_ent_state, self.log_ent_coef, ent_coef_loss, _ = optimize(
            self.ent_coef_loss,
            self.opt_ent,
            self.opt_ent_state,
            self.log_ent_coef,
            log_pi
        )

        self.logger.record(key='train/pi_loss', value=actor_loss.item())
        self.logger.record(key='train/qf_loss', value=qf_loss.item())
        self.logger.record(key='train/ent_coef_loss', value=ent_coef_loss.item())
        self.logger.record(key='etc/current_ent_coef', value=ent_coef.item())

        self.param_critic_target = soft_update(self.param_critic_target, self.param_critic, self.soft_update_coef)

    def make_placeholder(self):
        a = self.env.action_space.sample()
        s = self.env.observation_space.sample()
        return jnp.asarray(s[None]), jnp.asarray(a[None])

    def save(self, path):
        params = {"param_actor": self.param_actor,
                  "param_critic": self.param_critic
                  }
        return np.savez(path, **params)

    def load(self, path):
        params = np.load(path, allow_pickle=True)

        self.param_actor = params["param_actor"].item()
        self.param_critic = params["param_critic"].item()
