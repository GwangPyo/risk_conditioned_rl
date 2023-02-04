from functools import partial

from rl_agents.td3 import TD3
from utils.optimize import soft_update
from rl_agents.spec_td3.policy import RiskConditionedActor, RiskConditionedCritic
from risk_sampler.model import RiskProposalNet

import numpy as np

import haiku as hk
import jax.numpy as jnp
import jax
import optax
from utils.optimize import optimize
import gym


@jax.jit
def soft_update(
    target_params: hk.Params,
    online_params: hk.Params,
    tau: float,
) -> hk.Params:
    """
    Update target network using Polyak-Ruppert Averaging.
    """
    return optax.incremental_update(new_tensors=online_params, old_tensors=target_params,
                                    step_size=tau)


class SpecTD3(TD3):
    def __init__(self,
                 env: gym.Env,
                 buffer_size: int = 1000_000,
                 gamma: float = 0.99,
                 batch_size: int = 256,
                 warmup_steps: int = 2000,
                 seed: int = 0,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 delay: int = 2,
                 soft_update_coef: float = 5e-3,
                 target_noise: float = 0.3,
                 target_noise_clip: float = 0.5,
                 risk_sample_coef: float = 0.6,
                 drop_per_net: int = 2,
                 actor_learning_starts: int = 500,
                 ablation=False,
                 wandb=False,
                 n_critics: int = 2,
                 ):

        self.rng = hk.PRNGSequence(seed)
        self.env = env
        z_dim = 256
        n_quantiles = 32
        self.z_dim, self.n_quantiles = z_dim, n_quantiles
        self.risk_net = RiskProposalNet(z_dim, n_quantiles).load_param('/home/yoo/'
                                                                       'risk_rl/risk_sampler/'
                                                                       'risk_proposal_seed_{}.npz'.format(seed))

        self.encoder, self.decoder = self.risk_net.encoder, self.risk_net.decoder
        self.param_risk = self.risk_net.param_generator
        self.n_critics = n_critics

        def actor_fn(z, obs):
            return RiskConditionedActor(self.env.action_space.shape[0])(z, obs)

        def critic_fn(z, obs, action, taus):
            return RiskConditionedCritic(z_dim, n_critics=self.n_critics,
                                         ablation=ablation,
                                         )(z, obs, action, taus)

        obs_placeholder, a_placeholder = self.make_placeholder()
        z_placeholder, quantile_placeholder = jnp.ones((1, z_dim + 1)), jnp.ones((1, self.n_quantiles))

        self.actor = hk.without_apply_rng(hk.transform(actor_fn))
        self.param_actor = self.param_actor_target = self.actor.init(next(self.rng), z_placeholder, obs_placeholder)

        self.critic = hk.without_apply_rng(hk.transform(critic_fn))
        self.param_critic = self.param_critic_target = self.critic.init(next(self.rng),
                                                                        z_placeholder,
                                                                        obs_placeholder,
                                                                        a_placeholder,
                                                                        quantile_placeholder)

        super().__init__(env,
                         buffer_size=buffer_size,
                         gamma=gamma,
                         batch_size=batch_size,
                         warmup_steps=warmup_steps,
                         seed=seed,
                         actor_fn=actor_fn,
                         critic_fn=critic_fn,
                         wandb=wandb
                         )


        opt_init, self.opt_actor = optax.adabelief(lr_actor)

        self.opt_actor_state = opt_init(self.param_actor)

        opt_init, self.opt_critic = optax.adabelief(lr_critic)

        self.opt_critic_state = opt_init(self.param_critic)

        self.risk_sample_coef = risk_sample_coef
        self.delay = delay
        self.soft_update_coef = soft_update_coef
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self._n_updates = 0
        self.current_risk = self.set_z()
        self.drop_per_net = drop_per_net
        self.actor_learning_starts = actor_learning_starts
        self.ablation = ablation
        self.actions_dim = len(env.action_space.shape)

    def explore(self, observations, state=None, *args, **kwargs) -> np.ndarray:
        predictions = self.predict(observations, state)
        return predictions

    def set_z(self):
        z = jax.random.uniform(key=next(self.rng), shape=(1, self.z_dim))
        c = self.risk_sample_coef * jax.random.uniform(key=next(self.rng), shape=(1, 1))
        return jnp.concatenate((z, c), axis=-1)

    def predict(self, observations, alpha=None) -> np.ndarray:
        # policy predict and post process to be numpy
        observations = observations[None]
        if alpha is None:
            z = self.current_risk

        elif isinstance(alpha, float):
            taus = jnp.arange(0, 1, 1 / self.n_quantiles)[None]
            z = jnp.concatenate(self.encoder(self.param_risk, taus, alpha * taus), axis=-1)
        else:
            z = alpha

        actions = self._predict(self.param_actor,
                                observations=observations,
                                z=z)
        # , key=next(self.rng))

        return np.asarray(actions)[0]

    def done_callback(self):
        self.current_risk = self.set_z()

    @partial(jax.jit, static_argnums=0)
    def _predict(self, param_actor, observations, z):
        return self.actor.apply(param_actor, z, observations)

    @partial(jax.jit, static_argnums=0)
    def actor_loss(self,
                         param_actor: hk.Params,
                         param_critic: hk.Params,
                         param_risk: hk.Params,
                         obs: jnp.ndarray,
                         z: jnp.ndarray,
                         key: jax.random.PRNGKey
                         ):
        actions = self.actor.apply(param_actor, z, obs)
        taus = jax.random.uniform(key, shape=(z.shape[0], self.n_quantiles))
        taus = taus.sort(axis=-1)
        phi = jax.lax.stop_gradient(self.decoder(param_risk, z[:, :-1], z[:, [-1]], taus))
        qf_1 = self.critic.apply(param_critic, z, obs, actions, phi)[:, 0, :]
        return -qf_1.mean(), phi

    @partial(jax.jit, static_argnums=0)
    def quantile_loss(self,
                      y: jnp.ndarray,
                      x: jnp.ndarray,
                      taus: jnp.ndarray) -> jnp.ndarray:
        pairwise_delta = y[:, None, :] - x[:, :, None]
        abs_pairwise_delta = jnp.abs(pairwise_delta)
        huber = jnp.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta ** 2 * 0.5)
        loss = jnp.abs(taus[..., None] - jax.lax.stop_gradient(pairwise_delta < 0)) * huber
        return loss

    @partial(jax.jit, static_argnums=0)
    def sample_taus(self, key, placeholder):
        presume_tau = jax.random.uniform(key, placeholder.shape) + 0.01
        presume_tau = presume_tau / presume_tau.sum(axis=-1, keepdims=True)
        tau = jnp.cumsum(presume_tau, axis=-1)
        tau_hat = jnp.zeros_like(tau)
        tau_hat = tau_hat.at[:, 0:1].set(tau[:, 0:1] / 2)
        tau_hat = tau_hat.at[:, 1:].set((tau[:, 1:] + tau[:, :-1])/2)
        return jax.lax.stop_gradient(tau), jax.lax.stop_gradient(tau_hat), jax.lax.stop_gradient(presume_tau)

    @partial(jax.jit, static_argnums=0)
    def critic_loss(self,
                    param_critic: hk.Params,
                    param_critic_target: hk.Params,
                    param_actor_target: hk.Params,
                    param_risk: hk.Params,
                    obs: jnp.ndarray,
                    z: jnp.ndarray,
                    actions: jnp.ndarray,
                    reward: jnp.ndarray,
                    dones: jnp.ndarray,
                    next_obs: jnp.ndarray,
                    key: jax.random.PRNGKey
                    ):
        key1, key2, key3 = jax.random.split(key, 3)
        current_taus = jax.random.uniform(key2, shape=(z.shape[0], self.n_quantiles)).sort(axis=-1)
        next_taus = jax.random.uniform(key1, shape=(z.shape[0], self.n_quantiles)).sort(axis=-1)
        target_qf = self.compute_target_qf(param_critic_target,
                                           param_actor_target,
                                           next_obs, z, next_taus,
                                           reward, dones, key2)

        current_qf = self.critic.apply(param_critic, z, obs, actions, current_taus)
        loss = jnp.stack([self.quantile_loss(target_qf, current_qf[:, i, :],
                                             current_taus).mean(axis=(-1, -2))
                         for i in range(self.n_critics)], axis=1).mean(axis=-1).mean()

        return loss, None

    @partial(jax.jit, static_argnums=0)
    def compute_target_qf(self,
                          param_critic_target: hk.Params,
                          param_actor_target: hk.Params,
                          next_obs: jnp.ndarray,
                          z: jnp.ndarray,
                          taus_dash: jnp.ndarray,
                          rewards: jnp.ndarray,
                          dones: jnp.ndarray,
                          key: jax.random.PRNGKey,
                          ):
        key1, key2 = jax.random.split(key)
        next_actions = self.actor.apply(param_actor_target, z, next_obs)
        noise = self.target_noise * jax.random.normal(key2, next_actions.shape)
        next_actions = (next_actions + noise.clip(-self.target_noise_clip, self.target_noise_clip)).clip(-1., 1.)
        next_qf = self.critic.apply(param_critic_target, z, next_obs, next_actions, taus_dash)
        next_qf = next_qf.reshape(next_qf.shape[0], -1).sort(axis=-1)
        if self.drop_per_net > 0:
            next_qf = next_qf[..., self.drop_per_net:-self.drop_per_net]
        return jax.lax.stop_gradient(rewards + self.gamma * (1. - dones) * next_qf)

    @partial(jax.jit, static_argnums=0)
    def beta_loss(self, log_beta, risk_premium_loss):
        return (-log_beta * jax.lax.stop_gradient(risk_premium_loss)).mean(), None

    def train_step(self):
        obs, actions, rewards, dones, next_obs = self.buffer.sample(self.batch_size)
        randoms = self.risk_sample_coef * jax.random.uniform(key=next(self.rng), shape=(self.batch_size, 1))
        z = jax.lax.stop_gradient(jax.random.uniform(key=next(self.rng), shape=(self.batch_size, self.z_dim)))
        z = jnp.concatenate((z, randoms), axis=-1)

        self.opt_critic_state, self.param_critic, qf_loss, _ = optimize(
            self.critic_loss,
            self.opt_critic,
            self.opt_critic_state,
            self.param_critic,
            self.param_critic_target,
            self.param_actor_target,
            self.param_risk,
            obs, z, actions, rewards, dones,
            next_obs, next(self.rng))
        self.logger.record(key='train/qf_loss', value=qf_loss.item())

        if self._n_updates % self.delay == 0 and self._n_updates > self.actor_learning_starts:

            self.opt_actor_state, self.param_actor, actor_loss, decoded = optimize(
                self.actor_loss,
                self.opt_actor,
                self.opt_actor_state,
                self.param_actor,
                self.param_critic,
                self.param_risk,
                obs,
                z,
                next(self.rng))
            self.logger.record(key='train/pi_loss', value=actor_loss.item())
            self.param_critic_target = soft_update(self.param_critic_target, self.param_critic, self.soft_update_coef)
            self.param_actor_target = soft_update(self.param_actor_target, self.param_actor, self.soft_update_coef)

    def make_placeholder(self):
        a = self.env.action_space.sample()
        s = self.env.observation_space.sample()
        return jnp.asarray(s[None]), jnp.asarray(a[None])

    def save(self, path):
        params = {"param_actor": self.param_actor,
                  "param_critic": self.param_critic,
                  "param_risk": self.param_risk
                  }
        return np.savez(path, **params)

    def load(self, path):
        params = np.load(path, allow_pickle=True)
        self.param_actor = params["param_actor"].item()
        self.param_critic = params["param_critic"].item()
        # print(type(self.param_risk))
        # self.param_risk = params['param_risk']
        # print(self.param_risk)
