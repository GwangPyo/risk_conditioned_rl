import pickle

import jax.numpy as jnp
import jax.random
import numpy as np
import optax
from flax import nnx
import distrax
from typing import Callable, Sequence
from base.jax_layers import (create_mlp, FourierFeatureNetwork, IQNHead, ConvolutionLayer,
                             ConvCriticFeatureExtractor, imq_kernel)
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim
from risk_sampler.risk_proposal_network import RiskProposalNetwork
from functools import partial
from risk_sensitive_rl.utils import copy_param
from risk_sensitive_rl.ent_coef import EntCoef
from base.losses import quanitle_regression_loss
from typing import Any
from stable_baselines3.common.buffers import ReplayBufferSamples, DictReplayBufferSamples
import gymnasium as gym


def image_normalization(x):
    return x / 127.5 - 1


class Actor(nnx.Module):
    def __init__(self,
                 features_dim: int,
                 action_space: gym.spaces.box,
                 net_arch: Sequence[int] = (256, 256),
                 activation_fn: Callable = nnx.relu,
                 layer_norm: bool = True,
                 risk_measure_dim: int = 256,
                 *,
                 rngs: nnx.Rngs
                 ):
        self.low = action_space.low
        self.high = action_space.high
        self.scale = (self.high - self.low) / 2
        self.shift = (self.high + self.low) / 2

        self.actions_dim = get_action_dim(action_space)
        self.features_dim = features_dim
        self.net_arch = list(net_arch)
        self.activation_fn = activation_fn
        self.layer_norm = layer_norm
        self.risk_measure_dim = risk_measure_dim
        self.rng_collection: nnx.Rngs = rngs

        self.feature_extractor = self.build_feature_extractor()
        self.state_extractor = self.build_state_extractor()
        self.layers = self.action_extractor()

    def build_feature_extractor(self):
        return nnx.Sequential(
            *create_mlp(self.features_dim, 64, net_arch=self.net_arch,
                        activation_fn=self.activation_fn,
                        rngs=self.rng_collection)
        )

    def build_state_extractor(self):
        return nnx.Sequential(
            FourierFeatureNetwork(input_dim=self.risk_measure_dim, output_dim=64, rngs=self.rng_collection),
            *create_mlp(64, 64, net_arch=self.net_arch,
                        activation_fn=self.activation_fn,
                        rngs=self.rng_collection)
        )

    def action_extractor(self):
        return nnx.Sequential(
            *create_mlp(128, self.actions_dim * 2, net_arch=self.net_arch,
                        activation_fn=self.activation_fn, with_bias=False,
                        rngs=self.rng_collection)
        )

    def __call__(self, observations, risk_embedding):
        feature = self.feature_extractor(observations)
        risk = self.state_extractor(risk_embedding)
        mu_log_sigma = self.layers(jnp.concatenate([feature, risk], axis=-1))
        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1, )
        log_sigma = log_sigma.clip(-20., 1.)
        distr = distrax.Normal(loc=mu, scale=jnp.exp(log_sigma))
        chain = distrax.Chain([distrax.Tanh()])
        return distrax.Transformed(distr, bijector=chain)

    def sample_and_log_prob(self, observation, risk_embedding, *, sample_shape=(), seed=None, ):
        if seed is None:
            seed = self.rng_collection()
        sample, log_prob = self(observation, risk_embedding).sample_and_log_prob(seed=seed, sample_shape=sample_shape)

        return sample, log_prob.sum(axis=-1, keepdims=True)

    def mode(self, observations, risk_embedding):
        feature = self.feature_extractor(observations)
        risk = self.state_extractor(risk_embedding)
        mu_log_sigma = self.layers(jnp.concatenate([feature, risk], axis=-1))
        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1, )
        return jnp.tanh(mu)


class ConvActor(Actor):

    def build_feature_extractor(self):
        return nnx.Sequential(
            image_normalization,
            ConvolutionLayer(self.features_dim, out_feature=64, rngs=self.rng_collection))

    def build_state_extractor(self):
        return nnx.Sequential(
            FourierFeatureNetwork(input_dim=self.risk_measure_dim, output_dim=64, rngs=self.rng_collection),
            *create_mlp(64, 64, net_arch=self.net_arch,
                        activation_fn=self.activation_fn,
                        rngs=self.rng_collection)
        )

    def action_extractor(self):
        return nnx.Sequential(
            *create_mlp(128, self.actions_dim * 2, net_arch=self.net_arch,
                        activation_fn=self.activation_fn,
                        rngs=self.rng_collection)
        )


class MultiInputActor(ConvActor):
    def __init__(self,
                 channel: int,
                 features_dim: int,
                 action_space: gym.spaces.box,
                 net_arch: Sequence[int] = (256, 256),
                 activation_fn: Callable = nnx.relu,
                 layer_norm: bool = True,
                 risk_measure_dim: int = 256,
                 *,
                 rngs: nnx.Rngs
                 ):
        self.channel = channel
        super().__init__(
            features_dim,
            action_space,
            net_arch,
            activation_fn,
            layer_norm=layer_norm,
            risk_measure_dim=risk_measure_dim,
            rngs=rngs
        )

    def build_feature_extractor(self):
        return nnx.Sequential(
            image_normalization,
            ConvolutionLayer(self.channel, out_feature=128, rngs=self.rng_collection))

    def build_state_extractor(self):
        return nnx.Sequential(
            FourierFeatureNetwork(input_dim=self.risk_measure_dim + self.features_dim,
                                  output_dim=128, rngs=self.rng_collection),
            *create_mlp(128, 128, net_arch=self.net_arch,
                        activation_fn=self.activation_fn,
                        rngs=self.rng_collection)
        )

    def action_extractor(self):
        return nnx.Sequential(
            *create_mlp(256, self.actions_dim * 2, net_arch=self.net_arch,
                        activation_fn=self.activation_fn,
                        rngs=self.rng_collection)
        )

    def __call__(self, observations: dict[str, jax.Array], risk_embedding):
        feature = self.feature_extractor(observations['image'])
        risk_feature = self.state_extractor(jnp.concatenate([risk_embedding, observations['feature']], axis=-1))
        mu_log_sigma = self.layers(jnp.concatenate([feature, risk_feature], axis=-1))
        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1, )
        log_sigma = log_sigma.clip(-20., 1.)
        distr = distrax.Normal(loc=mu, scale=jnp.exp(log_sigma))
        chain = distrax.Chain([distrax.Tanh()])
        return distrax.Transformed(distr, bijector=chain)

    def mode(self, observations, risk_embedding):
        feature = self.feature_extractor(observations['image'])
        risk_feature = self.state_extractor(jnp.concatenate([risk_embedding, observations['feature']], axis=-1))
        mu_log_sigma = self.layers(jnp.concatenate([feature, risk_feature], axis=-1))
        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1, )
        return jnp.tanh(mu)


class Critic(nnx.Module):
    def __init__(self,
                 features_dim: int,
                 actions_dim: int,
                 net_arch: Sequence[int] = (256, 256),
                 activation_fn: Callable = nnx.relu,
                 layer_norm: bool = True,
                 risk_measure_dim: int = 256,
                 use_fourier: bool = True,
                 *,
                 rngs: nnx.Rngs
                 ):
        self.use_fourier = use_fourier
        self.actions_dim = actions_dim
        self.features_dim = features_dim
        self.net_arch = list(net_arch)
        self.activation_fn = activation_fn
        self.layer_norm = layer_norm
        self.risk_measure_dim = risk_measure_dim

        self.feature_extractor = self.build_feature_extractor(rngs)
        self.state_extractor = self.build_state_extractor(rngs)
        self.layers = nnx.Sequential(
            nnx.Linear(128, 64, rngs=rngs),
            nnx.relu,
            nnx.Linear(64, 64, rngs=rngs)
        )
        self.iqn_head = IQNHead(64, rngs=rngs)

    def build_feature_extractor(self, rngs):
        if self.use_fourier:
            return nnx.Sequential(
                FourierFeatureNetwork(self.features_dim + self.actions_dim, 128, rngs=rngs),
                *create_mlp(128, 64, net_arch=self.net_arch,
                            activation_fn=self.activation_fn,
                            rngs=rngs)
            )
        else:
            return nnx.Sequential(
                *create_mlp(self.features_dim + self.actions_dim, 64, net_arch=self.net_arch,
                            activation_fn=self.activation_fn,
                            rngs=rngs)
            )

    def build_state_extractor(self, rngs):
        return nnx.Sequential(
            FourierFeatureNetwork(input_dim=self.risk_measure_dim, output_dim=64, rngs=rngs),
            *create_mlp(64, 64, net_arch=self.net_arch,
                        activation_fn=self.activation_fn,
                        rngs=rngs)
        )

    def __call__(self, observations, actions, risk_embedding, taus):
        feature = self.feature_extractor(jnp.concatenate([observations, actions], axis=-1))
        risk = self.state_extractor(risk_embedding)
        feature = self.layers(jnp.concatenate([feature, risk], axis=-1))
        return self.iqn_head(feature, taus)

    @classmethod
    def ensemble_critic(cls,
                        features_dim: int,
                        actions_dim: int,
                        n_critics: int = 3,
                        net_arch: Sequence[int] = (256, 256),
                        activation_fn: Callable = nnx.relu,
                        layer_norm: bool = True,
                        risk_measure_dim: int = 256,
                        use_fourier: bool = True,
                        *,
                        rngs: nnx.Rngs
                        ):
        model = nnx.Vmap(cls,
                         in_axes=None, state_axes={...: 0}, out_axes=-1,
                         module_init_args=(features_dim, actions_dim),
                         module_init_kwargs={"net_arch": net_arch, "activation_fn": activation_fn,
                                             "layer_norm": layer_norm, "rngs": rngs,
                                             "risk_measure_dim": risk_measure_dim, "use_fourier": use_fourier},
                         axis_size=n_critics, )
        return model


class ConvEnsembleCritic(nnx.Module):

    def __init__(self,
                 features_dim: int,
                 actions_dim: int,
                 n_critics: int = 3,
                 net_arch: Sequence[int] = (256, 64),
                 activation_fn: Callable = nnx.relu,
                 layer_norm: bool = True,
                 risk_measure_dim: int = 256,
                 *,
                 in_feature_dim: int = 256,
                 rngs: nnx.Rngs
                 ):
        self.in_feature_dim = in_feature_dim
        self.critic = Critic.ensemble_critic(
            features_dim=in_feature_dim,
            n_critics=n_critics,
            actions_dim=actions_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            layer_norm=layer_norm,
            risk_measure_dim=risk_measure_dim, rngs=rngs,
            use_fourier=False,
        )
        self.actions_embedding = nnx.Sequential(*create_mlp(
            actions_dim,
            self.conv_feature,
            net_arch=[64, 64],
            rngs=rngs
        ))

        self.obs = ConvolutionLayer(features_dim, self.conv_feature, rngs=rngs)

    @property
    def conv_feature(self) -> int:
        return self.in_feature_dim

    def __call__(self, observations, actions, risk_embedding, taus):
        observations = image_normalization(observations)
        obs = self.obs(observations)
        return self.critic(obs, actions, risk_embedding, taus)

    def qf_and_recon_loss(self, observations, actions, risk_embedding, taus,
                          next_observations, dones, key):
        keys = jax.random.split(key, 3)
        observations = image_normalization(observations)
        next_observations = image_normalization(next_observations)
        z = self.obs(observations)

        mmd_1 = imq_kernel(z, jax.random.normal(keys[2], shape=z.shape), h_dim=z.shape[-1], ) / z.shape[0]

        z = z * self.actions_embedding(actions)
        wae_loss, recon_loss = self.obs.wae_loss(next_observations, z, keys[1])

        recon_loss = recon_loss * (1 - dones).reshape(-1, )
        recon_loss = recon_loss.sum() / (1 - dones).sum()
        wae_loss = wae_loss.mean() + recon_loss + mmd_1

        return self.critic(z, actions, risk_embedding, taus), wae_loss, recon_loss

    def encode(self, observation):
        observation = image_normalization(observation)
        return self.obs(observation)


class MultiInputCritic(ConvEnsembleCritic):
    def __init__(self,
                 channel_dim: int,
                 actions_dim: int,
                 n_critics: int = 3,
                 net_arch: Sequence[int] = (256, 64),
                 activation_fn: Callable = nnx.relu,
                 layer_norm: bool = True,
                 risk_measure_dim: int = 256,
                 *,
                 in_feature_dim: int,
                 rngs: nnx.Rngs
                 ):
        self.others = nnx.Sequential(FourierFeatureNetwork(in_feature_dim, 256, rngs=rngs),
                                     *create_mlp(256, 256, rngs=rngs),
                                     )
        self.merge = nnx.Sequential(
            FourierFeatureNetwork(512 + actions_dim, 64, rngs=rngs),
            nnx.LayerNorm(64, rngs=rngs),
            jnp.sinc,
            nnx.Linear(64, 256, rngs=rngs)
        )

        self.feature_decoder = nnx.Sequential(*create_mlp(256, in_feature_dim + 1, rngs=rngs))

        super().__init__(channel_dim,
                         actions_dim,
                         n_critics,
                         net_arch,
                         activation_fn=activation_fn,
                         layer_norm=layer_norm,
                         risk_measure_dim=risk_measure_dim,
                         in_feature_dim=256 + 256,
                         rngs=rngs
                         )

    @property
    def conv_feature(self) -> int:
        return 256

    def __call__(self, observations, actions, risk_embedding, taus):
        img_observations = image_normalization(observations['image'])
        obs = self.obs(img_observations)
        feature = self.others(observations['feature'])

        return self.critic(jnp.concatenate([obs, feature], axis=-1),
                           actions, risk_embedding, taus)

    def qf_and_recon_loss(self, observations, actions, risk_embedding,
                          taus, next_observations, reward, dones, key):
        keys = jax.random.split(key, 3)

        img_observations = observations['image']
        img_observations = image_normalization(img_observations)
        next_img_observations = image_normalization(next_observations['image'])
        z = self.obs(img_observations)

        feature = self.others(observations['feature'])
        merged = self.merge(jnp.concatenate([z, feature, actions], axis=-1))
        wae_loss, recon_loss = self.obs.wae_loss(next_img_observations, merged, keys[1])
        # make embeddings to follow the normal
        mmd_1 = imq_kernel(z, jax.random.normal(keys[2], shape=z.shape), h_dim=z.shape[-1], ) / z.shape[0]

        mmd_2 = imq_kernel(feature, jax.random.normal(key, shape=feature.shape), h_dim=feature.shape[-1], ) / \
                feature.shape[0]
        feature_decode = self.feature_decoder(merged)
        reward_prediction = feature_decode[:, -1][..., None]
        feature_prediction = feature_decode[:, :-1]
        # look forward
        recon_reward = optax.huber_loss(reward_prediction, reward)
        recon_feature = optax.huber_loss(feature_prediction, next_observations['feature']).mean(axis=-1)
        recon_loss = recon_loss + recon_feature

        recon_loss = recon_loss * (1 - dones).reshape(-1, )
        recon_loss = recon_loss.sum() / (1 - dones).sum() + recon_reward.mean()
        wae_loss = wae_loss + mmd_2 + mmd_1

        return self.critic(jnp.concatenate([z, feature], axis=-1),
                           actions, risk_embedding, taus), wae_loss, recon_loss

    def encode(self, observation):
        img = self.obs(image_normalization(observation['image']))
        feature = self.others(observation['feature'])
        return jnp.concatenate([img, feature], axis=-1)

    def mmd(self, encoded_observation, actions, key):
        z = self.merge(jnp.concatenate([encoded_observation, actions], axis=-1))
        mmd = imq_kernel(z, jax.random.normal(key, shape=z.shape), h_dim=z.shape[-1], ) / z.shape[0]
        return mmd


class RiskConditionedPolicy(object):
    risk_proposal_network: RiskProposalNetwork
    critic: Any
    opt_critic: nnx.Optimizer
    metric_critic: nnx.Metric
    state_critic: nnx.State
    graph_critic: nnx.GraphDef
    params_target_critic: nnx.Param

    actor: nnx.Module
    opt_actor: nnx.Optimizer
    metric_actor: nnx.Metric
    state_actor: nnx.State
    graph_actor: nnx.GraphDef

    ent_coef: EntCoef
    opt_ent_coef: nnx.Optimizer
    metric_ent_coef: nnx.Metric
    state_ent_coef: nnx.State
    graph_ent_coef: nnx.GraphDef

    graph_rp: nnx.GraphDef
    state_rp: nnx.State

    def __init__(self,
                 observation_space,
                 action_space,
                 risk_proposal_network_path,
                 target_entropy: float,
                 ent_coef_init: float | str = 'auto',
                 gamma: float = 0.995,
                 lr: float = 3e-4,
                 soft_update_ratio: float = 5e-3,
                 *,
                 seed: int = 42
                 ):
        self.soft_update_ratio = soft_update_ratio
        self.observation_space = observation_space
        self.action_space = action_space
        self.seed = seed
        self.rngs = nnx.Rngs(seed)
        self.rp_network_path = risk_proposal_network_path
        self.obs_dim = self.observation_dim()
        self.action_dim = get_action_dim(self.action_space)
        self.gamma = gamma
        self.lr = lr
        self.target_entropy = target_entropy
        self.ent_coef_init = ent_coef_init

    def observation_dim(self):
        return get_flattened_obs_dim(self.observation_space)

    def build(self):
        self.load_risk_proposal_network(self.rp_network_path)
        self.build_critic()
        self.build_actor()
        self.build_ent_coef()

    def predict(self, observation, risk_embedding, *, deterministic: bool = False):
        if deterministic:
            action = self._predict_deterministic(self.graph_actor, self.state_actor, observation, risk_embedding)
            return np.asarray(action)
        else:
            # reassign because of seed problem
            action, self.state_actor = self._predict_rand(self.graph_actor, self.state_actor, observation,
                                                          risk_embedding)
            return np.asarray(action)

    @staticmethod
    @jax.jit
    def _predict_rand(graph_actor, state_actor, observations, z):
        actor, *others = nnx.merge(graph_actor, state_actor)
        action, _ = actor.sample_and_log_prob(observations, z)
        _, state_actor = nnx.split((actor, *others))
        return action, state_actor

    @staticmethod
    @jax.jit
    def _predict_deterministic(graph_actor, state_actor, observations, z):
        actor, _, _ = nnx.merge(graph_actor, state_actor)
        action = actor.mode(observations, z)
        return action

    def build_actor(self):
        self.actor = Actor(self.obs_dim, self.action_space, rngs=self.rngs)
        self.opt_actor = nnx.Optimizer(self.actor, optax.adabelief(self.lr))
        self.metric_actor = nnx.MultiMetric(
            pi_loss=nnx.metrics.Average('pi_loss'),
            ent=nnx.metrics.Average('ent')
        )
        self.graph_actor, self.state_actor = nnx.split((self.actor, self.opt_actor, self.metric_actor))

    def build_critic(self):
        self.critic = Critic.ensemble_critic(self.obs_dim, self.action_dim, rngs=self.rngs, n_critics=3)
        self.opt_critic = nnx.Optimizer(self.critic, optax.adabelief(self.lr))
        self.metric_critic = nnx.MultiMetric(qf_loss=nnx.metrics.Average('qf_loss'))

        self.graph_critic, self.state_critic = nnx.split((self.critic, self.opt_critic, self.metric_critic))
        self.params_target_critic = copy_param(self.critic)

    def build_ent_coef(self, ):
        if self.ent_coef_init == 'auto':
            self.ent_coef_init = np.exp(
                1 - (np.prod(self.action_space.shape))).item()  # (1 -> ent_coef: 1., 2 -> exp(-1) ... )

        self.ent_coef = EntCoef(init=self.ent_coef_init, rngs=self.rngs)
        self.opt_ent_coef = nnx.Optimizer(self.ent_coef, optax.adam(self.lr))
        self.metric_ent_coef = nnx.MultiMetric(ent_loss=nnx.metrics.Average('ent_loss'),
                                               ent_coef=nnx.metrics.Average('ent_coef'))
        self.graph_ent_coef, self.state_ent_coef = nnx.split((self.ent_coef, self.opt_ent_coef, self.metric_ent_coef))

    def load_risk_proposal_network(self, path):
        state = RiskProposalNetwork.load(path)
        model = RiskProposalNetwork(200, 256, rngs=nnx.Rngs(1))
        opt = nnx.Optimizer(model, optax.chain(optax.scale_by_belief(),
                                               optax.add_decayed_weights(1e-4),
                                               optax.scale_by_learning_rate(3e-4)
                                               ))

        metric = nnx.MultiMetric(q_loss=nnx.metrics.Average('q_loss'),
                                 aae_loss=nnx.metrics.Average("aae_loss"),
                                 cut_off=nnx.metrics.Average('cut_off'))

        graph, _ = nnx.split((model, opt, metric))
        self.risk_proposal_network = nnx.merge(graph, state)[0]
        self.graph_rp, self.state_rp = graph, state

    def train_step(self, batch: ReplayBufferSamples):
        observations = batch.observations.cpu().numpy()
        actions = batch.actions.cpu().numpy()
        rewards = batch.rewards.cpu().numpy()
        dones = batch.dones.cpu().numpy()
        next_observation = batch.next_observations.cpu().numpy()

        self.state_critic, self.params_target_critic, self.state_actor, self.state_ent_coef, self.state_rp = self._train_step(
            self.graph_critic, self.state_critic, self.params_target_critic,
            self.graph_actor, self.state_actor,
            self.graph_ent_coef, self.state_ent_coef, self.graph_rp, self.state_rp,
            observations, actions, rewards, dones, next_observation,
            key=self.rngs()
        )

    @partial(jax.jit, static_argnums=(0,))
    def _train_step(self,
                    graph_critic, state_critic, params_target_critic,
                    graph_actor, state_actor,
                    graph_ent_coef, state_ent_coef,
                    graph_rp, state_rp,
                    observations, actions, rewards, dones, next_observations,
                    key,
                    ):
        keys = jax.random.split(key, 3)

        state_critic, state_rp = self.critic_loss(graph_critic, state_critic, params_target_critic,
                                                  graph_actor, state_actor,
                                                  graph_ent_coef, state_ent_coef, graph_rp, state_rp,
                                                  observations, actions, rewards, dones, next_observations,
                                                  key=keys[0],
                                                  )

        state_actor, log_prob, state_rp = self.actor_loss(graph_critic, state_critic,
                                                          graph_actor, state_actor,
                                                          graph_ent_coef, state_ent_coef, graph_rp, state_rp,
                                                          observations, key=keys[1])
        params_target_critic = self.polyak_update(graph_critic, state_critic, params_target_critic,
                                                  soft_update_ratio=self.soft_update_ratio
                                                  )
        state_ent_coef = self.ent_coef_loss(graph_ent_coef, state_ent_coef, log_prob)
        return state_critic, params_target_critic, state_actor, state_ent_coef, state_rp

    @partial(jax.jit, static_argnums=(0,))
    def critic_loss(self,
                    graph_critic, state_critic, params_target_critic,
                    graph_actor, state_actor,
                    graph_ent_coef, state_ent_coef,
                    graph_rp, state_rp,
                    observations, actions, rewards, dones, next_observations,
                    *,
                    key
                    ):
        critic, opt_critic, metric_critic = nnx.merge(graph_critic, state_critic)
        graph, param, *others = nnx.split(critic, nnx.Param, ...)

        target_critic = nnx.merge(graph, params_target_critic, *others)
        keys = jax.random.split(key, 2)
        current_taus = jax.random.uniform(keys[0], shape=(actions.shape[0], 32))
        next_taus = jax.random.uniform(keys[1], shape=(actions.shape[0], 32))
        ent_coef_model, _, _ = nnx.merge(graph_ent_coef, state_ent_coef)
        ent_coef = ent_coef_model()
        risk_proposal_network, *rp_others = nnx.merge(graph_rp, state_rp)
        z, _ = risk_proposal_network.generate(next_taus)
        actor, _, _ = nnx.merge(graph_actor, state_actor)
        actor: Actor
        next_action, next_log_prob = actor.sample_and_log_prob(next_observations, z, )

        next_qf = target_critic(next_observations, next_action, z, next_taus)

        next_qf = next_qf.reshape(next_qf.shape[0], -1).sort(axis=-1)
        next_qf = next_qf[..., :-3] - ent_coef * next_log_prob

        td_target = rewards.reshape(-1, 1) + self.gamma * (1 - dones.reshape(-1, 1)) * next_qf

        def loss_fn(qf):
            current_qf = qf(observations, actions, z, current_taus)

            loss = jax.vmap(quanitle_regression_loss, in_axes=(None, -1, None), out_axes=-1)(td_target,
                                                                                             current_qf, current_taus)
            loss = loss.sum(axis=(-1, -2)).mean()
            return loss

        qf_loss, grad = nnx.value_and_grad(loss_fn)(critic)
        opt_critic.update(grad)
        metric_critic.update(qf_loss=qf_loss)
        _, new_state = nnx.split((critic, opt_critic, metric_critic))
        _, rp_state = nnx.split((risk_proposal_network, *rp_others))
        return new_state, rp_state

    @partial(jax.jit, static_argnums=(0,))
    def actor_loss(self,
                   graph_critic, state_critic,
                   graph_actor, state_actor,
                   graph_ent_coef, state_ent_coef,
                   graph_rp, state_rp,
                   observations,
                   *,
                   key
                   ):
        critic, _, _ = nnx.merge(graph_critic, state_critic)
        actor, opt_actor, metric_actor = nnx.merge(graph_actor, state_actor)
        ent_coef_model, _, _ = nnx.merge(graph_ent_coef, state_ent_coef)
        ent_coef = ent_coef_model()
        taus = jax.random.uniform(key, shape=(observations.shape[0], 32))
        risk_proposal_network, *rp_others = nnx.merge(graph_rp, state_rp)
        z, taus_risk = risk_proposal_network.generate(taus)
        taus_risk = jax.lax.stop_gradient(taus_risk.copy())

        def loss_fn(actor_model):
            actions, log_prob = actor_model.sample_and_log_prob(observations, z)
            qfs = critic(observations, actions, z, taus_risk)

            qfs = jnp.mean(qfs, axis=-2).min(axis=-1, keepdims=True)
            loss = ent_coef * log_prob - qfs
            loss = loss.mean()
            return loss, (loss, log_prob)

        grad, (loss, log_prob) = nnx.grad(loss_fn, has_aux=True)(actor)
        opt_actor.update(grads=grad)
        metric_actor.update(ent=-log_prob.mean(), pi_loss=loss)
        _, new_state = nnx.split((actor, opt_actor, metric_actor))
        _, rp_state = nnx.split((risk_proposal_network, *rp_others))
        return new_state, log_prob, rp_state

    @staticmethod
    @partial(jax.jit, static_argnames=("soft_update_ratio",))
    def polyak_update(graph_def, state, target_params, soft_update_ratio: float):
        model, _, _ = nnx.merge(graph_def, state)
        _, current_param, _ = nnx.split(model, nnx.Param, ...)
        new_params = optax.incremental_update(current_param, target_params, soft_update_ratio)
        return new_params

    @partial(jax.jit, static_argnums=(0,))
    def ent_coef_loss(self,
                      graph_ent_coef, state_ent_coef,
                      log_prob
                      ):
        ent_coef_model, opt_ent_coef, metric = nnx.merge(graph_ent_coef, state_ent_coef)

        def loss_fn(coef_):
            ent_coef = coef_()
            loss = -coef_.denormalize(ent_coef) * (log_prob.mean() + self.target_entropy)
            return loss.mean(), (loss, ent_coef)

        grad, (loss, coef) = nnx.grad(loss_fn, has_aux=True)(ent_coef_model)
        opt_ent_coef.update(grad)
        metric.update(ent_loss=loss, ent_coef=coef)

        _, new_state = nnx.split((ent_coef_model, opt_ent_coef, metric))
        return new_state

    @property
    def train_log(self):
        self.critic, self.opt_critic, self.metric_critic = nnx.merge(self.graph_critic, self.state_critic)
        critic_loss = self.metric_critic.compute()
        self.metric_critic.reset()
        # no need to reassign graph
        _, self.state_critic = nnx.split((self.critic, self.opt_critic, self.metric_critic))

        self.actor, self.opt_actor, self.metric_actor = nnx.merge(self.graph_actor, self.state_actor)
        actor_loss = self.metric_actor.compute()
        self.metric_actor.reset()
        _, self.state_actor = nnx.split((self.actor, self.opt_actor, self.metric_actor))

        self.ent_coef, self.opt_ent_coef, self.metric_ent_coef = nnx.merge(self.graph_ent_coef, self.state_ent_coef)
        ent_coef_loss = self.metric_ent_coef.compute()
        self.metric_ent_coef.reset()
        _, self.state_ent_coef = nnx.split((self.ent_coef, self.opt_ent_coef, self.metric_ent_coef))
        return {**critic_loss, **actor_loss, **ent_coef_loss}

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({"critic": self.state_critic, "actor": self.state_actor,
                         "ent_coef": self.state_ent_coef, "rp": self.state_rp}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            states = pickle.load(f)
        self.state_critic = states['critic']
        self.state_actor = states['actor']
        self.state_ent_coef = states['ent_coef']
        self.state_rp = states['rp']


class ConvRiskConditionedPolicy(RiskConditionedPolicy):
    def build_actor(self):
        self.critic: ConvEnsembleCritic
        self.actor = Actor(self.critic.conv_feature,
                           self.action_space, rngs=self.rngs)
        self.opt_actor = nnx.Optimizer(self.actor, optax.chain(
            optax.adabelief(learning_rate=self.lr)
        ))
        self.metric_actor = nnx.MultiMetric(
            pi_loss=nnx.metrics.Average('pi_loss'),
            ent=nnx.metrics.Average('ent')
        )
        self.graph_actor, self.state_actor = nnx.split((self.actor, self.opt_actor, self.metric_actor))

    def build_critic(self):
        self.critic = ConvEnsembleCritic(self.observation_space.shape[-1],
                                         self.action_dim, rngs=self.rngs, n_critics=3, )
        self.opt_critic = nnx.Optimizer(self.critic, optax.chain(
            optax.adabelief(learning_rate=self.lr)
        ))
        self.metric_critic = nnx.MultiMetric(qf_loss=nnx.metrics.Average('qf_loss'),
                                             mmd=nnx.metrics.Average('mmd'),
                                             recon=nnx.metrics.Average('recon'))

        self.graph_critic, self.state_critic = nnx.split((self.critic, self.opt_critic, self.metric_critic))
        self.params_target_critic = copy_param(self.critic)

    def build_ent_coef(self, ):
        if self.ent_coef_init == 'auto':
            # for sufficiently broad action space, the entropy is often overestimated
            # therefore, policy action value is often overestimated
            self.ent_coef_init = np.exp(1 - (np.prod(self.action_space.shape))).item()

        self.ent_coef = EntCoef(init=self.ent_coef_init, rngs=self.rngs)
        self.opt_ent_coef = nnx.Optimizer(self.ent_coef, optax.adam(self.lr))
        self.metric_ent_coef = nnx.MultiMetric(ent_loss=nnx.metrics.Average('ent_loss'),
                                               ent_coef=nnx.metrics.Average('ent_coef'))
        self.graph_ent_coef, self.state_ent_coef = nnx.split((self.ent_coef, self.opt_ent_coef, self.metric_ent_coef))

    def observation_dim(self):
        return self.observation_space.shape[-1]

    @partial(jax.jit, static_argnums=(0,))
    def critic_loss(self,
                    graph_critic, state_critic, params_target_critic,
                    graph_actor, state_actor,
                    graph_ent_coef, state_ent_coef,
                    graph_rp, state_rp,
                    observations, actions, rewards, dones, next_observations,
                    *,
                    key
                    ):
        critic, opt_critic, metric_critic = nnx.merge(graph_critic, state_critic)
        graph, param, *others = nnx.split(critic, nnx.Param, ...)

        target_critic = nnx.merge(graph, params_target_critic, *others)
        keys = jax.random.split(key, 3)
        current_taus = jax.random.uniform(keys[0], shape=(actions.shape[0], 32))
        next_taus = jax.random.uniform(keys[1], shape=(actions.shape[0], 32))
        ent_coef_model, _, _ = nnx.merge(graph_ent_coef, state_ent_coef)
        ent_coef = ent_coef_model()
        risk_proposal_network, *rp_others = nnx.merge(graph_rp, state_rp)
        z, _ = risk_proposal_network.generate(next_taus)
        actor_next_observation = target_critic.encode(next_observations)

        actor, _, _ = nnx.merge(graph_actor, state_actor)
        actor: Actor

        next_action, next_log_prob = actor.sample_and_log_prob(actor_next_observation, z, )

        next_qf = target_critic(next_observations, next_action, z, next_taus)

        next_qf = next_qf.reshape(next_qf.shape[0], -1).sort(axis=-1)
        next_qf = next_qf[..., :-3] - ent_coef * next_log_prob

        td_target = rewards.reshape(-1, 1) + self.gamma * (1 - dones.reshape(-1, 1)) * next_qf

        def loss_fn(qf):
            current_qf, wae_loss, recon_loss = qf.qf_and_recon_loss(observations, actions, z, current_taus,
                                                                    next_observations, dones,
                                                                    keys[2])

            qf_loss = jax.vmap(quanitle_regression_loss, in_axes=(None, -1, None), out_axes=-1)(td_target,
                                                                                                current_qf,
                                                                                                current_taus)
            loss = qf_loss.sum(axis=(-1,)).mean() + 0.5 * wae_loss.mean() + recon_loss.mean()
            return loss, (qf_loss.sum(axis=-1).mean(), wae_loss.mean(), recon_loss.mean())

        grad, (qf_loss, wae_loss, recon_loss) = nnx.grad(loss_fn, has_aux=True)(critic)
        opt_critic.update(grad)
        metric_critic.update(qf_loss=qf_loss, mmd=wae_loss, recon=recon_loss)
        _, new_state = nnx.split((critic, opt_critic, metric_critic))
        _, rp_state = nnx.split((risk_proposal_network, *rp_others))
        return new_state, rp_state

    @partial(jax.jit, static_argnums=(0,))
    def actor_loss(self,
                   graph_critic, state_critic,
                   graph_actor, state_actor,
                   graph_ent_coef, state_ent_coef,
                   graph_rp, state_rp,
                   observations,
                   *,
                   key
                   ):

        critic, _, _ = nnx.merge(graph_critic, state_critic)
        actor, opt_actor, metric_actor = nnx.merge(graph_actor, state_actor)
        ent_coef_model, _, _ = nnx.merge(graph_ent_coef, state_ent_coef)
        ent_coef = ent_coef_model()
        taus = jax.random.uniform(key, shape=(observations.shape[0], 32))
        risk_proposal_network, *rp_others = nnx.merge(graph_rp, state_rp)
        z, taus_risk = risk_proposal_network.generate(taus)
        taus_risk = jax.lax.stop_gradient(taus_risk.copy())
        encoded_observation = critic.encode(observations)

        def loss_fn(actor_model):
            actions, log_prob = actor_model.sample_and_log_prob(encoded_observation, z)
            qfs = critic.critic(encoded_observation, actions, z, taus_risk)
            qfs = jnp.mean(qfs, axis=-2).min(axis=-1, keepdims=True)
            loss = ent_coef * log_prob - qfs
            loss = loss.mean()
            return loss, (loss, log_prob)

        grad, (loss, log_prob) = nnx.grad(loss_fn, has_aux=True)(actor)
        opt_actor.update(grads=grad)
        metric_actor.update(ent=-log_prob.mean(), pi_loss=loss)
        _, new_state = nnx.split((actor, opt_actor, metric_actor))
        _, rp_state = nnx.split((risk_proposal_network, *rp_others))
        return new_state, log_prob, rp_state

    def predict(self, observation, risk_embedding, *, deterministic: bool = False):

        if deterministic:
            action, obs = self._predict_deterministic(self.graph_actor, self.state_actor,
                                                      self.graph_critic, self.state_critic,
                                                      observation, risk_embedding)

            return np.asarray(action)
        else:
            # reassign because of  seed problem
            action, self.state_actor = self._predict_rand(self.graph_actor, self.state_actor,
                                                          self.graph_critic, self.state_critic,
                                                          observation, risk_embedding)

            return np.asarray(action)

    @staticmethod
    @jax.jit
    def _predict_rand(graph_actor, state_actor,
                      graph_critic, state_critic,
                      observations, z):
        critic, _, _ = nnx.merge(graph_critic, state_critic)
        observations = critic.encode(observations)
        actor, *others = nnx.merge(graph_actor, state_actor)
        action, _ = actor.sample_and_log_prob(observations, z)
        _, state_actor = nnx.split((actor, *others))
        return action, state_actor

    @staticmethod
    @jax.jit
    def _predict_deterministic(graph_actor, state_actor,
                               graph_critic, state_critic, observations, z):
        critic, _, _ = nnx.merge(graph_critic, state_critic)
        observations = critic.encode(observations)
        actor, _, _ = nnx.merge(graph_actor, state_actor)
        action = actor.mode(observations, z)
        return action, observations


class MultiInputRiskConditionedPolicy(ConvRiskConditionedPolicy):
    def observation_dim(self):
        return None

    def train_step(self, batch: DictReplayBufferSamples):
        def transform(sample):
            keys = sample.keys()
            for k in keys:
                sample[k] = sample[k].cpu().numpy()
            return sample

        observations = transform(batch.observations)
        actions = batch.actions.cpu().numpy()
        rewards = batch.rewards.cpu().numpy()
        dones = batch.dones.cpu().numpy()
        next_observation = transform(batch.next_observations)

        self.state_critic, self.params_target_critic, self.state_actor, self.state_ent_coef, self.state_rp = self._train_step(
            self.graph_critic, self.state_critic, self.params_target_critic,
            self.graph_actor, self.state_actor,
            self.graph_ent_coef, self.state_ent_coef, self.graph_rp, self.state_rp,
            observations, actions, rewards, dones, next_observation,
            key=self.rngs()
        )

    def build_actor(self):
        self.critic: ConvEnsembleCritic
        self.actor = Actor(self.critic.conv_feature +
                           + 256,
                           self.action_space, rngs=self.rngs)
        self.opt_actor = nnx.Optimizer(self.actor, optax.chain(
            optax.adabelief(learning_rate=self.lr)
        ))
        self.metric_actor = nnx.MultiMetric(
            pi_loss=nnx.metrics.Average('pi_loss'),
            ent=nnx.metrics.Average('ent')
        )
        self.graph_actor, self.state_actor = nnx.split((self.actor, self.opt_actor, self.metric_actor))

    def build_critic(self):
        self.critic = MultiInputCritic(self.observation_space['image'].shape[-1],
                                       self.action_dim, rngs=self.rngs, n_critics=3,
                                       in_feature_dim=self.observation_space['feature'].shape[-1], )
        self.opt_critic = nnx.Optimizer(self.critic, optax.chain(

            optax.adabelief(learning_rate=self.lr)
        ))
        self.metric_critic = nnx.MultiMetric(qf_loss=nnx.metrics.Average('qf_loss'),
                                             mmd=nnx.metrics.Average('mmd'),
                                             recon=nnx.metrics.Average('recon'))

        self.graph_critic, self.state_critic = nnx.split((self.critic, self.opt_critic, self.metric_critic))
        self.params_target_critic = copy_param(self.critic)

    @partial(jax.jit, static_argnums=(0,))
    def critic_loss(self,
                    graph_critic, state_critic, params_target_critic,
                    graph_actor, state_actor,
                    graph_ent_coef, state_ent_coef,
                    graph_rp, state_rp,
                    observations, actions, rewards, dones, next_observations,
                    *,
                    key
                    ):
        critic, opt_critic, metric_critic = nnx.merge(graph_critic, state_critic)
        graph, param, *others = nnx.split(critic, nnx.Param, ...)

        target_critic = nnx.merge(graph, params_target_critic, *others)
        keys = jax.random.split(key, 3)
        current_taus = jax.random.uniform(keys[0], shape=(actions.shape[0], 32))
        next_taus = jax.random.uniform(keys[1], shape=(actions.shape[0], 32))
        ent_coef_model, _, _ = nnx.merge(graph_ent_coef, state_ent_coef)
        ent_coef = ent_coef_model()
        risk_proposal_network, *rp_others = nnx.merge(graph_rp, state_rp)
        z, _ = risk_proposal_network.generate(next_taus)
        actor_next_observation = target_critic.encode(next_observations)

        actor, _, _ = nnx.merge(graph_actor, state_actor)
        actor: Actor

        next_action, next_log_prob = actor.sample_and_log_prob(actor_next_observation, z, )

        next_qf = target_critic(next_observations, next_action, z, next_taus)

        next_qf = next_qf.reshape(next_qf.shape[0], -1).sort(axis=-1)
        next_qf = next_qf[..., :-3] - ent_coef * next_log_prob

        td_target = rewards.reshape(-1, 1) + self.gamma * (1 - dones.reshape(-1, 1)) * next_qf

        def loss_fn(qf):
            current_qf, wae_loss, recon_loss = qf.qf_and_recon_loss(observations, actions, z, current_taus,
                                                                    next_observations, rewards, dones,
                                                                    keys[2])

            qf_loss = jax.vmap(quanitle_regression_loss, in_axes=(None, -1, None), out_axes=-1)(td_target,
                                                                                                current_qf,
                                                                                                current_taus)
            loss = qf_loss.sum(axis=(-1,)).mean() + 0.5 * wae_loss.mean() + recon_loss.mean()
            return loss, (qf_loss.sum(axis=-1).mean(), wae_loss.mean(), recon_loss.mean())

        grad, (qf_loss, wae_loss, recon_loss) = nnx.grad(loss_fn, has_aux=True)(critic)
        opt_critic.update(grad)
        metric_critic.update(qf_loss=qf_loss, mmd=wae_loss, recon=recon_loss)
        _, new_state = nnx.split((critic, opt_critic, metric_critic))
        _, rp_state = nnx.split((risk_proposal_network, *rp_others))
        return new_state, rp_state

    @partial(jax.jit, static_argnums=(0,))
    def actor_loss(self,
                   graph_critic, state_critic,
                   graph_actor, state_actor,
                   graph_ent_coef, state_ent_coef,
                   graph_rp, state_rp,
                   observations,
                   *,
                   key
                   ):

        critic, _, _ = nnx.merge(graph_critic, state_critic)
        actor, opt_actor, metric_actor = nnx.merge(graph_actor, state_actor)
        ent_coef_model, _, _ = nnx.merge(graph_ent_coef, state_ent_coef)
        ent_coef = ent_coef_model()
        taus = jax.random.uniform(key, shape=(observations['image'].shape[0], 32))
        risk_proposal_network, *rp_others = nnx.merge(graph_rp, state_rp)
        z, taus_risk = risk_proposal_network.generate(taus)
        taus_risk = jax.lax.stop_gradient(taus_risk.copy())
        encoded_observation = critic.encode(observations)

        def loss_fn(actor_model):
            actions, log_prob = actor_model.sample_and_log_prob(encoded_observation, z)
            qfs = critic.critic(encoded_observation, actions, z, taus_risk)
            qfs = jnp.mean(qfs, axis=-2).min(axis=-1, keepdims=True)
            loss = ent_coef * log_prob - qfs
            loss = loss.mean()
            return loss, (loss, log_prob)

        grad, (loss, log_prob) = nnx.grad(loss_fn, has_aux=True)(actor)
        opt_actor.update(grads=grad)
        metric_actor.update(ent=-log_prob.mean(), pi_loss=loss)
        _, new_state = nnx.split((actor, opt_actor, metric_actor))
        _, rp_state = nnx.split((risk_proposal_network, *rp_others))
        return new_state, log_prob, rp_state

    @partial(jax.jit, static_argnums=(0,))
    def critic_loss(self,
                    graph_critic, state_critic, params_target_critic,
                    graph_actor, state_actor,
                    graph_ent_coef, state_ent_coef,
                    graph_rp, state_rp,
                    observations, actions, rewards, dones, next_observations,
                    *,
                    key
                    ):
        critic, opt_critic, metric_critic = nnx.merge(graph_critic, state_critic)
        graph, param, *others = nnx.split(critic, nnx.Param, ...)

        target_critic = nnx.merge(graph, params_target_critic, *others)
        keys = jax.random.split(key, 3)
        current_taus = jax.random.uniform(keys[0], shape=(actions.shape[0], 32))
        next_taus = jax.random.uniform(keys[1], shape=(actions.shape[0], 32))
        ent_coef_model, _, _ = nnx.merge(graph_ent_coef, state_ent_coef)
        ent_coef = ent_coef_model()
        risk_proposal_network, *rp_others = nnx.merge(graph_rp, state_rp)
        z, _ = risk_proposal_network.generate(next_taus)
        actor_next_observation = target_critic.encode(next_observations)

        actor, _, _ = nnx.merge(graph_actor, state_actor)
        actor: Actor

        next_action, next_log_prob = actor.sample_and_log_prob(actor_next_observation, z, )

        next_qf = target_critic(next_observations, next_action, z, next_taus)

        next_qf = next_qf.reshape(next_qf.shape[0], -1).sort(axis=-1)
        next_qf = next_qf[..., :-3] - ent_coef * next_log_prob

        td_target = rewards.reshape(-1, 1) + self.gamma * (1 - dones.reshape(-1, 1)) * next_qf

        def loss_fn(qf):
            current_qf, wae_loss, recon_loss = qf.qf_and_recon_loss(observations, actions, z, current_taus,
                                                                    next_observations, rewards, dones,
                                                                    keys[2])

            qf_loss = jax.vmap(quanitle_regression_loss, in_axes=(None, -1, None), out_axes=-1)(td_target,
                                                                                                current_qf,
                                                                                                current_taus)
            loss = qf_loss.sum(axis=(-1,)).mean() + 0.5 * wae_loss.mean() + recon_loss.mean()
            return loss, (qf_loss.sum(axis=-1).mean(), wae_loss.mean(), recon_loss.mean())

        grad, (qf_loss, wae_loss, recon_loss) = nnx.grad(loss_fn, has_aux=True)(critic)
        opt_critic.update(grad)
        metric_critic.update(qf_loss=qf_loss, mmd=wae_loss, recon=recon_loss)
        _, new_state = nnx.split((critic, opt_critic, metric_critic))
        _, rp_state = nnx.split((risk_proposal_network, *rp_others))
        return new_state, rp_state

    def predict(self, observation, risk_embedding, *, deterministic: bool = False):

        if deterministic:
            action, obs = self._predict_deterministic(self.graph_actor, self.state_actor,
                                                      self.graph_critic, self.state_critic,
                                                      observation, risk_embedding)

            return np.asarray(action)
        else:
            # reassign because of  seed problem
            action, self.state_actor = self._predict_rand(self.graph_actor, self.state_actor,
                                                          self.graph_critic, self.state_critic,
                                                          observation, risk_embedding)

            return np.asarray(action)

    @staticmethod
    @jax.jit
    def _predict_rand(graph_actor, state_actor,
                      graph_critic, state_critic,
                      observations, z):
        critic, _, _ = nnx.merge(graph_critic, state_critic)
        observations = critic.encode(observations)
        actor, *others = nnx.merge(graph_actor, state_actor)
        action, _ = actor.sample_and_log_prob(observations, z)
        _, state_actor = nnx.split((actor, *others))
        return action, state_actor

    @staticmethod
    @jax.jit
    def _predict_deterministic(graph_actor, state_actor,
                               graph_critic, state_critic, observations, z):
        critic, _, _ = nnx.merge(graph_critic, state_critic)
        observations = critic.encode(observations)
        actor, _, _ = nnx.merge(graph_actor, state_actor)
        action = actor.mode(observations, z)
        return action, observations
