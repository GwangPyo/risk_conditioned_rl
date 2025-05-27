from risk_sensitive_rl.base import RiskSensitiveOffPolicyRLJax
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
import gymnasium as gym
from typing import Callable, Optional
from risk_sensitive_rl.grips.policy import (RiskConditionedPolicy, ConvRiskConditionedPolicy,
                                            MultiInputRiskConditionedPolicy)


class GRIPS(RiskSensitiveOffPolicyRLJax):
    policy: RiskConditionedPolicy

    def __init__(self,
                 env: gym.Env | VecEnv,
                 risk_proposal_path: str,
                 policy_type: str = 'MlpPolicy',
                 gamma: float = 0.99,
                 batch_size: int = 256,
                 buffer_size: int = int(3e+5),
                 lr: float = 3e-4,
                 policy_kwargs: Optional[dict] = None,
                 eval_risk: Callable = lambda x: 0.1 * x,
                 *,
                 seed: int = 42,
                 ):
        self.policy_type = policy_type
        self.eval_risk = eval_risk
        self.path = risk_proposal_path
        self.seed = seed
        self.np_rngs = np.random.default_rng(seed)
        super().__init__(
            env, gamma, batch_size, buffer_size, lr, policy_kwargs,
        )
        self.policy.build()
        ph = np.zeros(shape=(self.env.num_envs, 4))
        self.learning_state = np.asarray(self.rp_net.generate(ph)[0]).copy()
        low = self.env.action_space.low
        high = self.env.action_space.high
        self.scale = (high - low) / 2
        self.shift = (high + low) / 2

    def risk_measure_to_embedding(self, eval_risk) -> np.ndarray:
        if eval_risk is None:
            eval_risk = self.eval_risk
        x = np.linspace(0, 1, self.rp_net.n_x)
        y = eval_risk(x)
        z = self.rp_net.encoder(np.concatenate([x, y], axis=-1))
        return z

    @property
    def rp_net(self):
        return self.policy.risk_proposal_network

    def set_state(self, index):
        z_i = self.np_rngs.uniform(size=(self.rp_net.latent_dim,))
        z_i[-1] = self.rp_net.cut_off_coef * z_i[-1]
        self.learning_state[index] = z_i

    def build_policy(self):
        if self.policy_type == 'MlpPolicy':
            self.policy = RiskConditionedPolicy(self.env.observation_space, self.env.action_space,
                                                risk_proposal_network_path=self.path, lr=self.lr,
                                                # float(-np.prod(self.env.action_space.shape).astype(np.float32))
                                                target_entropy=-(np.prod(self.env.action_space.shape)),
                                                seed=self.seed,
                                                )
        elif self.policy_type == 'CnnPolicy':
            self.policy = ConvRiskConditionedPolicy(
                self.env.observation_space, self.env.action_space,
                risk_proposal_network_path=self.path,
                target_entropy=-(np.prod(self.env.action_space.shape)),
                seed=self.seed, lr=self.lr
            )
        elif self.policy_type == 'MultiInputPolicy':
            self.policy = MultiInputRiskConditionedPolicy(
                self.env.observation_space, self.env.action_space,
                risk_proposal_network_path=self.path,
                target_entropy=-(np.prod(self.env.action_space.shape)),
                seed=self.seed, lr=self.lr
            )

    def predict(self, observation, *, state: Optional = None, deterministic: bool = True) -> np.ndarray:
        batch_add_flag = False
        if state is None:
            state = self.learning_state
        if self.policy_type != 'MultiInputPolicy':
            if observation.shape == self.env.observation_space.shape:
                observation = observation[None]
                batch_add_flag = True
        else:

            key = self.env.observation_space.keys()
            for k in key:
                if observation[k].shape == self.env.observation_space[k].shape:
                    observation[k] = observation[k][None]
                    batch_add_flag = True

        actions = self.policy.predict(observation, state)
        if batch_add_flag:
            actions = actions.squeeze(axis=0)

        return np.asarray(actions)

    def train_step(self):
        batch = self.buffer.sample(self.batch_size)
        self.policy.train_step(batch)

    def get_train_log(self) -> dict:
        return self.policy.train_log

    def save(self, path):
        self.policy.save(path)

    def load(self, path):
        self.policy.load(path)


if __name__ == "__main__":
    from risk_sensitive_rl.test_env import RiskConditionTestEnv
    from stable_baselines3.common.vec_env import SubprocVecEnv
    env = gym.make('HalfCheetah-v5')
    model = GRIPS(env=SubprocVecEnv([lambda: gym.make('HalfCheetah-v5') for _ in range(1)]),
                  risk_proposal_path='/home/yoo/risk_conditioned/risk_sampler/rp_trained_0.pkl')
    model.learn(int(3e+6), log_interval=1, episodic_learn=False)
    model.save("test_hopper")
    z = model.risk_measure_to_embedding(lambda x: 0.1 * x)
