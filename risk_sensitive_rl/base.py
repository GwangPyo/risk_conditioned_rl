import numpy as np
from gymnasium.core import WrapperActType, ActType
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
import gymnasium as gym
from typing import Callable, Optional
from stable_baselines3.common.logger import Logger, HumanOutputFormat
import sys
from datetime import timedelta
from time import time
from collections import deque
from tqdm import trange
from gymnasium import ActionWrapper


class NormalizeActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = env.action_space.low
        high = env.action_space.high
        self.scale = (high - low) / 2
        self.shift = (high + low) / 2
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.env.action_space.shape)

    def action(self, action: WrapperActType) -> ActType:
        return self.scale * action + self.shift


class RiskSensitiveOffPolicyRLJax(object):
    def __init__(self,
                 env: gym.Env | VecEnv,
                 gamma: float = 0.99,
                 batch_size: int = 256,
                 buffer_size: int = int(3e+5),
                 lr: float = 3e-4,
                 policy_kwargs: Optional[dict] = None,
                 ):
        if not isinstance(env, VecEnv):
            print("wrap with dummy vec env")
            env = DummyVecEnv([lambda: NormalizeActionWrapper(env)])
        self.env = env
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            self.buffer: ReplayBuffer = DictReplayBuffer(
                n_envs=env.num_envs,
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                buffer_size=buffer_size
            )
        else:
            self.buffer: ReplayBuffer = ReplayBuffer(
                n_envs=env.num_envs,
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                buffer_size=buffer_size
            )
        if policy_kwargs is None:
            policy_kwargs = {}
        self.lr = lr
        self.policy_kwargs = policy_kwargs.update(lr=lr, gamma=gamma)
        self.score_deque = deque(maxlen=100)
        self.cost_deque = deque(maxlen=100)
        self.step_deque = deque(maxlen=100)
        self.build_policy()
        self.logger = Logger(folder=None, output_formats=[HumanOutputFormat(sys.stdout)])
        self.state: Optional = self.get_state()

    def get_state(self):
        return None

    def build_policy(self):
        pass

    def predict(self, observation, *, state: Optional = None, deterministic: bool = True) -> np.ndarray:
        pass

    def get_train_log(self) -> dict:
        pass

    def train_step(self):
        pass

    def pretrain_step(self):
        pass

    def set_state(self, index):
        pass

    def learn(self, n_steps: int,
              log_interval: int = 4,
              train_frequency: int = 1,
              n_train: int = 1,
              learning_start: int = 100,
              test_interval: int = int(1e+5),
              episodic_learn: bool = False,
              need_pretrain: bool = False,
              ):
        last_obs = self.env.reset()
        score = np.zeros(self.env.num_envs)
        epicnt = 0.
        step_cnt = np.zeros(self.env.num_envs, dtype=np.int32)
        at_least_one_train: bool = False
        need_pretrain = need_pretrain
        train_start_time = time()
        for s in range(n_steps):
            start_time = time()
            if s < learning_start:
                action = np.asarray([self.env.action_space.sample() for _ in range(self.env.num_envs)])
            else:
                action = self.predict(last_obs, deterministic=False)

                if s % train_frequency == 0 and not episodic_learn:
                    it = range(n_train) if n_train < 100 else trange(n_train)
                    at_least_one_train = True
                    for _ in it:
                        self.train_step()

            next_obs, reward, done, info = self.env.step(action)
            self.buffer.add(
                obs=last_obs.copy(), next_obs=next_obs, action=action, reward=reward, done=done, infos=info,
            )
            score = score + reward
            last_obs = next_obs.copy()
            step_cnt += 1
            end_time = time()
            elapsed_time = time()
            fps = self.env.num_envs / (end_time - start_time)
            time_spent = elapsed_time - train_start_time
            self.logger.record_mean("Time/fps", fps)
            self.logger.record("Time/elapsed",
                               str(timedelta(seconds=int(time_spent))))
            # v = s / t
            v = (s + 1) / time_spent

            remaining_frames = n_steps - s
            eta_seconds = remaining_frames / v

            eta = timedelta(seconds=int(eta_seconds))

            self.logger.record("Time/eta", str(eta))

            if done.any():
                index = np.where(done)[0]
                s_ = []

                for i in index:
                    epicnt += 1
                    s_.append(score[i])
                    self.score_deque.append(score[i])
                    self.step_deque.append(step_cnt[i])
                    if episodic_learn and s > learning_start:
                        if need_pretrain:
                            for _ in trange(learning_start * 10):
                                self.pretrain_step()
                            need_pretrain = False

                        learnings = int(n_train) * int(step_cnt[i])

                        it = trange(learnings)
                        at_least_one_train = True
                        for _ in it:
                            self.train_step()
                    self.logger.record(key='Episode/num_epi', value=epicnt)
                    self.logger.record(key='Episode/epilen', value=step_cnt[i])
                    score[i] = 0
                    step_cnt[i] = 0
                    self.set_state(i)

                self.logger.record(key="Episode/score", value=np.mean(s_))
                self.logger.record(key='Episode/mean_score', value=np.mean(self.score_deque))
                self.logger.record(key='Episode/mean_epilen', value=np.mean(self.step_deque))
                self.logger.record(key='Train/current_step', value=s * self.env.num_envs)

                if at_least_one_train:
                    train_log = self.get_train_log()
                    for k, v in train_log.items():
                        self.logger.record(key=f"Train/{k}", value=v)
                    at_least_one_train = False
                if epicnt % log_interval == 0:
                    self.logger.dump()
