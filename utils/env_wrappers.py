import gym
import numpy as np


def _get_space_dim(spaces: gym.spaces.Space):
    if isinstance(spaces, gym.spaces.Box):
        return np.prod(spaces.shape)
    elif isinstance(spaces, gym.spaces.Discrete):
        return spaces.n
    else:
        raise NotImplementedError("Unsupported space ")


class NormalizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env=env)
        assert isinstance(env.action_space, gym.spaces.Box)
        dim = _get_space_dim(env.action_space)
        self.action_shape = env.action_space.shape
        self.action_space = gym.spaces.Box(-1, 1, shape=(dim, ), dtype=np.float32)
        self._low = env.action_space.low
        self._delta = env.action_space.high - env.action_space.low

    def action(self, action):
        action = action.copy()
        return ((action + 1.0) * self._delta / 2.0 + self._low).reshape(self.action_shape)

    def reverse_action(self, action):
        action = action.flatten()
        action = (2 * (action - self._low) / self._delta) - 1.
        return action.reshape(self.env.action_space.shape)


class NormalizeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, mu, sigma, eps=1e-12):
        super().__init__(env=env)
        self.mu = mu
        self.sigma = sigma
        self.eps = eps

    def observation(self, observation):
        return (observation - self.mu.squeeze())/(self.sigma.squeeze() + self.eps)


class TimeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, scale=0.001, limit=None):
        super().__init__(env)
        self.cnt = 0
        self.scale = scale
        self.limit = limit
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(np.prod(self.env.observation_space.shape) + 1, ))

    def reset(self, **kwargs):
        self.cnt = 0
        return super().reset(**kwargs)

    def step(self, action):
        self.cnt += 1
        obs, reward, done, info = super().step(action)
        if self.limit is not None:
            if self.cnt >= self.limit:
                done = True
        return obs, reward, done, info

    def observation(self, observation):
        observation = observation.flatten()
        return np.concatenate((observation, [self.cnt * self.scale]), axis=0)
