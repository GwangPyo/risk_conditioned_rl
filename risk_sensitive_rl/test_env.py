import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame


def r_worst(x):
    return -1


def r_cvar(x):
    return 8 * x - 2


def r_neutral(x):
    return (5 / 2) * (12 * x ** 2 + 12 * x - 5)


def r(a, x=None):
    a = np.asarray(a)
    if x is None:
        x = np.random.uniform(0, 1, size=a.shape)
    else:
        x = np.asarray(x)
        if x.shape != a.shape:
            raise ValueError("x and a must have the same shape if x is provided")

    abs_a = np.abs(a)
    is_neg = a <= 0

    r_worst_val = r_worst(x)
    r_cvar_val = r_cvar(x)
    r_neutral_val = r_neutral(x)

    result = np.where(
        is_neg,
        abs_a * r_worst_val + (1 - abs_a) * r_cvar_val,
        (1 - abs_a) * r_cvar_val + abs_a * r_neutral_val
    )

    return result


class RiskConditionTestEnv(gym.Env):
    def __init__(self, seed: int = 4):
        self.observation_space = gym.spaces.Box(-1, 1, shape=(1,))
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,))
        self.np_rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        return np.ones(shape=(1,)), {}

    def step(self, action):
        return np.zeros(1, ), r(action, self.np_rng.uniform(size=(1,))).item(), True, True, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass
