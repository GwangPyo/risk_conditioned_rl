from gymnasium import ActionWrapper
import gymnasium as gym

class NormalizeActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = env.action_space.low
        high = env.action_space.high
        self.scale = (high - low) / 2
        self.shift = (high + low) / 2
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.env.action_space.shape)

    def action(self, action):
        return self.scale * action + self.shift
