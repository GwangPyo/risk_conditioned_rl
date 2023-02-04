from rl_agents.rcdsac import RCDSAC
import gym


if __name__ == '__main__':
    model = RCDSAC(env=gym.make("Pendulum-v1"))
    model.learn(int(3e+5))
