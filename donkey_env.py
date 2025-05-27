from typing import Optional, Union

import gymnasium
import gym
import numpy as np
from gym.core import ObsType

import math
from collections import deque
from typing import Tuple
from typing import Sequence
import scipy
import cv2


def calc_reward(self, done: bool) -> float:
    # Normalization factor, real max speed is around 30
    # but only attained on a long straight line
    # max_speed = 10

    if done:
        return -100.

    if self.cte > self.max_cte:
        return -100.

    # Collision
    if self.hit != "none":
        return -100.

    # going fast close to the center of lane yeilds best reward
    if self.forward_vel > 0.0:
        return ((1.0 - (math.fabs(self.cte) / self.max_cte)) ** 2) * self.forward_vel

    # in reverse, reward doesn't have centering term as this can result in some exploits
    return self.forward_vel * self.max_cte


class WaypointRewardTracker:
    def __init__(self, waypoints: list[tuple[float, float, float]], distance_threshold: float = 3.):
        self.waypoints = waypoints
        self.distance_threshold = distance_threshold
        self.current_target_idx = 0
        self.prev_dist = None
        self.done = False

    def compute_distance(self, pos, target):
        pos = np.array([pos[0], pos[-1]])

        target = np.array([target[0], target[-1]])
        return np.linalg.norm(pos - target)

    def compute_reward(self, pos) -> Tuple[float, bool]:

        target = self.waypoints[self.current_target_idx]
        dist = self.compute_distance(pos, target)

        reward = 25 * (self.prev_dist - dist)

        reset_timeout = False
        self.prev_dist = dist

        if dist < self.distance_threshold:
            self.current_target_idx += 1
            if self.current_target_idx >= len(self.waypoints):
                print("All waypoints reached.")
                self.reset(pos)
                reset_timeout = True
            else:
                target = self.waypoints[self.current_target_idx]
                print(f"Reached waypoint {self.current_target_idx}, moving to next.")
                reset_timeout = True
            # reward += 100
        # recompute for waypoint change
        self.prev_dist = self.compute_distance(pos, target)
        return reward, reset_timeout

    def _init_distance(self, pos):
        if self.waypoints and self.current_target_idx < len(self.waypoints):
            self.prev_dist = self.compute_distance(pos, self.waypoints[self.current_target_idx])

    def reset(self, pos):
        self.current_target_idx = 0
        self.prev_dist = None
        self.done = False
        if pos is not None:
            self._init_distance(pos)

    @property
    def current_waypoint(self):
        return np.asarray(self.waypoints[self.current_target_idx]).copy()

    @property
    def current_waypoint_index(self):
        return self.current_target_idx


class DonkeyEnv(gymnasium.Env):
    render_mode: str = 'Human'
    positions = [
        (-3.791579246521, 0.0701264292001724, -5.81955289840698),
        (-3.07351398468018, 0.0701260790228844, -3.7985405921936),
        (1.39314949512482, 0.0701222568750381, 1.15984702110291),
        (6.62557649612427, 0.0701224133372307, 6.78724718093872),
        (12.8750696182251, 0.0701277703046799, 13.9314346313477),
        (14.4823923110962, 0.0701236724853516, 17.7815742492676),
        (15.5519123077393, 0.0701224580407143, 20.1752605438232),
        (14.5904531478882, 0.0701258778572083, 26.2067947387695),
        (9.25092887878418, 0.0701227188110352, 31.4559535980225),
        (-3.76456594467163, 0.070128321647644, 26.4042663574219),
        (-4.23381233215332, 0.0701223239302635, 20.5129413604736),
        (-2.04961228370667, 0.0701279863715172, 17.5442066192627),
        (0.737050235271454, 0.0701256096363068, 14.7147130966187),
        (5.82819652557373, 0.0701252818107605, 8.02784442901611),
        (9.41132736206055, 0.0701224878430367, 0.902119338512421),
        (11.7731447219849, 0.0701282620429993, -2.05879926681519),
        (13.2073383331299, 0.0701228752732277, -4.83134889602661),
        (14.3866453170776, 0.0701228156685829, -9.38142585754395),
        (13.6388740539551, 0.0701282918453217, -12.3565168380737),
        (11.786696434021, 0.0701228007674217, -14.9013385772705),
        (4.93426513671875, 0.0701529532670975, -18.0295352935791),
        (-0.549292266368866, 0.0701255798339844, -15.7595367431641),
        (-3.50244545936584, 0.0701279193162918, -10.2267789840698),
    ]

    steer_limit: float = 1
    throttle_max: float = 1
    throttle_min: float = 0.
    frame_skip: int = 1
    max_cte: float = 30

    def __init__(self,
                 host: str,
                 port: int = 9091,
                 stack: int = 3,
                 size: int | Sequence[int] = (60, 80),
                 timeout_step: int = int(2e+2)
                 ):
        self.stack = stack
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.max_timeout_step = timeout_step // self.frame_skip

        self.env = gym.make('donkey-avc-sparkfun-v0', conf={
            "host": host,
            "port": port,
            "max_cte": self.max_cte,
            "start_delay": 1.0,
            "throttle_max": self.throttle_max,
            "throttle_min": self.throttle_min,
            "steer_limit": self.steer_limit,
            "frame_skip": self.frame_skip
        })
        self.waypoint_tracker = WaypointRewardTracker(self.positions)

        self.obs_deque = deque(maxlen=stack)
        self.action_deque = deque(maxlen=self.max_timeout_step)
        self.env.set_reward_fn(calc_reward)
        self.step_cnt = 0

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        # flush deque
        del self.obs_deque
        del self.action_deque
        self.step_cnt = 0
        self.action_deque = deque(maxlen=self.max_timeout_step)
        self.obs_deque = deque(maxlen=self.stack)

        obs = self.process_image(self.env.reset())
        _, _, _, info = self.env.viewer.observe()
        a_sample = self.action_space.sample()
        for _ in range(self.stack):
            self.obs_deque.append(np.zeros_like(obs))
        for _ in range(self.max_timeout_step):
            self.action_deque.append(np.zeros_like(a_sample))
        self.obs_deque.append(obs)
        self.waypoint_tracker.reset(info['pos'])
        dist = self.waypoint_tracker.compute_distance(info['pos'], self.waypoint_tracker.current_waypoint)
        others = (list(info['gyro']) + list(info['accel']) + list(info['vel'])
                  + list(info['car']) + list([info['speed']]) + list([dist]) + list(info['pos']) + [
                      10 * info['cte'] / self.max_cte] +
                  self.waypoint_tracker.current_waypoint.tolist())

        ar = scipy.fft.dct(np.asarray(self.action_deque).copy(), axis=0, norm='ortho')
        ar = ar.flatten()[:100]
        return {"image": np.concatenate(self.obs_deque, axis=-1),
                "feature": np.concatenate([others,
                                           ar], axis=0)
                }, info

    def step(self, action: np.ndarray):
        done = False
        # to prevent loop rolloing

        self.step_cnt += 1
        reward = 0
        prev_way = self.waypoint_tracker.current_target_idx
        self.action_deque.append(action)
        for _ in range(self.env.frame_skip):
            self.env.viewer.take_action(action)

            observation, _reward, _done, info = self.env.viewer.observe()

            if _done:
                reward += _reward
            else:
                waypoint_reward, reset_timeout = self.waypoint_tracker.compute_reward(info['pos'])
                current_way_point = self.waypoint_tracker.current_waypoint
                reward += waypoint_reward

                if reset_timeout:
                    self.step_cnt = 0

            done = _done
            if done:
                break
        timeout = False
        if self.step_cnt > self.max_timeout_step:
            timeout = True
            done = True
            reward = -1000
        observation = self.process_image(observation)
        self.obs_deque.append(observation)
        obs = np.concatenate(self.obs_deque, axis=-1)
        dist = self.waypoint_tracker.compute_distance(info['pos'],
                                                      self.waypoint_tracker.current_waypoint)

        others = (list(info['gyro']) + list(info['accel']) + list(info['vel'])
                  + list(info['car']) + list([info['speed']]) + list([dist]) + list(info['pos']) + [
                      10 * info['cte'] / self.max_cte] +
                  self.waypoint_tracker.current_waypoint.tolist())

        ar = scipy.fft.dct(np.asarray(self.action_deque).copy(), axis=0, norm='ortho')
        ar = ar.flatten()[:100]

        return ({"image": obs, "feature": np.concatenate([np.asarray(others), ar], axis=0)},
                reward, done, timeout, info)

    def manual_step(self, action: np.ndarray):

        for _ in range(self.env.frame_skip):
            self.env.viewer.handler.send_control(*(action.tolist()))
            observation, _reward, _done, info = self.env.viewer.observe()

        return info

    @property
    def observation_space(self):
        return gymnasium.spaces.Dict(
            {"image": gymnasium.spaces.Box(0, self.env.VAL_PER_PIXEL,
                                           self.size + (self.stack,), dtype=np.uint8),
             "feature": gymnasium.spaces.Box(-np.inf, np.inf, shape=(121,), dtype=np.float32)
             }
        )

    @property
    def action_space(self):
        return gymnasium.spaces.Box(
            low=np.array([-self.steer_limit, self.throttle_min]),
            high=np.array([self.steer_limit, self.throttle_max]),

            dtype=np.float32,
        )

    def render(self, mode="human"):
        return self.env.render()

    def process_image(self, img):
        img = cv2.resize(img, dsize=tuple(reversed(self.size)))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None]
        return img

    @property
    def viewer(self):
        return self.env.viewer

    @property
    def handler(self):
        return self.env.viewer.handler

    @property
    def current_waypoint_index(self):
        return self.waypoint_tracker.current_target_idx

