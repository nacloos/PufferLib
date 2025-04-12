'''Simple Platformer Environment

A 2D platformer game where the player jumps between platforms to collect coins.
'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.platformer.cy_platformer import CyPlatformer


class Platformer(pufferlib.PufferEnv):
    def __init__(self, seed=None, num_envs=1, render_mode=None, report_interval=128,
             width=800, height=600, player_width=30, player_height=50, buf=None):

        # 12 observations: player (x,y,vx,vy) + 5 platforms (x,y)
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(12,), dtype=np.float32)
        # 4 actions: left, right, jump, no action
        self.single_action_space = gymnasium.spaces.Discrete(4)
        self.report_interval = report_interval
        self.render_mode = render_mode
        self.num_agents = num_envs

        super().__init__(buf=buf)
        self.c_envs = CyPlatformer(self.observations, self.actions, self.rewards,
            self.terminals, num_envs, width, height, player_width, player_height)

    def reset(self, seed=None):
        self.c_envs.reset()
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.c_envs.step()
        self.tick += 1

        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render()

    def close(self):
        self.c_envs.close()


def test_performance(timeout=10, atn_cache=1024, num_envs=1024):
    import time

    env = Platformer(num_envs=num_envs)
    env.reset()
    tick = 0

    actions = np.random.randint(
        0,
        env.single_action_space.n,
        (atn_cache, num_envs),
    )

    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]         
        env.step(atn)
        tick += 1

    print(f'SPS: {num_envs * tick / (time.time() - start)}')


if __name__ == '__main__':
    test_performance() 