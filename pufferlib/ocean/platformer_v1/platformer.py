'''Minimalistic Platformer Environment

A simple platformer game with a player, platforms, and a goal.
The player receives a reward of +1 when reaching the goal.
'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.platformer_v1.cy_platformer import CyPlatformer


class Platformer(pufferlib.PufferEnv):
    def __init__(self, seed=None, num_envs=1, render_mode=None, report_interval=128, 
                width=800, height=600, player_size=32, buf=None):
        """
        Platformer environment with a player, ground, and goal.
        
        Args:
            seed: Random seed
            num_envs: Number of parallel environments
            render_mode: Render mode
            report_interval: Interval for reporting metrics
            width: Width of the game screen
            height: Height of the game screen
            player_size: Size of the player character
            buf: Buffer for observations
        """
        # Define observation space: x, y, x_vel, y_vel
        self.single_observation_space = gymnasium.spaces.Box(
            low=-float('inf'), high=float('inf'), shape=(4,), dtype=np.float32)
        
        # Define action space: 0=noop, 1=left, 2=right, 3=jump
        self.single_action_space = gymnasium.spaces.Discrete(4)
        
        self.report_interval = report_interval
        self.render_mode = render_mode
        self.num_agents = num_envs

        super().__init__(buf=buf)
        self.c_envs = CyPlatformer(
            self.observations, self.actions, self.rewards,
            self.terminals, num_envs, width, height, player_size
        )

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
    env.reset(0)
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
    # Uncomment one of these to run:
    test_performance()
