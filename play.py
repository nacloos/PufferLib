from pufferlib.ocean import env_creator
import matplotlib.pyplot as plt
import time
from raylib import rl, colors
import pyray
import sys
import numpy as np

def play_random(env_name="platformer_v2", fps=30):
    """Run the environment with random actions."""
    make_env = env_creator(env_name)
    env = make_env()

    env.reset()
    frame_time = 1/fps
    frame_count = 0

    prev_reward = 0
    while True:
        start_time = time.time()
        
        # action = env.action_space.sample()
        action = 2
        ob, reward, done, truncated, info = env.step(action)

        env.render()
        print(f"Frame {frame_count}: {ob}, {reward}")
        frame_count += 1
        
        print("reward", reward)

        if done:
            print("done")
            break
            
        # Sleep to maintain target FPS
        elapsed = time.time() - start_time
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

        prev_reward = reward
            

# TODO: fail to capture inputs
def play_manual(env_name="platformer", fps=60):
    """Play the game manually using raylib input handling."""
    # Create environment
    make_env = env_creator(env_name)
    env = make_env()
    
    # Initialize environment
    env.reset()
    
    print("Controls:")
    print("LEFT ARROW: Move left")
    print("RIGHT ARROW: Move right")
    print("UP ARROW or SPACE: Jump")
    print("ESC: Quit")
    
    # Main game loop
    while not rl.WindowShouldClose():
        # Default to no action
        action = 3  # NOOP
        
        # Process input - must be checked each frame while window is open
        if rl.IsKeyDown(rl.KEY_LEFT):
            action = 0  # LEFT
        elif rl.IsKeyDown(rl.KEY_RIGHT):
            action = 1  # RIGHT
        elif rl.IsKeyDown(rl.KEY_UP) or rl.IsKeyDown(rl.KEY_SPACE):
            action = 2  # JUMP
        
        # Step the environment with the action
        ob, reward, done, _, info = env.step(action)
        
        # Render
        env.render()
        
        # Print debug info
        if reward != 0:
            print(f"Action: {action}, Reward: {reward}")
            
        # If episode is done, reset
        if done:
            print("Episode finished! Resetting...")
            env.reset()
    
    # Close environment when done
    env.close()
    print("Game closed")

if __name__ == "__main__":
    play_random()
    # play_manual()
