import gymnasium as gym
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from base_env import BaseEnv

"""
CartPole-v1 Environment
State: 4 continuous numbers
    1. Cart position    
    2. Cart velocity
    3. Pole angle
    4. Pole angular velocity
Actions: 2 discrete actions
    1. 0 - Push cart to the left
    2. 1 - Push cart to the right
    
Reward; +1 for every step which makes the pole stay up
Episode ends when pole falls, cart goes off screen, or after 500 steps
"""

class CartPoleEnv(BaseEnv):
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        
    def reset(self):
        obs, info = self.env.reset()
        return obs
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
    
    def get_action_space(self):
        return (self.env.action_space.n, "discrete")
    
    def get_state_space(self):
        return (self.env.observation_space.shape[0], "continuous")
    
    def render(self):
        self.env.render()
        
    def close(self):
        self.env.close()