import sys
import numpy as np
sys.path.insert(0, '.')
from envs.cartpole_env import CartPoleEnv
from envs.tictactoe_env import TicTacToeEnv
from envs.snake_env import SnakeEnv

def run_episode(env, env_name, max_steps=1000):
    # we run a single episode with random actions to test the environment
    print(f"Environment: {env_name}")
    print(f"Action space: {env.get_action_space()}")
    print(f"State space: {env.get_state_space()}")
    
    state = env.reset()
    total_reward = 0
    done = False
    step = 0
    
    while not done and step < max_steps:
        # we get action space size and pick a random action
        action_size, action_type = env.get_action_space()
        action = np.random.randint(action_size)
        state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        
    print(f"Episode finished in {step} steps. Total reward: {total_reward}.")
    env.close
    
if __name__ == "__main__":
    # CartPole environment
    run_episode(CartPoleEnv(), "CartPole")
    
    # TicTacToe environment
    run_episode(TicTacToeEnv(), "TicTacToe")
    
    # Snake environment (default 10x10, no wrapping, no obstacles)
    run_episode(SnakeEnv(grid_size = 10), "Snake 10x10")
    
    # Snake with wall wrapping
    run_episode(SnakeEnv(grid_size = 10, wall_wrapping = True), "Snake 10x10 with wall wrapping")
    
    # Snake with obstacles
    run_episode(SnakeEnv(grid_size = 10, obstacles=[(2,2),(2,3),(4,5)]), "Snake 10x10 with obstacles")
    
    print("All environments ran successfully!")