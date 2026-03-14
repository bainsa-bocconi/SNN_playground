import gymnasium as gym
from pettingzoo.classic import tictactoe_v3

# CartPole
print("CartPole")
env = gym.make("CartPole-v1")
obs, info = env.reset()
print(f"Initial state: {obs}") # 4 numbers, cart position, cart velocity, pole angle, pole velocity 
print(f"Action space: {env.action_space}") # discrete: 0 = left, 1 = right

for step in range(5):
    action = env.action_space.sample() # pick a random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"Step {step + 1}: action = {action}, reward = {reward}, done = {done}")
    if done:
        obs, info = env.reset()
env.close()

# Pendulum
print("\nPendulum")
env = gym.make("Pendulum-v1")
obs, info = env.reset()
print(f"Initial state: {obs}") # 3 numbers, cos(angle), sin(angle), angle velocity
print(f"Action space: {env.action_space}") # continuous: 1 number in range [-2, 2], which is the torque applied to the pendulum

for step in range(5):
    action = env.action_space.sample() # pick a random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"Step {step + 1}: action = {action}, reward = {reward}, done = {done}")
    if done:
        obs, info = env.reset()
env.close()

# FrozenLake
print("\nFrozenLake")
env = gym.make("FrozenLake-v1")
obs, info = env.reset()
print(f"Initial state: {obs}") # which tile the agent is on (0-15)
print(f"Action space: {env.action_space}") # discrete: 0 = left, 1 = down, 2 = right, 3 = up

for step in range(5):
    action = env.action_space.sample() # pick a random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"Step {step + 1}: action = {action}, reward = {reward}, done = {done}")
    if done:
        obs, info = env.reset()
env.close()

# Tic Tac Toe
print("\nTic Tac Toe")
env = tictactoe_v3.env()
env.reset()
print(f"Agents: {env.agents}") # ['player_1', 'player_2']

for step in range(5):
    agent = env.agent_selection # whose turn it is
    obs, reward, terminated, truncated, info = env.last()
    done = terminated or truncated
    if done:
        action = None
    else:
        action = env.action_space(agent).sample() # pick a random action
    env.step(action)
    print(f"Step {step + 1}: agent = {agent}, action = {action}, reward = {reward}, done = {done}")
env.close()

print("\nAll environments ran successfully!")