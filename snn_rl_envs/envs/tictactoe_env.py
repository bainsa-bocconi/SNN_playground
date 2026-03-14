import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from base_env import BaseEnv
from pettingzoo.classic import tictactoe_v3

"""
Defines the TicTacToe environment.
Our agent plays as player_1. player_2 is a random opponent controlled automatically.

State: 9 numbers (a 3x3 grid), where:
    1. 1 means our piece is there
    2. -1 means the opponent's piece is there
    3. 0 means the cell is empty

Action: 9 possible actions (which cell to place your piece, 0-8)
Reward: +1 for winning, -1 for losing, 0 for draw or ongoing game
Episode ends when somebody wins or when the board is full.
"""

class TicTacToeEnv(BaseEnv):
    def __init__(self):
        self.env = tictactoe_v3.env()
        self.our_player = 'player_1'
        self.opponent_player = 'player_2'
        
    def reset(self):
        self.env.reset()
        obs, _, _, _, _ = self.env.last()
        return self._process_obs(obs)
    
    def step(self, action):
        self.env.step(action) # player 1 acts
        
        # check if game ended after our move
        obs, reward, terminated, truncated, info = self.env.last()
        done = terminated or truncated
        if done:
            return self._process_obs(obs), reward, done, info
        
        # player 2 acts randomly
        opponent_action = self._get_legal_action()
        self.env.step(opponent_action)
        
        # get state after opponent's move
        obs, reward, terminated, truncated, info = self.env.last()
        done = terminated or truncated
        return self._process_obs(obs), -reward, done, info # reward is from opponent's perspective, so we negate it
    
    def _process_obs(self, obs):
        board = obs['observation']
        our_pieces = board[:, :, 0] # layer 0 = our pieces
        opp_pieces = board[:, :, 1] # layer 1 = opponent pieces
        state = our_pieces - opp_pieces # 1 for our piece, -1 for opponent, 0 for empty
        return state.flatten() # flatten to 1D array of length 9
    
    def _get_legal_action(self):
        obs, _, _, _, _ = self.env.last()
        action_mask = obs['action_mask']
        legal_actions = np.where(action_mask == 1)[0]
        return np.random.choice(legal_actions)
    
    def get_action_space(self):
        return (9, 'discrete')
    
    def get_state_space(self):
        return (9, 'discrete')
    
    def render(self):
        self.env.render()
        
    def close(self):
        self.env.close()
        