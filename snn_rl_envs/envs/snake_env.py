import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from base_env import BaseEnv
"""
Snake Environment.
State: flat array of grid_size * grid_size numbers, where:
    1. 0 = empty
    2. 1 = snake body
    3. 2 = snake head
    4. 3 = food
    5. 4 = obstacle
    
Actions: 4 discrete
    1. 0 = up
    2. 1 = down
    3. 2 = left
    4. 3 = right
    
Reward: +1 for eating food, -1 for dying, 0 otherwise

Episode ends when snake hits a wall, snake hits itself, or an obstacle
"""

# action constants
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class SnakeEnv(BaseEnv):
    def __init__(self, grid_size=10, wall_wrapping=False, obstacles=None):
        super().__init__()
        self.grid_size = grid_size
        self.wall_wrapping = wall_wrapping
        self.obstacles = obstacles if obstacles is not None else []
        
    def reset(self):
        # create empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # place obstacles
        for (r, c) in self.obstacles:
            self.grid[r, c] = 4
            
        # we place the snake in the middle, initially 3 cells long
        mid = self.grid_size // 2
        self.snake = [(mid, mid), (mid, mid-1), (mid, mid-2)] 
        self.direction = RIGHT
        
        # draw snake on grid
        for i, (r, c) in enumerate(self.snake):
            self.grid[r, c] = 2 if i == 0 else 1 # 2 = head, 1 = body
        
        # place food randomly
        self._place_food()
        
        return self._get_state()
    
    def get_state_space(self):
        return (self.grid_size * self.grid_size, 'discrete')
    
    def step(self, action):
        # update the direction so we can prevent moving back into yourself
        opposite = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
        if action != opposite[self.direction]: 
            self.direction = action
            
        # we calculate the new head position
        head_r, head_c = self.snake[0]
        if self.direction == UP:
            new_r, new_c = head_r - 1, head_c
        elif self.direction == DOWN:
            new_r, new_c = head_r + 1, head_c
        elif self.direction == LEFT:
            new_r, new_c = head_r, head_c - 1   
        elif self.direction == RIGHT:
            new_r, new_c = head_r, head_c + 1
            
        # handle wall wrapping
        if self.wall_wrapping:
            new_r = new_r % self.grid_size
            new_c = new_c % self.grid_size
        else:
            # check for wall collision
            if not (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size):
                return self._get_state(), -1, True, {} # because hitting the wall ends the game
            
        # check for self collision or obstacle collision
        if self.grid[new_r, new_c] == 1 or self.grid[new_r, new_c] == 4: # 1 = body, 4 = obstacle
            return self._get_state(), -1, True, {} # because hitting yourself or an obstacle ends the game
        
        # check if there's food
        ate_food = self.grid[new_r, new_c] == 3
        
        # move the snake
        self.grid[self.snake[0][0], self.snake[0][1]] = 1 # old head becomes body
        if not ate_food:
            tail_r, tail_c = self.snake.pop() # remove tail
            self.grid[tail_r, tail_c] = 0 # clear old tail from grid
        self.snake.insert(0, (new_r, new_c)) # add new head
        self.grid[new_r, new_c] = 2 # update grid with new head
        
        # spawn new food if we ate it
        if ate_food:
            self._place_food()
            return self._get_state(), 1, False, {} # +1 reward
        
        return self._get_state(), 0, False, {} # no reward otherwise
    
    def _place_food(self):
        empty_cells = list(zip(*np.where(self.grid == 0)))
        if empty_cells:
            idx = np.random.randint(len(empty_cells))
            r, c = empty_cells[idx]
            self.grid[r, c] = 3 # place food
    
    def _get_state(self):
        return self.grid.flatten()
    
    def get_action_space(self):
        return (4, 'discrete') # up, down, left, right
    
    def render(self):
        symbols = {0: '.', 1: 'o', 2: 'H', 3: 'F', 4: 'X'}
        print('\n' + '-' * (self.grid_size * 2 + 1))
        for row in self.grid:
            print('|' + ' '.join(symbols[cell] for cell in row) + '|')
        print('-' * (self.grid_size * 2 + 1) + '\n')
        
    def close(self):
        pass