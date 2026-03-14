from abc import ABC, abstractmethod

# Abstract and base class for all environments. Every environment should inherit from this class
# and implement the abstract methods.

class BaseEnv(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def get_action_space(self):
        pass
    
    @abstractmethod
    def get_state_space(self):
        pass
    
    def render(self):
        pass