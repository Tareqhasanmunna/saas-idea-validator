"""
Base Agent Interface
All RL agents inherit from this base class
"""

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Abstract base class for all RL agents"""
    
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.action_count = 0
        
    @abstractmethod
    def choose_action(self, state, available_actions=None):
        """Choose an action given a state"""
        pass
    
    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """Update agent based on experience"""
        pass
    
    @abstractmethod
    def save(self, filepath):
        """Save agent to file"""
        pass
    
    @abstractmethod
    def load(self, filepath):
        """Load agent from file"""
        pass
