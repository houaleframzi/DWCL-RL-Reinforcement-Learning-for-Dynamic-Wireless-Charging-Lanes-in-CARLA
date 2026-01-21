"""
Experience replay buffer for DQN
"""
import random
import numpy as np
from collections import deque, namedtuple
from typing import List, Tuple, Optional


Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Circular buffer for storing experiences"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of experiences"""
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough experiences in buffer")
            
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([exp.state for exp in batch])
        actions = np.array([exp.action for exp in batch])
        rewards = np.array([exp.reward for exp in batch])
        next_states = np.array([exp.next_state for exp in batch])
        dones = np.array([exp.done for exp in batch])
        
        return states, actions, rewards, next_states, dones
        
    def __len__(self) -> int:
        return len(self.buffer)
        
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()