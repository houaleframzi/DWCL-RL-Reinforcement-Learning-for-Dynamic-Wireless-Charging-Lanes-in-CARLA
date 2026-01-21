"""
Deep Q-Network Agent for DWCL Management
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Optional
import os

from .policy_network import DQN
from .replay_buffer import ReplayBuffer
from config import constants as const


class DQNAgent:
    """DQN Agent with experience replay and target network"""
    
    def __init__(self, state_dim: int, action_dim: int, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.policy_net = DQN(state_dim, action_dim, config.hidden_layers).to(self.device)
        self.target_net = DQN(state_dim, action_dim, config.hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.learning_rate
        )
        
        # Experience replay
        self.memory = ReplayBuffer(config.replay_buffer_size)
        
        # Training parameters
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.target_update_freq = config.target_update_freq
        
        # Exploration
        self.epsilon = config.epsilon_start
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min
        
        # Training state
        self.train_step_counter = 0
        self.losses = []
        
        print(f"🤖 DQN Agent initialized on {self.device}")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  Network: {config.hidden_layers}")
        print(f"  Gamma: {self.gamma}, LR: {config.learning_rate}")
        
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select action using epsilon-greedy policy
        """
        if not evaluate and random.random() < self.epsilon:
            # Random exploration
            return random.randint(0, self.action_dim - 1)
        else:
            # Greedy exploitation
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
                
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
        
    def train(self) -> Optional[float]:
        """
        Train agent on a batch from replay buffer
        Returns loss if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        
        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # Compute loss
        loss = nn.functional.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Decay epsilon
        #self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        # Update target network
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # Record loss
        self.losses.append(loss.item())
        
        return loss.item()
        
    def save(self, path: str):
        """Save agent state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses,
            'train_step_counter': self.train_step_counter
        }
        
        torch.save(checkpoint, path)
        print(f"💾 Agent saved to {path}")
        
    def load(self, path: str):
        """Load agent state"""
        if not os.path.exists(path):
            print(f"⚠️ No checkpoint found at {path}")
            return
            
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.losses = checkpoint.get('losses', [])
        self.train_step_counter = checkpoint.get('train_step_counter', 0)
        
        print(f"📂 Agent loaded from {path}")
        print(f"  Epsilon: {self.epsilon:.4f}")
        print(f"  Training steps: {self.train_step_counter}")
        
    def get_epsilon(self) -> float:
        """Get current epsilon value"""
        return self.epsilon
        
    def get_losses(self) -> List[float]:
        """Get training losses"""
        return self.losses
        
    def set_eval_mode(self):
        """Set agent to evaluation mode"""
        self.policy_net.eval()
        self.target_net.eval()
        
    def set_train_mode(self):
        """Set agent to training mode"""
        self.policy_net.train()
        self.target_net.train()