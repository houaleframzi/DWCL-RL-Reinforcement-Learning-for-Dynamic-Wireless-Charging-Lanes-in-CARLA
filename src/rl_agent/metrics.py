"""
Training metrics collection and management
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class TrainingMetrics:
    """Collects and manages training metrics"""
    
    # Episode metrics
    episode_rewards: List[float] = field(default_factory=list)
    episode_steps: List[int] = field(default_factory=list)
    episode_times: List[float] = field(default_factory=list)
    final_socs: List[float] = field(default_factory=list)
    arrival_flags: List[bool] = field(default_factory=list)
    
    # Agent metrics
    epsilon_values: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    
    # Performance metrics
    lane_switch_counts: List[int] = field(default_factory=list)
    invalid_action_counts: List[int] = field(default_factory=list)
    
    def update(self, episode_metrics: Dict[str, Any]):
        """Update metrics with new episode data"""
        self.episode_rewards.append(episode_metrics.get('reward', 0))
        self.episode_steps.append(episode_metrics.get('steps', 0))
        self.episode_times.append(episode_metrics.get('time', 0))
        self.final_socs.append(episode_metrics.get('final_soc', 0))
        self.arrival_flags.append(episode_metrics.get('arrived', False))
        self.epsilon_values.append(episode_metrics.get('epsilon', 0))
        
        # Append losses if available
        if 'losses' in episode_metrics:
            self.losses.extend(episode_metrics['losses'])
            
    def add_agent_metrics(self, agent_losses: List[float], epsilon: float):
        """Add agent-specific metrics"""
        self.losses.extend(agent_losses)
        self.epsilon_values.append(epsilon)
        
    def add_performance_metrics(self, lane_switches: int, invalid_actions: int):
        """Add performance metrics"""
        self.lane_switch_counts.append(lane_switches)
        self.invalid_action_counts.append(invalid_actions)
        
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'episode_times': self.episode_times,
            'final_socs': self.final_socs,
            'arrival_flags': self.arrival_flags,
            'epsilon_values': self.epsilon_values,
            'losses': self.losses,
            'lane_switch_counts': self.lane_switch_counts,
            'invalid_action_counts': self.invalid_action_counts
        }
        
    def from_dict(self, data: Dict):
        """Load metrics from dictionary"""
        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_steps = data.get('episode_steps', [])
        self.episode_times = data.get('episode_times', [])
        self.final_socs = data.get('final_socs', [])
        self.arrival_flags = data.get('arrival_flags', [])
        self.epsilon_values = data.get('epsilon_values', [])
        self.losses = data.get('losses', [])
        self.lane_switch_counts = data.get('lane_switch_counts', [])
        self.invalid_action_counts = data.get('invalid_action_counts', [])
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame"""
        # Ensure all lists have the same length
        max_len = max(
            len(self.episode_rewards),
            len(self.episode_steps),
            len(self.episode_times),
            len(self.final_socs),
            len(self.arrival_flags),
            len(self.epsilon_values),
            len(self.lane_switch_counts),
            len(self.invalid_action_counts)
        )
        
        # Pad lists if necessary
        data = {
            'episode': list(range(1, max_len + 1)),
            'reward': self._pad_list(self.episode_rewards, max_len),
            'steps': self._pad_list(self.episode_steps, max_len),
            'time': self._pad_list(self.episode_times, max_len),
            'final_soc': self._pad_list(self.final_socs, max_len),
            'arrived': self._pad_list(self.arrival_flags, max_len, fill_value=False),
            'epsilon': self._pad_list(self.epsilon_values, max_len),
            'lane_switches': self._pad_list(self.lane_switch_counts, max_len),
            'invalid_actions': self._pad_list(self.invalid_action_counts, max_len)
        }
        
        return pd.DataFrame(data)
        
    def _pad_list(self, lst: List, length: int, fill_value=0):
        """Pad list to specified length"""
        if len(lst) >= length:
            return lst[:length]
        else:
            return lst + [fill_value] * (length - len(lst))
            
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.episode_rewards:
            return {}
            
        return {
            'total_episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'success_rate': np.mean(self.arrival_flags) * 100,
            'mean_final_soc': np.mean(self.final_socs),
            'mean_episode_time': np.mean(self.episode_times),
            'mean_steps': np.mean(self.episode_steps)
        }