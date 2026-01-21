"""
Simulation and Training Settings
"""

class SimulationSettings:
    """Configuration for CARLA simulation"""
    
    def __init__(self):
        # CARLA Settings
        self.host = 'localhost'
        self.port = 2000
        self.town = 'Town06'
        self.synchronous = True
        self.fixed_delta_seconds = 0.05
        self.no_rendering_mode = False
        
        # Vehicle Settings
        self.vehicle_model = 'nissan'
        self.spawn_location = (22.6, 251, 0.0)
        self.target_speed = 40  # km/h
        
        # DWCL Settings
        self.dwcl_length = 569
        self.dwcl_width = 3.5
        self.coil_width = 6
        self.coil_height = 1
        self.coil_spacing = 2
        self.num_coils = None  # Auto-calculated


class TrainingSettings:
    """Configuration for RL training"""
    
    def __init__(self):
        # DQN Parameters
        self.gamma = 0.9
        self.learning_rate = 0.001
        self.batch_size = 64
        self.replay_buffer_size = 10000
        self.target_update_freq = 1000
        
        # Exploration Parameters
        self.epsilon_start = 1.0
        self.epsilon_decay = 0.99995
        self.epsilon_min = 0.05
        
        # Training Schedule
        self.num_episodes = 6500
        self.max_steps_per_episode = 500
        self.checkpoint_interval = 50
        
        # Network Architecture
        self.hidden_layers = [128, 64]  # DQN hidden layer sizes
        self.activation = 'relu'
        
        # Reward Parameters
        self.reward_weights = {
            'soc_penalty': -20,
            'soc_reward': 20,
            'lane_switch_penalty': -1,
            'invalid_action_penalty': -20,
            'dwcl_charging_reward': 15,
            'time_reward': 20
        }