"""
CARLA Environment for RL Training
"""
import carla
import numpy as np
import gym
from gym import spaces
import time
import math
from typing import Tuple, Dict, Any, Optional

from .vehicle_manager import VehicleManager
from .dwcl_manager import DWCLManager
from .traffic_manager import TrafficManager
from .behavior_agent import CustomBehaviorAgent
from config import constants as const


class CarlaEnv(gym.Env):
    """CARLA environment for DWCL-RL training"""
    
    def __init__(self, config):
        super(CarlaEnv, self).__init__()
        
        # Initialize components
        self.config = config
        self.vehicle_manager = None
        self.dwcl_manager = None
        self.traffic_manager = None
        self.battery_manager = None
        self.behavior_agent = None
        
        # State and action spaces
        self.observation_space = spaces.Box(
            low=np.array(const.OBSERVATION_LOW),
            high=np.array(const.OBSERVATION_HIGH),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(const.ACTION_DIM)
        
        # Training metrics
        self.start_time = None
        self.lane_switch_counter = 0
        self.invalid_action_counter = 0
        self.has_arrived = False
        self.target_speed = 40  # km/h
        
    def initialize_simulation(self, power_model):
        # Connect to CARLA
        self.client = carla.Client(const.SIM_HOST, const.SIM_PORT)
        self.client.set_timeout(30.0)

        # Load world
        self.world = self.client.load_world(const.TOWN)

        # Apply simulation settings
        settings = self.world.get_settings()
        settings.synchronous_mode = const.SYNC_MODE
        settings.fixed_delta_seconds = const.FIXED_DELTA_SECONDS
        settings.no_rendering_mode = bool(getattr(self.config, "no_rendering_mode", False))
        self.world.apply_settings(settings)

        # Prime the world (important in sync mode)
        if settings.synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Initialize managers that don't require the vehicle actor yet
        self.vehicle_manager = VehicleManager(self.world)
        self.dwcl_manager = DWCLManager(self.world, self.config)

        # Spawn vehicle FIRST
        self.vehicle_manager.spawn_vehicle()

        # Tick once after spawning (helps avoid race conditions)
        if settings.synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Now initialize Traffic Manager with a valid vehicle actor
        self.traffic_manager = TrafficManager(self.client, self.vehicle_manager.vehicle)

        # Create DWCL after world and vehicle exist
        self.dwcl_manager.create_dwcl()

        # Battery manager
        from src.battery_management.battery_model import BatteryManager
        destination = self.dwcl_manager.get_destination()
        self.battery_manager = BatteryManager(
            self.vehicle_manager.vehicle,
            destination,
            self.world,
            power_model,
            const.BATTERY_CAPACITY,
            const.AUXILIARY_POWER,
            self.dwcl_manager.get_all_corners()
        )

        # Behavior agent
        self.behavior_agent = CustomBehaviorAgent(
            self.vehicle_manager.vehicle,
            destination,
            self.config
        )

        self._set_spectator()

        # Initialize episode time baseline here or in reset()
        self.start_time = self.world.get_snapshot().timestamp.elapsed_seconds

        print("✅ CARLA Environment Initialized")

        
    def _set_spectator(self):
        """Set spectator view"""
        spectator = self.world.get_spectator()
        transform = self.vehicle_manager.vehicle.get_transform()
        transform.location.z += 10
        transform.rotation.pitch = -90
        spectator.set_transform(transform)
        
    def step(self, action: int):
        if self._is_terminal():
            return np.zeros(const.STATE_DIM, dtype=np.float32), 0.0, True, {}

        reward = self._execute_action(action)

        # Advance CARLA in sync mode
        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Battery update + next state
        delta_time = self._get_delta_time()
        self.battery_manager.update(delta_time)

        next_state = self._get_state()
        done = self._is_terminal()
        return next_state, reward, done, {}

        
    def _execute_action(self, action: int) -> float:
        """
        Execute the selected action
        """
        reward = 0
        lane_type = int(self.battery_manager.is_on_dwcl())
        
        if action == 0 and lane_type == 0:  # Move to DWCL
            self._move_to_dwcl()
            reward += 5
        elif action == 1 and lane_type == 1:  # Leave DWCL
            self._exit_dwcl()
            reward += 5
        elif action == 2 and lane_type == 1:  # Accelerate
            self._adjust_speed(increase=True)
            reward += 5
        elif action == 3 and lane_type == 1:  # Decelerate
            self._adjust_speed(increase=False)
            reward += 5
        elif action == 4 and lane_type == 1:  # Maintain speed
            self._maintain_speed()
            reward += 5
        elif action == 5 and lane_type == 0:  # Stay out of DWCL
            self.traffic_manager.set_vehicle_speed(0)
            reward += 5
        else:
            reward -= 20  # Invalid action penalty
            self.invalid_action_counter += 1
            
        # Additional reward based on state
        reward += self._compute_state_reward()
        
        return reward
    
    def _move_to_dwcl(self):
        """Move vehicle to nearest DWCL lane"""
        #dwcl_location = self.dwcl_manager.get_nearest_dwcl_location()
        #self.behavior_agent.navigate_to(dwcl_location)
        vehicle_location = self.vehicle_manager.vehicle.get_location()
        dwcl_location = self.dwcl_manager.get_nearest_dwcl_location(vehicle_location)
        self.behavior_agent.navigate_to(dwcl_location)

        
    def _exit_dwcl(self):
        """Exit DWCL to adjacent lane"""
        #left_lane = self.dwcl_manager.get_adjacent_lane(left=True)
        #self.behavior_agent.navigate_to(left_lane)
        vehicle_location = self.vehicle_manager.vehicle.get_location()
        left_lane = self.dwcl_manager.get_adjacent_lane(vehicle_location, left=True)
        self.behavior_agent.navigate_to(left_lane)

        
    def _adjust_speed(self, increase: bool = True, delta: float = 5):
        """Adjust vehicle speed"""
        if increase:
            self.target_speed = min(self.target_speed + delta, 100)
        else:
            self.target_speed = max(self.target_speed - delta, 10)
        self.traffic_manager.set_vehicle_speed(self.target_speed)
        
    def _maintain_speed(self):
        """Maintain current speed"""
        self.traffic_manager.set_vehicle_speed(self.target_speed)
        
    def _compute_state_reward(self) -> float:
        """
        Compute reward based on current state
        """
        reward = 0
        soc = self.battery_manager.get_soc()
        required_soc = self.battery_manager.get_required_soc()
        in_dwcl = self.battery_manager.is_on_dwcl()
        
        # SOC-based rewards
        if soc < 20:
            reward -= 20
        elif soc >= 20:
            reward += 20
            
        # DWCL charging rewards
        if in_dwcl and soc < required_soc + 20:
            reward += 15
        elif in_dwcl and soc > 80:
            reward -= 10
            
        # Lane switch penalty
        if self.lane_switch_counter > 0:
            reward -= self.lane_switch_counter ** 2
            
        # Time reward
        elapsed_time = self._get_elapsed_time()
        eta = self.battery_manager.get_eta()
        if elapsed_time < eta * 1.2:
            reward += 20
            
        # Destination reward
        if self.battery_manager.get_remaining_distance() < 10:
            if soc >= 20:
                reward += 30
            else:
                reward -= 30
                
        return reward
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector"""
        soc = self.battery_manager.get_soc()
        required_soc = self.battery_manager.get_required_soc()
        eta = self.battery_manager.get_eta()
        distance = self.battery_manager.get_remaining_distance()
        lane_type = int(self.battery_manager.is_on_dwcl())
        
        return np.array([soc, required_soc, eta, distance, lane_type, self.target_speed],
                       dtype=np.float32)
    
    def _get_delta_time(self) -> float:
        """Get time since last update"""
        current_time = self.world.get_snapshot().timestamp.elapsed_seconds
        delta = current_time - self.battery_manager.last_update_time
        self.battery_manager.last_update_time = current_time
        return delta
    
    def _get_elapsed_time(self) -> float:
        """Get elapsed simulation time"""
        if self.start_time is None:
            return 0
        current_time = self.world.get_snapshot().timestamp.elapsed_seconds
        return current_time - self.start_time
    
    def _is_terminal(self) -> bool:
        """Check if episode should terminate"""
        # Destination reached
        if self.battery_manager.get_remaining_distance() < 5:
            print("✅ Destination reached")
            self.has_arrived = True
            return True
            
        # Timeout
        elapsed_time = self._get_elapsed_time()
        initial_eta = self.battery_manager.get_initial_eta()
        if elapsed_time > 2 * initial_eta:
            print("⏰ Timeout - Episode terminated")
            return True
            
        # Battery depleted
        if self.battery_manager.get_soc() <= 0:
            print("🔋 Battery depleted")
            return True
            
        return False
    
    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode
        """
        # Reset metrics
        self.lane_switch_counter = 0
        self.invalid_action_counter = 0
        self.has_arrived = False
        self.target_speed = 40
        
        # Reset vehicle position
        start_transform = self.dwcl_manager.get_start_transform()
        self.vehicle_manager.reset_vehicle(start_transform)
        
        # Reset battery manager
        self.battery_manager.reset()
        
        # Reset time
        self.start_time = self.world.get_snapshot().timestamp.elapsed_seconds
        
        # Get initial state
        initial_state = self._get_state()
        
        print("🔄 Environment reset")
        return initial_state
    
    def close(self):
        """Clean up environment"""
        if self.vehicle_manager:
            self.vehicle_manager.destroy()
        if self.dwcl_manager:
            self.dwcl_manager.cleanup()
        print("🧹 Environment cleaned up")