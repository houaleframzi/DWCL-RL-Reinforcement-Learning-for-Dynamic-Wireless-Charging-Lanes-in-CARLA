"""
CARLA Environment for DWCL RL + interactive control.

This version keeps the "notebook-style" blocking manoeuvres (enter/exit DWCL):
- Enter/exit is executed via a blocking BehaviorAgent loop (run_step/apply_control),
  with world.tick() advancing the sim and BatteryManager updated inside the loop.
- Outside DWCL, the vehicle should cruise under Traffic Manager autopilot.

Public helpers:
- initialize_simulation(power_model, spawn_location=None)
- move_to_dwcl()
- exit_dwcl()
- stop_car()
- cleanup()/close()
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import carla
import gym
import numpy as np
from gym import spaces

# If you have CARLA PythonAPI in a non-standard location, keep your sys.path edits.
# Prefer configuring PYTHONPATH instead, but we won't change your workflow here.
# (Comment out if not needed.)
sys.path.insert(0, r"D:\WindowsNoEditor\PythonAPI")
sys.path.insert(0, r"D:\WindowsNoEditor\PythonAPI\carla")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import constants as const
from src.carla_simulator.vehicle_manager import VehicleManager
from src.carla_simulator.dwcl_manager import DWCLManager
from src.carla_simulator.traffic_manager import TrafficManager
from src.carla_simulator.behavior_agent import CustomBehaviorAgent
from src.battery_management.battery_model import BatteryManager


class CarlaEnv(gym.Env):
    """CARLA environment for DWCL-RL training + interactive play."""

    def __init__(self, config):
        super().__init__()

        self.config = config

        # CARLA handles
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None

        # Managers
        self.vehicle_manager: Optional[VehicleManager] = None
        self.dwcl_manager: Optional[DWCLManager] = None
        self.traffic_manager: Optional[TrafficManager] = None
        self.battery_manager: Optional[BatteryManager] = None

        # Behavior agent (used inside manoeuvres)
        self.behavior_agent: Optional[CustomBehaviorAgent] = None

        # Spaces
        self.observation_space = spaces.Box(
            low=np.array(const.OBSERVATION_LOW, dtype=np.float32),
            high=np.array(const.OBSERVATION_HIGH, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(const.ACTION_DIM)

        # Episode bookkeeping
        self.start_time: Optional[float] = None
        self.lane_switch_counter = 0
        self.invalid_action_counter = 0
        self.has_arrived = False

        # Cruise target speed (km/h)
        self.target_speed = 40.0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize_simulation(
        self,
        power_model,
        spawn_location: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Connect to CARLA, spawn ego vehicle, create DWCL, init BatteryManager and agent."""

        self.client = carla.Client(const.SIM_HOST, const.SIM_PORT)
        self.client.set_timeout(30.0)

        self.world = self.client.load_world(const.TOWN)

        # Settings
        settings = self.world.get_settings()
        settings.synchronous_mode = bool(const.SYNC_MODE)
        settings.fixed_delta_seconds = float(const.FIXED_DELTA_SECONDS)
        settings.no_rendering_mode = bool(getattr(self.config, "no_rendering_mode", False))
        self.world.apply_settings(settings)

        # Prime
        if settings.synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Managers
        self.vehicle_manager = VehicleManager(self.world)
        self.dwcl_manager = DWCLManager(self.world, self.config)

        # Spawn ego
        if spawn_location is not None:
            self.vehicle_manager.spawn_vehicle(location=spawn_location)
        else:
            self.vehicle_manager.spawn_vehicle()

        if settings.synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Traffic manager (after ego exists)
        self.traffic_manager = TrafficManager(self.client, self.vehicle_manager.vehicle)

        # Create DWCL
        self.dwcl_manager.create_dwcl()

        # Battery manager (expects Keras power_model with .predict)
        destination = self.dwcl_manager.get_destination()
        self.battery_manager = BatteryManager(
            self.vehicle_manager.vehicle,
            destination,
            self.world,
            power_model,
            const.BATTERY_CAPACITY,
            const.AUXILIARY_POWER,
            self.dwcl_manager.get_all_corners(),
        )

        # Behavior agent
        self.behavior_agent = CustomBehaviorAgent(
            self.vehicle_manager.vehicle,
            destination,
            self.config,
        )

        # Spectator & time baseline
        self._set_spectator()
        self.start_time = self.world.get_snapshot().timestamp.elapsed_seconds

        # Start cruising under TM initially
        self._resume_outside_dwcl_cruise()

        print("✅ CARLA Environment Initialized")

    def _set_spectator(self) -> None:
        if self.world is None or self.vehicle_manager is None:
            return
        spectator = self.world.get_spectator()
        transform = self.vehicle_manager.vehicle.get_transform()
        transform.location.z += 10
        spectator.set_transform(transform)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.lane_switch_counter = 0
        self.invalid_action_counter = 0
        self.has_arrived = False
        self.target_speed = 40.0

        # Teleport vehicle
        start_transform = self.dwcl_manager.get_start_transform()
        self.vehicle_manager.reset_vehicle(start_transform)

        # Prime world (CRITICAL)
        if self.world.get_settings().synchronous_mode:
            for _ in range(6):
                self.world.tick()
        else:
            for _ in range(6):
                self.world.wait_for_tick()

        # Rebind Traffic Manager (CRITICAL)
        self.traffic_manager = TrafficManager(
            self.client,
            self.vehicle_manager.vehicle
        )

        # Recreate BatteryManager (CRITICAL)
        destination = self.dwcl_manager.get_destination()
        self.battery_manager = BatteryManager(
            self.vehicle_manager.vehicle,
            destination,
            self.world,
            self.battery_manager.power_model,
            const.BATTERY_CAPACITY,
            const.AUXILIARY_POWER,
            self.dwcl_manager.get_all_corners()
        )

        # Reset time bookkeeping
        now = self.world.get_snapshot().timestamp.elapsed_seconds
        self.start_time = now
        self.battery_manager.last_update_time = now

        # Resume cruise (outside DWCL)
        self._resume_outside_dwcl_cruise()

        return self._get_state()


    def step(self, action: int):
        if self._is_terminal():
            return np.zeros(const.STATE_DIM, dtype=np.float32), 0.0, True, {}

        reward = self._execute_action(action)

        # Advance CARLA
        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Battery update + next state
        dt = self._get_delta_time()
        self.battery_manager.update(dt)

        next_state = self._get_state()
        done = self._is_terminal()
        return next_state, reward, done, {}

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------
    def _execute_action(self, action: int) -> float:
        reward = 0.0
        lane_type = int(self.battery_manager.is_on_dwcl())

        # DWCL lane switching actions
        if action == 0 and lane_type == 0:  # Move to DWCL
            self._move_to_dwcl()
            reward += 5.0

        elif action == 1 and lane_type == 1:  # Leave DWCL
            self._exit_dwcl()
            reward += 5.0

        # Speed control actions (only meaningful inside DWCL in your design)
        elif action == 2 and lane_type == 1:  # Accelerate
            self._adjust_speed(increase=True)
            reward += 5.0

        elif action == 3 and lane_type == 1:  # Decelerate
            self._adjust_speed(increase=False)
            reward += 5.0

        elif action == 4 and lane_type == 1:  # Maintain speed
            self._maintain_speed()
            reward += 5.0

        # Outside DWCL: keep moving under autopilot/TM (do NOT set speed=0)
        elif action == 5 and lane_type == 0:
            self._resume_outside_dwcl_cruise()
            reward += 5.0

        else:
            reward -= 20.0
            self.invalid_action_counter += 1

        # Additional shaping reward from current state
        reward += self._compute_state_reward()
        return float(reward)

    # ------------------------------------------------------------------
    # Public control helpers (for interactive / play mode)
    # ------------------------------------------------------------------
    def move_to_dwcl(self) -> None:
        self._move_to_dwcl()

    def exit_dwcl(self) -> None:
        self._exit_dwcl()

    def set_autopilot(self, enabled: bool) -> None:
        if self.traffic_manager is not None:
            self.traffic_manager.set_autopilot(enabled)

    # ------------------------------------------------------------------
    # Maneuver helpers (notebook-style blocking maneuvers)
    # ------------------------------------------------------------------
    def _get_sim_time(self) -> float:
        try:
            snap = self.world.get_snapshot()
            return float(snap.timestamp.elapsed_seconds)
        except Exception:
            return time.time()

    def stop_car(self) -> None:
        """Hard stop the ego vehicle and take control from autopilot/TM."""
        veh = self.vehicle_manager.vehicle
        if veh is None:
            return

        # Disable autopilot (TM port if available)
        try:
            if self.traffic_manager is not None:
                veh.set_autopilot(False, self.traffic_manager.tm.get_port())
            else:
                veh.set_autopilot(False)
        except Exception:
            try:
                veh.set_autopilot(False)
            except Exception:
                pass

        # Apply a hard brake
        control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0, hand_brake=False)
        try:
            veh.apply_control(control)
        except Exception:
            pass

    def _release_brake(self) -> None:
        """Clear brake command so TM/autopilot can accelerate again."""
        veh = self.vehicle_manager.vehicle
        if veh is None:
            return
        try:
            veh.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
        except Exception:
            pass

    def _compute_brake_safe_distance(self) -> float:
        veh = self.vehicle_manager.vehicle
        if veh is None:
            return 30.0
        v = veh.get_velocity()
        speed_ms = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        speed_kmh = speed_ms * 3.6
        brake_safe_dist = (speed_kmh / 8.0) ** 2
        if speed_kmh < 15 and brake_safe_dist < 30:
            brake_safe_dist = 30
        return float(brake_safe_dist)

    def _get_nearest_dwcl_transform(self) -> Optional[carla.Transform]:
        """Pick nearest DWCL segment and offset forward by a braking-safe distance."""
        veh = self.vehicle_manager.vehicle
        if veh is None:
            return None

        corners_list = self.dwcl_manager.get_all_corners()
        if not corners_list:
            return None

        carla_map = self.world.get_map()
        veh_loc = veh.get_location()

        min_dist = float("inf")
        target_wp = None

        for corners in corners_list:
            if not corners:
                continue
            wp = carla_map.get_waypoint(corners[0], project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is None:
                continue
            d = veh_loc.distance(wp.transform.location)
            if d < min_dist:
                min_dist = d
                target_wp = wp

        if target_wp is None:
            return None

        target_transform = carla.Transform(target_wp.transform.location, target_wp.transform.rotation)
        target_transform.location.x += self._compute_brake_safe_distance()
        return target_transform

    def _get_nearest_no_dwcl_transform(self, left: bool = True) -> Optional[carla.Transform]:
        """Get adjacent lane transform ahead of the vehicle (defaults to left)."""
        veh = self.vehicle_manager.vehicle
        if veh is None:
            return None

        carla_map = self.world.get_map()
        current_wp = carla_map.get_waypoint(veh.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        if current_wp is None:
            return None

        adj_wp = current_wp.get_left_lane() if left else current_wp.get_right_lane()
        if adj_wp is None:
            return None

        target_transform = carla.Transform(adj_wp.transform.location, adj_wp.transform.rotation)
        target_transform.location.x += self._compute_brake_safe_distance()
        return target_transform

    def _execute_agent_maneuver(
        self,
        destination_transform: carla.Transform,
        done_predicate,
        timeout_seconds: float = 8.0,
        post_align: bool = True,
    ) -> None:
        """Blocking manoeuvre loop: tick -> battery update -> agent.run_step -> apply_control."""

        veh = self.vehicle_manager.vehicle
        if veh is None:
            return

        # Ensure agent exists
        if self.behavior_agent is None:
            self.behavior_agent = CustomBehaviorAgent(veh, self.dwcl_manager.destination, self.config)
        agent = self.behavior_agent

        # Disable autopilot for maneuver
        try:
            if self.traffic_manager is not None:
                veh.set_autopilot(False, self.traffic_manager.tm.get_port())
            else:
                veh.set_autopilot(False)
        except Exception:
            try:
                veh.set_autopilot(False)
            except Exception:
                pass

        self.stop_car()

        carla_map = self.world.get_map()

        end_wp = carla_map.get_waypoint(destination_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if end_wp is None:
            return

        start_wp = carla_map.get_waypoint(veh.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        if start_wp is None:
            return

        # Prefer set_global_plan if your CustomBehaviorAgent supports it; otherwise set_destination
        route_trace = None
        try:
            if hasattr(agent, "trace_route"):
                route_trace = agent.trace_route(start_wp, end_wp)
        except Exception:
            route_trace = None

        if route_trace:
            try:
                if hasattr(agent, "set_global_plan"):
                    agent.set_global_plan(route_trace)
                else:
                    agent.set_destination(end_wp.transform.location)
            except Exception:
                agent.set_destination(end_wp.transform.location)
        else:
            agent.set_destination(end_wp.transform.location)

        start_wall = time.time()
        last_sim = self._get_sim_time()

        while True:
            # Tick the world in sync mode (or sleep in async)
            try:
                if self.world.get_settings().synchronous_mode:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()
            except Exception:
                time.sleep(0.05)

            # Battery update
            try:
                now_sim = self._get_sim_time()
                dt = max(0.0, now_sim - last_sim)
                last_sim = now_sim
                self.battery_manager.update(dt)
            except Exception:
                pass

            # Apply agent control
            try:
                control = agent.run_step()
                veh.apply_control(control)
            except Exception:
                pass

            # Completion check
            try:
                if done_predicate(end_wp):
                    break
            except Exception:
                pass

            # Timeout
            if (time.time() - start_wall) > timeout_seconds:
                try:
                    veh.set_transform(end_wp.transform)
                except Exception:
                    pass
                break

        # Optional align to nearest driving waypoint
        if post_align:
            try:
                nearest_wp = carla_map.get_waypoint(veh.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
                if nearest_wp is not None:
                    veh.set_transform(nearest_wp.transform)
            except Exception:
                pass

        # Resume cruise under TM/autopilot OUTSIDE manoeuvre
        self._resume_outside_dwcl_cruise()

    def _move_to_dwcl(self) -> None:
        veh = self.vehicle_manager.vehicle
        if veh is None:
            return
        dest_tf = self._get_nearest_dwcl_transform()
        if dest_tf is None:
            return

        def done_pred(end_wp):
            in_dwcl = bool(self.battery_manager.is_on_dwcl())
            cur_wp = self.world.get_map().get_waypoint(veh.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
            same_lane = (cur_wp is not None and cur_wp.lane_id == end_wp.lane_id)
            return in_dwcl and same_lane

        self._execute_agent_maneuver(dest_tf, done_predicate=done_pred, timeout_seconds=8.0, post_align=True)

    def _exit_dwcl(self) -> None:
        veh = self.vehicle_manager.vehicle
        if veh is None:
            return
        dest_tf = self._get_nearest_no_dwcl_transform(left=True)
        if dest_tf is None:
            return

        def done_pred(end_wp):
            in_dwcl = bool(self.battery_manager.is_on_dwcl())
            cur_wp = self.world.get_map().get_waypoint(veh.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
            same_lane = (cur_wp is not None and cur_wp.lane_id == end_wp.lane_id)
            return (not in_dwcl) and same_lane

        self._execute_agent_maneuver(dest_tf, done_predicate=done_pred, timeout_seconds=8.0, post_align=True)

    # ------------------------------------------------------------------
    # Outside-DWCL cruise behaviour
    # ------------------------------------------------------------------
    def _resume_outside_dwcl_cruise(self) -> None:
        """
        Ensure the vehicle cruises under TM/autopilot.
        This is called after exiting DWCL and for action=5 outside DWCL.
        """
        if self.vehicle_manager is None or self.vehicle_manager.vehicle is None:
            return
        if self.traffic_manager is None:
            return

        try:
            self.traffic_manager.set_autopilot(True)
        except Exception:
            # fallback to direct autopilot enable
            try:
                self.vehicle_manager.vehicle.set_autopilot(True, self.traffic_manager.tm.get_port())
            except Exception:
                pass

        # Clear any hard brake applied previously
        self._release_brake()

        # Apply cruise target speed
        try:
            self.traffic_manager.set_vehicle_speed(self.target_speed)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Reward & state
    # ------------------------------------------------------------------
    def _adjust_speed(self, increase: bool = True, delta: float = 5.0) -> None:
        if increase:
            self.target_speed = min(self.target_speed + delta, 100.0)
        else:
            self.target_speed = max(self.target_speed - delta, 10.0)
        if self.traffic_manager is not None:
            self.traffic_manager.set_vehicle_speed(self.target_speed)

    def _maintain_speed(self) -> None:
        if self.traffic_manager is not None:
            self.traffic_manager.set_vehicle_speed(self.target_speed)

    def _compute_state_reward(self) -> float:
       
        reward = 0.0

        soc = float(self.battery_manager.get_soc())
        required_soc = float(self.battery_manager.get_required_soc())
        in_dwcl = bool(self.battery_manager.is_on_dwcl())

        # --- DWCL charging shaping ---
        if in_dwcl and soc < required_soc + 20:
            reward += 15
        if in_dwcl and soc > 80:
            reward -= 10
        if in_dwcl and soc > required_soc + 20:
            reward -= 3

        # --- Time/ETA shaping (notebook style) ---
        travel_time = float(self._get_elapsed_time())
        eta = float(self.battery_manager.get_eta())
        initial_eta = float(self.battery_manager.get_initial_eta())
        if travel_time < (initial_eta - eta) * 1.2:
            reward += 20

        # --- Speed shaping (inside DWCL) ---
        if in_dwcl and self.target_speed > 50:
            reward -= 5
        if in_dwcl and self.target_speed < 15:
            reward -= 15

        # --- SoC safety ---
        if soc < 20:
            reward -= 20
        else:
            reward += 20

        # --- Distance-to-destination shaping ---
        remaining_distance = float(self.battery_manager.get_remaining_distance())
        if remaining_distance < 10 and soc >= 20:
            reward += 30
        if remaining_distance < 10 and soc < 20:
            reward -= 30

        # --- Lane switch penalty ---
        if self.lane_switch_counter > 0:
            reward -= (self.lane_switch_counter ** 2)

        return float(reward)


    def _get_state(self) -> np.ndarray:
        soc = self.battery_manager.get_soc()
        required_soc = self.battery_manager.get_required_soc()
        eta = self.battery_manager.get_eta()
        distance = self.battery_manager.get_remaining_distance()
        lane_type = int(self.battery_manager.is_on_dwcl())

        return np.array([soc, required_soc, eta, distance, lane_type, self.target_speed], dtype=np.float32)

    def _get_delta_time(self) -> float:
        current_time = self.world.get_snapshot().timestamp.elapsed_seconds
        delta = current_time - self.battery_manager.last_update_time
        self.battery_manager.last_update_time = current_time
        return float(delta)

    def _get_elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        current_time = self.world.get_snapshot().timestamp.elapsed_seconds
        return float(current_time - self.start_time)

    def _is_terminal(self) -> bool:
        if self.battery_manager.get_remaining_distance() < 10:
            print("✅ Destination reached")
            self.has_arrived = True
            return True

        elapsed_time = self._get_elapsed_time()
        initial_eta = self.battery_manager.get_initial_eta()
        if elapsed_time > 2 * initial_eta:
            print("⏰ Timeout - Episode terminated")
            return True

        if self.battery_manager.get_soc() <= 0:
            print("🔋 Battery depleted")
            return True

        return False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        """Best-effort cleanup of CARLA actors and reset of world settings."""
        # Reset sync mode
        try:
            if self.world is not None:
                s = self.world.get_settings()
                if s.synchronous_mode:
                    s.synchronous_mode = False
                    s.fixed_delta_seconds = None
                    self.world.apply_settings(s)
        except Exception:
            pass

        # Destroy actors/managers
        try:
            if self.vehicle_manager is not None:
                self.vehicle_manager.destroy()
        except Exception:
            pass
        try:
            if self.dwcl_manager is not None:
                self.dwcl_manager.cleanup()
        except Exception:
            pass

    def close(self) -> None:
        self.cleanup()
        print("🧹 Environment cleaned up")
