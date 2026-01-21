"""
Custom Behavior Agent with DWCL-specific behaviors
"""
import sys
sys.path.insert(0, r"D:\WindowsNoEditor\PythonAPI")
sys.path.insert(0, r"D:\WindowsNoEditor\PythonAPI\carla")
"""
Custom Behavior Agent with DWCL-specific behaviors

Notes:
- Uses BehaviorAgent's public API (set_destination + run_step) to avoid relying on
  private/unstable methods like _trace_route, which may not exist in your CARLA build.
- Accepts both carla.Location and carla.Transform as navigation targets.
- Exposes a stable .vehicle property (BehaviorAgent typically stores the actor as _vehicle).
"""

import time
from typing import Union, Optional

import carla
from agents.navigation.behavior_agent import BehaviorAgent


TargetType = Union[carla.Location, carla.Transform]


class CustomBehaviorAgent(BehaviorAgent):
    """Extended Behavior Agent for DWCL scenarios"""

    def __init__(self, vehicle: carla.Vehicle, destination: carla.Location, config):
        """
        Args:
            vehicle: CARLA ego vehicle actor.
            destination: Main trip destination (carla.Location).
            config: Simulation/config object; should have target_speed ideally.
        """
        super().__init__(vehicle, behavior="normal")

        # Behavior customizations
        self._ignore_traffic_lights = True

        # Speed target (fallback to 40 km/h if not provided)
        self.target_speed: float = float(getattr(config, "target_speed", 40.0))

        # Planner tuning (safe-guarded for CARLA version differences)
        try:
            self._local_planner._min_waypoint_queue_length = 10
            self._local_planner._buffer_size = 20
        except Exception:
            # Some CARLA versions may structure planners differently; ignore if unavailable
            pass

        # Set initial destination (main trip)
        self.set_destination(destination)

    @property
    def vehicle(self) -> Optional[carla.Vehicle]:
        """
        CARLA BehaviorAgent typically stores the vehicle actor as self._vehicle.
        Provide a stable public accessor.
        """
        return getattr(self, "_vehicle", None)

    def navigate_to(
        self,
        target: TargetType,
        timeout: float = 8.0,
        reach_dist: float = 2.0,
        project_to_road: bool = True,
        lane_type: carla.LaneType = carla.LaneType.Driving,
        tick_sleep: float = 0.05,
    ) -> None:
        """
        Navigate to a target using BehaviorAgent's internal planner.

        Args:
            target: carla.Location or carla.Transform.
            timeout: seconds to attempt navigation before giving up.
            reach_dist: meters threshold to consider target reached.
            project_to_road: if True, project target to a valid driving waypoint.
            lane_type: lane type used for projection (default Driving).
            tick_sleep: sleep per loop (for async mode). If you run synchronous mode,
                        consider replacing this with world.tick() in your environment loop.
        """
        veh = self.vehicle
        if veh is None:
            raise RuntimeError("CustomBehaviorAgent has no vehicle assigned (self._vehicle is None).")

        # Normalize target to Location
        if isinstance(target, carla.Transform):
            target_loc = target.location
        elif isinstance(target, carla.Location):
            target_loc = target
        else:
            raise TypeError(f"navigate_to expected carla.Location or carla.Transform, got {type(target)}")

        world = veh.get_world()
        world_map = world.get_map()

        # Optionally project target to a drivable waypoint (prevents off-road targets)
        if project_to_road:
            try:
                wp = world_map.get_waypoint(
                    target_loc,
                    project_to_road=True,
                    lane_type=lane_type,
                )
                if wp is not None:
                    target_loc = wp.transform.location
            except Exception:
                # If waypoint projection fails, continue with the original target location
                pass

        # Take control from autopilot / Traffic Manager while we execute the maneuver
        veh.set_autopilot(False)

        # Let BehaviorAgent plan internally (version-stable approach)
        self.set_destination(target_loc)

        start_time = time.time()
        while (time.time() - start_time) < timeout:
            control = self.run_step()
            veh.apply_control(control)

            # Reached?
            if veh.get_location().distance(target_loc) <= reach_dist:
                break

            # For async mode, a small sleep avoids pegging CPU.
            # In synchronous mode, prefer ticking the world in your environment loop.
            if tick_sleep and tick_sleep > 0:
                time.sleep(tick_sleep)

        # Hand control back (optional: keep True if you rely on TM outside maneuvers)
        veh.set_autopilot(True)

    def run_step(self, debug: bool = False) -> carla.VehicleControl:
        """
        Override to handle missing waypoints gracefully.
        """
        try:
            return super().run_step(debug)
        except AttributeError:
            # Return neutral control on agent/planner attribute issues
            return carla.VehicleControl()
        except Exception:
            # Fail safe: neutral control
            return carla.VehicleControl()
