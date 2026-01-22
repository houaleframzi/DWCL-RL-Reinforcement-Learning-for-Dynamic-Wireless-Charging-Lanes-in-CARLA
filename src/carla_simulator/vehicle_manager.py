"""
Vehicle management for CARLA simulation
"""
import carla
import random
from typing import Optional, Tuple
from config import constants as const



class VehicleManager:
    """Manages vehicle spawning and control"""
    
    def __init__(self, world: carla.World):
        self.world = world
        self.vehicle = None
        self.vehicle_blueprint = None
        
    def spawn_vehicle(self, location: Optional[Tuple[float, float, float]] = None):
        """
        Spawn vehicle at a specified location or at a default DWCL-compatible location.
        Default location: (22.6, 251.0, 0.0)
        """

        # -----------------------------
        # Vehicle blueprint
        # -----------------------------
        blueprint_lib = self.world.get_blueprint_library()
        vehicle_bps = blueprint_lib.filter("vehicle.nissan.*")

        if not vehicle_bps:
            raise RuntimeError("No Nissan vehicle blueprint found")

        self.vehicle_blueprint = vehicle_bps[0]  # Nissan Micra (deterministic)

        # -----------------------------
        # Spawn location (default or user-defined)
        # -----------------------------
        if location is None:
            spawn_location = carla.Location(x=22.6, y=251.0, z=0.0)
        else:
            spawn_location = carla.Location(*location)

        # Project to closest drivable waypoint
        carla_map = self.world.get_map()
        current_wp = carla_map.get_waypoint(
            spawn_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if current_wp is None:
            raise RuntimeError("Failed to find a drivable waypoint for spawn location")

        spawn_transform = current_wp.transform

        # -----------------------------
        # Spawn vehicle
        # -----------------------------
        self.vehicle = self.world.try_spawn_actor(
            self.vehicle_blueprint,
            spawn_transform
        )

        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle actor")

        print(f"🚗 Vehicle spawned at ({spawn_transform.location.x:.2f}, "
            f"{spawn_transform.location.y:.2f}, "
            f"{spawn_transform.location.z:.2f})")

        

        return self.vehicle
    
    def reset_vehicle(self, transform: carla.Transform):
        """Reset vehicle to specified transform"""
        if self.vehicle:
            self.vehicle.set_transform(transform)
            self.stop_vehicle()
            
    def stop_vehicle(self):
        """Apply full brake to stop vehicle"""
        if self.vehicle:
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.brake = 1.0
            control.steer = 0.0
            self.vehicle.apply_control(control)
            
    def get_velocity_kmh(self) -> float:
        """Get vehicle speed in km/h"""
        if not self.vehicle:
            return 0.0
            
        velocity = self.vehicle.get_velocity()
        speed_ms = (velocity.x**2 + velocity.y**2)**0.5
        return speed_ms * 3.6
    
    def get_location(self) -> carla.Location:
        """Get vehicle location"""
        return self.vehicle.get_location() if self.vehicle else None
    
    def destroy(self):
        """Destroy vehicle actor"""
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None
            print("🗑️ Vehicle destroyed")
