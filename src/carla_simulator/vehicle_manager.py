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
        Spawn vehicle at specified location or random spawn point
        """
        # Get blueprint
        blueprint_lib = self.world.get_blueprint_library()
        vehicle_bps = blueprint_lib.filter('nissan')
        
        if not vehicle_bps:
            raise ValueError("No vehicle blueprint found")
            
        self.vehicle_blueprint = vehicle_bps[0]  # Use the first available blueprint the nissam micra or #random.choice(vehicle_bps)
        
        # Set spawn location
        if location:
            spawn_location = carla.Location(*location)
        else:
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise ValueError("No spawn points available")
            spawn_location = random.choice(spawn_points).location


        current_wp = self.world.get_map().get_waypoint(
            spawn_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
            
        # Create transform
        #spawn_transform = carla.Transform(
        #    location=spawn_location,
        #    rotation=carla.Rotation()
        #)
        
        # Spawn vehicle
        self.vehicle = self.world.try_spawn_actor(
            self.vehicle_blueprint,
            current_wp.transform
        )
        
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle")
            
        print(f"🚗 Vehicle spawned at {spawn_location}")
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