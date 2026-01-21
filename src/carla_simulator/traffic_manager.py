"""
Traffic Manager wrapper for vehicle control
"""
import carla


class TrafficManager:
    """Wrapper for CARLA Traffic Manager"""
    
    def __init__(self, client: carla.Client, vehicle: carla.Vehicle, port: int = 8000):
        self.client = client
        self.vehicle = vehicle
        self.tm = self.client.get_trafficmanager(port)
        
        # Configure traffic manager
        self._configure_traffic_manager()
        
    def _configure_traffic_manager(self):
        """Configure traffic manager settings"""
        self.tm.ignore_lights_percentage(self.vehicle, 100)
        self.tm.ignore_signs_percentage(self.vehicle, 100)
        self.tm.auto_lane_change(self.vehicle, False)
        self.tm.set_global_distance_to_leading_vehicle(2.0)
        self.tm.set_synchronous_mode(True)
        
    def set_vehicle_speed(self, speed_kmh: float):
        """Set vehicle target speed"""
        # Get current speed limit
        speed_limit = self.vehicle.get_speed_limit()
        if speed_limit <= 0:
            return
            
        # Calculate percentage difference
        ratio = speed_kmh / speed_limit
        percentage = 100.0 * (1.0 - ratio)
        
        # Apply to traffic manager
        self.tm.vehicle_percentage_speed_difference(self.vehicle, percentage)
        
    def set_autopilot(self, enable: bool = True):
        """Enable/disable autopilot"""
        self.vehicle.set_autopilot(enable, self.tm.get_port())
        
    def disable_autopilot(self):
        """Completely disable autopilot"""
        try:
            self.tm.unregister_vehicle(self.vehicle)
            self.vehicle.set_autopilot(False)
        except Exception as e:
            print(f"Warning: Could not unregister vehicle: {e}")