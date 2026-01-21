"""
Battery management and state estimation
"""
import numpy as np
import pandas as pd
import math
import carla
from typing import Tuple, Optional, List
from config import constants as const


class BatteryManager:
    """Manages battery state and energy calculations"""
    
    def __init__(self, vehicle, destination, world, power_model, C_bat, P_aux, all_corners):
        self.vehicle = vehicle
        self.destination = destination
        self.world = world
        self.power_model = power_model
        self.C_bat = C_bat
        self.P_aux = P_aux
        self.all_corners = all_corners
        
        # State variables
        self.soc = const.SOC_INITIAL  # State of Charge (%)
        self.total_energy = 0.0  # kWh
        self.last_update_time = self._get_simulation_time()
        self.prev_velocity = self.vehicle.get_velocity()
        self.prev_location = self.vehicle.get_location()
        
        # Derived metrics
        self.remaining_distance = self._compute_remaining_distance()
        self.initial_eta = self.remaining_distance / const.AVG_SPEED
        self.eta = self.initial_eta
        self.soc_required = 0.0
        self.in_dwcl = self._is_on_dwcl()
        
        # Charging state
        self.charging_power = 0.0
        self.coupling_coefficient = 0.0
        self.alignment_factor = 0.0
        self.transfer_efficiency = 0.0
        
    def update(self, delta_time: float):
        """
        Update battery state based on current conditions
        """
        # Get current measurements
        location = self.vehicle.get_location()
        velocity = self.vehicle.get_velocity()
        speed_ms = (velocity.x**2 + velocity.y**2)**0.5
        acceleration = self._get_acceleration(delta_time)
        slope = self._get_slope()
        
        # Compute forces and power
        forces = self._compute_forces(speed_ms, acceleration, slope)
        power_net, power_req = self._compute_power(forces)
        
        # Handle negative or zero speed
        if speed_ms <= 0:
            power_net = 0
            
        # Total power consumption
        power_total = delta_time * (self.P_aux + power_net) / 3600  # kWh
        
        # Update remaining distance
        self.remaining_distance = self._compute_remaining_distance()
        
        # Energy requirements
        energy_required = (self.remaining_distance / const.AVG_SPEED) / 3600 * (self.P_aux + power_req)
        self.soc_required = (energy_required / self.C_bat) * 100
        
        # Check charging
        on_coil, coil_center, coil_orientation, coil_length = self._is_on_charging_coil()
        
        if on_coil:
            # Compute charging parameters
            self.alignment_factor, self.coupling_coefficient = self._get_alignment_factor(
                coil_center, coil_orientation, coil_length
            )
            self.transfer_efficiency = self._get_power_transfer_efficiency(self.coupling_coefficient)
            
            # Compute charging power
            self.charging_power = const.P_EV_MAX * self.transfer_efficiency * const.ETA_CH
            energy_charged = (self.charging_power * delta_time) / 3600
        else:
            self.charging_power = 0
            self.coupling_coefficient = 0
            self.alignment_factor = 0
            self.transfer_efficiency = 0
            energy_charged = 0
            
        # Update battery state
        energy_net = power_total - energy_charged
        self.total_energy += energy_net
        self.soc -= (energy_net / self.C_bat) * 100
        
        # Clamp SOC between 0 and 100
        self.soc = max(0, min(100, self.soc))
        self.total_energy = max(0, self.total_energy)
        
        # Update ETA
        self.eta = self.remaining_distance / (const.AVG_SPEED + 0.01)
        
        # Update DWCL status
        self.in_dwcl = self._is_on_dwcl()
        
        # Update previous values
        self.prev_velocity = velocity
        self.prev_location = location
        self.last_update_time = self._get_simulation_time()
        
        return self._get_state()
    
    def _get_state(self) -> dict:
        """Get current battery state"""
        return {
            'soc': self.soc,
            'soc_required': self.soc_required,
            'total_energy': self.total_energy,
            'eta': self.eta,
            'charging_power': self.charging_power,
            'alignment_factor': self.alignment_factor,
            'coupling_coefficient': self.coupling_coefficient,
            'transfer_efficiency': self.transfer_efficiency,
            'remaining_distance': self.remaining_distance,
            'in_dwcl': self.in_dwcl
        }
    
    def _compute_forces(self, speed_ms: float, acceleration: float, slope: float) -> Tuple:
        """Compute forces acting on vehicle"""
        # Convert to vectors for averaging
        speed_vector = np.array([speed_ms, const.AVG_SPEED])
        acc_vector = np.array([acceleration, const.AVG_ACCELERATION])
        slope_vector = np.array([slope, const.AVG_SLOPE])
        
        # Rolling resistance
        F_roll = const.ROLLING_RESISTANCE * const.VEHICLE_MASS * const.GRAVITY * speed_vector / 1000
        
        # Climbing force
        F_climb = const.VEHICLE_MASS * const.GRAVITY * np.sin(slope_vector) * speed_vector / 1000
        
        # Aerodynamic drag
        F_aero = 0.5 * const.DRAG_COEFFICIENT * const.FRONTAL_AREA * const.AIR_DENSITY * speed_vector**2 * speed_vector / 1000
        
        # Acceleration force
        F_acc = const.VEHICLE_MASS * acc_vector * speed_vector / 1000
        
        return F_roll, F_climb, F_aero, F_acc
    
    def _compute_power(self, forces: Tuple) -> Tuple[float, float]:
        """Compute net power and required power"""
        F_roll, F_climb, F_aero, F_acc = forces
        
        # Create DataFrame for power model
        df = pd.DataFrame({
            'aeroF_P': F_aero,
            'climbF_P': F_climb,
            'rollF_P': F_roll,
            'accF_P': F_acc
        })
        
        # Predict power using model
        try:
            predictions = self.power_model.predict(df, verbose=0).flatten()
            P_net, P_req = -predictions[0], predictions[1]
        except:
            # Fallback calculation
            total_force = F_roll.mean() + F_climb.mean() + F_aero.mean() + F_acc.mean()
            P_net = total_force / const.TRANSMISSION_EFFICIENCY
            P_req = P_net * 1.2  # 20% safety margin
            
        return P_net, P_req
    
    def _is_on_charging_coil(self) -> Tuple[bool, Optional[carla.Location], Optional[float], Optional[float]]:
        """Check if vehicle is on a charging coil"""
        epsilon = 0.25
        vehicle_location = self.vehicle.get_location()
        vehicle_x, vehicle_y = vehicle_location.x, vehicle_location.y
        
        for corners in self.all_corners:
            # Get coil bounds
            x_min = min(c.x for c in corners) - epsilon
            x_max = max(c.x for c in corners) + epsilon
            y_min = min(c.y for c in corners) - epsilon
            y_max = max(c.y for c in corners) + epsilon
            
            if (x_min <= vehicle_x <= x_max) and (y_min <= vehicle_y <= y_max):
                # Calculate coil parameters
                center_x = sum(c.x for c in corners) / 4
                center_y = sum(c.y for c in corners) / 4
                dx = corners[1].x - corners[0].x
                dy = corners[1].y - corners[0].y
                coil_length = math.sqrt(dx**2 + dy**2)
                yaw_angle = math.degrees(math.atan2(dy, dx))
                
                coil_center = carla.Location(center_x, center_y, corners[0].z)
                return True, coil_center, yaw_angle, coil_length
                
        return False, None, None, None
    
    def _get_alignment_factor(self, coil_center: carla.Location, 
                            coil_orientation: float, coil_length: float) -> Tuple[float, float]:
        """Compute alignment factor and coupling coefficient"""
        vehicle_location = self.vehicle.get_location()
        vehicle_rotation = self.vehicle.get_transform().rotation.yaw
        
        # Lateral misalignment
        d_y = abs(vehicle_location.y - coil_center.y)
        f_lat = np.exp(-(d_y / const.D_MAX) ** 2)
        
        # Longitudinal misalignment
        d_x = abs(vehicle_location.x - coil_center.x)
        f_long = np.exp(-(d_x / coil_length) ** 2)
        
        # Angular misalignment
        theta_misalign = abs(vehicle_rotation - coil_orientation)
        theta_rad = math.radians(theta_misalign)
        f_ang = abs(np.cos(theta_rad) + 0.5 * np.sin(theta_rad))
        
        # Combined alignment factor
        f_align = f_lat * f_long * f_ang
        
        # Coupling coefficient
        k = const.K0 * f_align
        
        return f_align, k
    
    def _get_power_transfer_efficiency(self, k: float) -> float:
        """Compute power transfer efficiency"""
        return (k**2 * const.Q1 * const.Q2) / (1 + k**2 * const.Q1 * const.Q2)
    
    def _get_acceleration(self, delta_time: float) -> float:
        """Calculate current acceleration"""
        if delta_time <= 0:
            return 0
            
        current_velocity = self.vehicle.get_velocity()
        current_speed = (current_velocity.x**2 + current_velocity.y**2)**0.5
        prev_speed = (self.prev_velocity.x**2 + self.prev_velocity.y**2)**0.5
        
        acceleration = (current_speed - prev_speed) / delta_time if self.prev_velocity else 0
        return acceleration
    
    def _get_slope(self) -> float:
        """Get road slope from vehicle rotation"""
        rotation = self.vehicle.get_transform().rotation
        return math.radians(rotation.pitch)
    
    def _compute_remaining_distance_old(self) -> float:
        """Compute remaining distance to destination"""
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner
            
            carla_map = self.world.get_map()
            grp = GlobalRoutePlanner(carla_map, 2.0)
            route = grp.trace_route(self.vehicle.get_location(), self.destination)
            
            total_distance = 0
            for i in range(len(route) - 1):
                loc1 = route[i][0].transform.location
                loc2 = route[i + 1][0].transform.location
                total_distance += loc1.distance(loc2)
                
            return total_distance
        except:
            # Fallback: Euclidean distance
            current_loc = self.vehicle.get_location()
            return current_loc.distance(self.destination)
        
    def _compute_remaining_distance(self) -> float:
        """
        Compute remaining distance to destination using Euclidean distance.
        """
        try:
            current_loc = self.vehicle.get_location()
            return current_loc.distance(self.destination)
        except Exception:
            return 0.0

    
    def _is_on_dwcl(self) -> bool:
        """Check if vehicle is on DWCL lane"""
        vehicle_location = self.vehicle.get_location()
        
        # Simple check using coil bounds
        if not self.all_corners:
            return False
            
        # Get DWCL bounds from first and last coil
        first_corners = self.all_corners[0]
        last_corners = self.all_corners[-1]
        
        min_x = min(c.x for c in first_corners)
        max_x = max(c.x for c in last_corners)
        min_y = min(c.y for c in first_corners) - const.DWCL_WIDTH/2
        max_y = max(c.y for c in first_corners) + const.DWCL_WIDTH/2
        
        return (min_x <= vehicle_location.x <= max_x and 
                min_y <= vehicle_location.y <= max_y)
    
    def _get_simulation_time(self) -> float:
        """Get current simulation time"""
        return self.world.get_snapshot().timestamp.elapsed_seconds
    
    # Public API methods
    def get_soc(self) -> float:
        return self.soc
    
    def get_required_soc(self) -> float:
        return self.soc_required
    
    def get_eta(self) -> float:
        return self.eta
    
    def get_initial_eta(self) -> float:
        return self.initial_eta
    
    def get_remaining_distance(self) -> float:
        return self.remaining_distance
    
    def is_on_dwcl(self) -> bool:
        return self.in_dwcl
    
    def reset(self):
        """Reset battery state"""
        self.soc = const.SOC_INITIAL
        self.total_energy = 0.0
        self.last_update_time = self._get_simulation_time()
        self.prev_velocity = self.vehicle.get_velocity()
        self.prev_location = self.vehicle.get_location()
        self.remaining_distance = self._compute_remaining_distance()
        self.initial_eta = self.remaining_distance / const.AVG_SPEED
        self.eta = self.initial_eta
        self.soc_required = 0.0
        self.in_dwcl = self._is_on_dwcl()
        self.charging_power = 0.0
        self.coupling_coefficient = const.K0
        self.alignment_factor = 0.0
        self.transfer_efficiency = 0.0