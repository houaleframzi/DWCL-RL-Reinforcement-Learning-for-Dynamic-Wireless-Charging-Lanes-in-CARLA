
import carla
import numpy as np
import math
from typing import List, Tuple, Optional
from config import constants as const


class DWCLManager:
    """Manages DWCL creation and queries"""
    
    def __init__(self, world: carla.World, config):
        self.world = world
        self.config = config
        self.all_corners = []
        self.start_location = None
        self.destination = None
        self.dwcl_length = const.DWCL_LENGTH
        self.dwcl_width = const.DWCL_WIDTH
        
    def create_dwcl(self, start_location: Optional[carla.Location] = None):
        """
        Create DWCL with charging coils and green lane markings
        """
        if start_location is None:
            # Default start location
            #spawn_point = self.world.get_map().get_spawn_points()[0]
            self.start_location = carla.Location(x=20, y=250, z=0.0)
        else:
            self.start_location = start_location
            
        # Get waypoint for the road
        carla_map = self.world.get_map()
        current_wp = carla_map.get_waypoint(
            self.start_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        if not current_wp:
            raise ValueError("Could not find valid road waypoint")
            
        self.start_transform = current_wp.transform #current_wp.get_right_lane().transform
        self.start_location = self.start_transform.location
        
        # Create destination
        self.destination = carla.Location(
            x=self.start_location.x + self.dwcl_length,
            y=self.start_location.y,
            z=self.start_location.z
        )
        
        # Create green lane markings
        self._draw_green_lane_markings()
        
        # Create coils
        self._create_coils()
        
        print(f"🔌 DWCL created from {self.start_location} to {self.destination}")
        
    def _draw_green_lane_markings(self):
        """Draw green lines under the charging lane"""
        print(f"🎨 Drawing green lane markings (length={self.dwcl_length}m, width={self.dwcl_width}m)")
        
        # Parameters for green lane markings
        num_lines = 200  # Number of parallel lines to draw
        line_spacing = self.dwcl_width / num_lines
        line_thickness = 0.05
        green_color = carla.Color(r=0, g=1, b=0, a=100)  # Semi-transparent green
        
        # Draw multiple parallel green lines to create a "plane" effect
        for i in range(num_lines):
            # Calculate y-offset to center lines in the lane
            y_offset = (i - num_lines / 2) * line_spacing
            
            # Start location for this line
            start_line_location = carla.Location(
                x=self.start_location.x,
                y=self.start_location.y + y_offset,
                z=self.start_location.z + 0.01  # Slightly above road to avoid z-fighting
            )
            
            # End location (full length ahead)
            end_line_location = carla.Location(
                x=start_line_location.x + self.dwcl_length,
                y=start_line_location.y,
                z=start_line_location.z
            )
            
            # Draw the green line
            self.world.debug.draw_line(
                start_line_location,
                end_line_location,
                thickness=line_thickness,
                color=green_color,
                life_time=0  # Persistent
            )
        
        # Also draw the lane boundaries
        self._draw_lane_boundaries()
        
        print(f"✅ Drew {num_lines} green lines for DWCL lane markings")
        
    def _draw_lane_boundaries(self):
        """Draw green boundaries around the DWCL lane"""
        # Bright green for boundaries
        boundary_color = carla.Color(r=0, g=1, b=0, a=0)
        boundary_thickness = 0.1
        
        # Calculate boundary positions
        half_width = self.dwcl_width / 2
        
        # Define the 4 corners of the DWCL lane
        corners = [
            # Top-left corner (looking from start)
            carla.Location(
                x=self.start_location.x,
                y=self.start_location.y + half_width,
                z=self.start_location.z + 0.02
            ),
            # Top-right corner
            carla.Location(
                x=self.start_location.x + self.dwcl_length,
                y=self.start_location.y + half_width,
                z=self.start_location.z + 0.02
            ),
            # Bottom-left corner
            carla.Location(
                x=self.start_location.x,
                y=self.start_location.y - half_width,
                z=self.start_location.z + 0.02
            ),
            # Bottom-right corner
            carla.Location(
                x=self.start_location.x + self.dwcl_length,
                y=self.start_location.y - half_width,
                z=self.start_location.z + 0.02
            )
        ]
        
        # Draw the boundary rectangle
        # Top boundary
        self.world.debug.draw_line(
            corners[0], corners[1],
            thickness=boundary_thickness,
            color=boundary_color,
            life_time=0
        )
        
        # Bottom boundary
        self.world.debug.draw_line(
            corners[2], corners[3],
            thickness=boundary_thickness,
            color=boundary_color,
            life_time=0
        )
        
        # Left boundary
        self.world.debug.draw_line(
            corners[0], corners[2],
            thickness=boundary_thickness,
            color=boundary_color,
            life_time=0
        )
        
        # Right boundary
        self.world.debug.draw_line(
            corners[1], corners[3],
            thickness=boundary_thickness,
            color=boundary_color,
            life_time=0
        )
        
        
        
    def _create_coils(self):
        """Create charging coils along the DWCL"""
        coil_width = const.COIL_WIDTH
        coil_height = const.COIL_HEIGHT
        coil_spacing = const.COIL_SPACING
        
        print(f"⚡ Creating charging coils (width={coil_width}m, spacing={coil_spacing}m)")
        
        # Calculate coil positions
        coil_positions = []
        current_pos = 5  # Start 5m from beginning
        coil_count = 0
        
        while current_pos < self.dwcl_length:
            coil_positions.append(current_pos)
            current_pos += coil_width + coil_spacing
            coil_count += 1
            
        # Colors for alternating coils
        colors = [
            carla.Color(r=1, g=1, b=0, a=0),   # Yellow
            carla.Color(r=1, g=1, b=1, a=0)   # White
        ]
        coil_color = colors[1]  # Start with white and the alter to have segmented array of coils, each 3 coils are connected
        # Create each coil
        for i, pos in enumerate(coil_positions):
            
            if (i+1)%3==0 and coil_color == colors[1]:
                coil_color = colors[0]
            elif (i+1)%3==0 and coil_color == colors[0]:
                coil_color = colors[1]
            
            # Calculate coil center
            coil_center_x = self.start_location.x + pos
            coil_center_y = self.start_location.y
            
            # Define corners (slightly above green lines)
            corners = [
                carla.Location(
                    x=coil_center_x - coil_width/2,
                    y=coil_center_y - coil_height/2,
                    z=self.start_location.z + 0.05  # Above the green lines
                ),
                carla.Location(
                    x=coil_center_x + coil_width/2,
                    y=coil_center_y - coil_height/2,
                    z=self.start_location.z + 0.05
                ),
                carla.Location(
                    x=coil_center_x + coil_width/2,
                    y=coil_center_y + coil_height/2,
                    z=self.start_location.z + 0.05
                ),
                carla.Location(
                    x=coil_center_x - coil_width/2,
                    y=coil_center_y + coil_height/2,
                    z=self.start_location.z + 0.05
                )
            ]
            self.all_corners.append(corners)
            
            # Draw coil rectangle
            self._draw_rectangle(corners, coil_color)
            
            # Draw coil ID/number
            self._draw_coil_label(coil_center_x, coil_center_y, i + 1)
            
           
        
        print(f"✅ Created {coil_count} charging coils")
        
    def _draw_rectangle(self, corners: List[carla.Location], color: carla.Color):
        """Draw rectangle for coil visualization"""
        # Draw all 4 sides of the rectangle
        for i in range(4):
            self.world.debug.draw_line(
                corners[i],
                corners[(i + 1) % 4],
                thickness=0.15,  # Thicker for better visibility
                color=color,
                life_time=0
            )
            
        # Fill the rectangle with a semi-transparent version
        fill_color = carla.Color(
            r=color.r,
            g=color.g,
            b=color.b,
            a=50  # More transparent
        )
        
        # Draw filled triangles to create a filled rectangle
        # Triangle 1: corners 0, 1, 2
        self.world.debug.draw_line(corners[0], corners[2], thickness=0.05, color=fill_color, life_time=0)
        # Triangle 2: corners[0], corners[2], corners[3]
        self.world.debug.draw_line(corners[0], corners[3], thickness=0.05, color=fill_color, life_time=0)
        
    def _draw_coil_label(self, x: float, y: float, coil_number: int):
        """Draw coil number label"""
        label_location = carla.Location(
            x=x,
            y=y,
            z=self.start_location.z + 0.1  # Above the coil
        )
        
        # Draw text label
        self.world.debug.draw_string(
            label_location,
            f"C{coil_number}",
            draw_shadow=True,
            color=carla.Color(r=0, g=0, b=0, a=255),  # Black text
            life_time=0,
            persistent_lines=True
        )
        
    
        
        
            
    def get_nearest_dwcl_location(self, vehicle_location: carla.Location) -> carla.Transform:
        """Get nearest DWCL location to vehicle"""
        if not self.all_corners:
            return self.start_transform
            
        min_distance = float('inf')
        nearest_transform = self.start_transform
        
        for corners in self.all_corners:
            dwcl_wp = self.world.get_map().get_waypoint(corners[0])
            distance = vehicle_location.distance(dwcl_wp.transform.location)
            
            if distance < min_distance:
                min_distance = distance
                nearest_transform = dwcl_wp.transform
                
        # Add safety distance
        velocity = carla.Vector3D(0, 0, 0)  # Default
        if hasattr(self, 'vehicle'):
            velocity = self.vehicle.get_velocity()
            
        speed_kmh = (velocity.x**2 + velocity.y**2)**0.5 * 3.6
        brake_distance = max((speed_kmh / 8)**2, 30)
        nearest_transform.location.x += brake_distance
        
        return nearest_transform


    def get_nearest_dwcl_location_forward(
        self,
        vehicle_transform: carla.Transform,
        min_ahead_distance: float = 2.0,
        allow_behind_fallback: bool = False,
    ) -> Optional[carla.Transform]:
        """Get nearest DWCL transform that is ahead of the vehicle (direction of travel).

        This prevents selecting a DWCL segment that the vehicle has already passed, which can
        cause the agent to attempt to turn back.

        Args:
            vehicle_transform: Current vehicle transform (used for forward vector).
            min_ahead_distance: Minimum forward projection (meters) to consider a DWCL ahead.
            allow_behind_fallback: If True, will fall back to the nearest DWCL even if behind.

        Returns:
            carla.Transform if a forward DWCL candidate exists; otherwise None (or fallback).
        """
        if not self.all_corners:
            return self.start_transform

        veh_loc = vehicle_transform.location
        fwd = vehicle_transform.get_forward_vector()
        fwd_xy_norm = math.sqrt(fwd.x * fwd.x + fwd.y * fwd.y) or 1.0
        fx, fy = fwd.x / fwd_xy_norm, fwd.y / fwd_xy_norm

        best = None
        best_dist = float("inf")

        # Track nearest overall (optional fallback)
        nearest_overall = None
        nearest_overall_dist = float("inf")

        carla_map = self.world.get_map()
        for corners in self.all_corners:
            try:
                wp = carla_map.get_waypoint(corners[0], project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp is None:
                    continue
                t = wp.transform
                dx = t.location.x - veh_loc.x
                dy = t.location.y - veh_loc.y
                ahead = dx * fx + dy * fy

                dist = math.sqrt(dx * dx + dy * dy)

                if dist < nearest_overall_dist:
                    nearest_overall_dist = dist
                    nearest_overall = t

                if ahead >= min_ahead_distance:
                    if dist < best_dist:
                        best_dist = dist
                        best = t
            except Exception:
                continue

        if best is not None:
            return best
        if allow_behind_fallback:
            return nearest_overall if nearest_overall is not None else self.start_transform
        return None

    def get_adjacent_lane(
        self,
        vehicle_location: Optional[carla.Location] = None,
        left: bool = True,
        min_ahead_distance: float = 2.0,
    ) -> Optional[carla.Transform]:
        """Get transform for adjacent lane near the vehicle (optionally forward-biased).

        Args:
            vehicle_location: Current vehicle location. If None, falls back to start location.
            left: If True, choose left lane; else right lane.
            min_ahead_distance: If >0, returns a point slightly ahead on the adjacent lane.

        Returns:
            carla.Transform of adjacent lane waypoint, or None if not available.
        """
        carla_map = self.world.get_map()
        base_loc = vehicle_location if vehicle_location is not None else self.start_location
        if base_loc is None:
            return None

        current_wp = carla_map.get_waypoint(
            base_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if current_wp is None:
            return None

        adjacent_wp = current_wp.get_left_lane() if left else current_wp.get_right_lane()
        if adjacent_wp is None:
            return None

        t = adjacent_wp.transform

        # Nudge forward along lane direction to reduce oscillations / going backward.
        if min_ahead_distance and min_ahead_distance > 0:
            fwd = t.get_forward_vector()
            norm = math.sqrt(fwd.x * fwd.x + fwd.y * fwd.y) or 1.0
            t.location.x += (fwd.x / norm) * min_ahead_distance
            t.location.y += (fwd.y / norm) * min_ahead_distance

        return t

    def get_adjacent_lane_forward(
        self,
        vehicle_transform: carla.Transform,
        left: bool = True,
        min_ahead_distance: float = 2.0,
    ) -> Optional[carla.Transform]:
        """Convenience wrapper: adjacent lane selection using vehicle transform."""
        return self.get_adjacent_lane(
            vehicle_location=vehicle_transform.location,
            left=left,
            min_ahead_distance=min_ahead_distance,
        )


    def get_adjacent_lane(self, left: bool = True) -> carla.Transform:
        """Get transform for adjacent lane"""
        carla_map = self.world.get_map()
        current_wp = carla_map.get_waypoint(
            self.start_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        if left:
            adjacent_wp = current_wp.get_left_lane()
        else:
            adjacent_wp = current_wp.get_right_lane()
            
        if adjacent_wp:
            return adjacent_wp.transform
        else:
            return current_wp.transform
            
    def is_on_dwcl(self, location: carla.Location) -> bool:
        """Check if location is within DWCL bounds"""
        if not self.all_corners:
            return False
            
        # Get bounds from first and last coil
        first_corners = self.all_corners[0]
        last_corners = self.all_corners[-1]
        
        min_x = min(c.x for c in first_corners) - 1.0  # Add 1m buffer
        max_x = max(c.x for c in last_corners) + 1.0   # Add 1m buffer
        min_y = min(c.y for c in first_corners) - self.dwcl_width/2
        max_y = max(c.y for c in first_corners) + self.dwcl_width/2
        
        return (min_x <= location.x <= max_x and 
                min_y <= location.y <= max_y)
    
    def get_all_corners(self) -> List[List[carla.Location]]:
        """Get all coil corners"""
        return self.all_corners
    
    def get_destination(self) -> carla.Location:
        """Get destination location"""
        return self.destination
    
    def get_start_transform(self) -> carla.Transform:
        """Get start transform"""
        return self.start_transform
    
    def get_dwcl_bounds(self) -> Tuple[float, float, float, float]:
        """Get DWCL bounds as (min_x, max_x, min_y, max_y)"""
        if not self.all_corners:
            return (0, 0, 0, 0)
            
        all_x = []
        all_y = []
        for corners in self.all_corners:
            for corner in corners:
                all_x.append(corner.x)
                all_y.append(corner.y)
                
        return (min(all_x) - 1.0, max(all_x) + 1.0, 
                min(all_y) - self.dwcl_width/2, max(all_y) + self.dwcl_width/2)
    
    def cleanup(self):
        """Clean up DWCL resources"""
        self.all_corners.clear()
        print("🧹 DWCL cleaned up")