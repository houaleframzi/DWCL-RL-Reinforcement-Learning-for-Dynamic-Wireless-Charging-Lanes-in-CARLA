"""
Interactive PyGame visualization for DWCL simulation with manual control
Enhanced with better visuals and more data display
"""
import pygame
import sys
import os
import numpy as np
import carla
import cv2
import math
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from keras.models import load_model

import config.constants as const
import config.settings as settings

from src.carla_simulator.carla_env import CarlaEnv



class DWCLPygameVisualizer:
    """
    Interactive PyGame visualization for DWCL simulation
    Enhanced visual design with more data and better layout
    """
    
    def __init__(self, width=1600, height=1000):
        self.width = width
        self.height = height
        
        # Initialize PyGame
        pygame.init()
        pygame.font.init()
        
        # Create window
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("DWCL Interactive Simulator - Manual Control")
        
        # Fonts - More professional
        self.title_font = pygame.font.SysFont('Segoe UI', 25, bold=True)
        self.header_font = pygame.font.SysFont('Segoe UI', 18, bold=True)
        self.data_font = pygame.font.SysFont('Consolas', 15)
        self.small_font = pygame.font.SysFont('Segoe UI', 15)
        self.value_font = pygame.font.SysFont('Consolas', 15, bold=True)
        
        # Modern color palette
        self.colors = {
            'background': (15, 20, 30),
            'panel': (30, 35, 45),
            'panel_dark': (25, 30, 40),
            'border': (60, 70, 100),
            'border_light': (80, 90, 120),
            'text': (220, 230, 240),
            'text_dim': (180, 190, 200),
            'warning': (255, 100, 100),
            'success': (100, 255, 150),
            'charging': (255, 215, 0),
            'battery_low': (255, 80, 80),
            'battery_med': (255, 180, 50),
            'battery_high': (80, 220, 100),
            'dwcl_active': (0, 200, 255),
            'speed': (100, 200, 255),
            'acceleration': (255, 150, 100),
            'slope': (150, 255, 150),
            'power': (255, 100, 255),
            'panel_accent': (40, 100, 180),
            'button': (60, 120, 200),
            'button_hover': (80, 150, 240),
            'gauge_bg': (40, 45, 55),
            'gauge_fill': (0, 180, 255),
        }
        
        # Simulation components
        self.env: Optional[CarlaEnv] = None
        self.client = None
        self.world = None
        self.vehicle = None
        self.battery_manager = None
        self.traffic_manager = None
        self.dwcl_manager = None
        self.camera = None
        
        # Control state
        self.manual_mode = True
        self.lane_assist_active = False
        self.lane_assist_target_in_dwcl = None  # type: Optional[bool]
        self.control = carla.VehicleControl()
        self.keys_pressed = set()
        self.last_key_time = {}
        
        # Camera
        self.camera_image = None
        self.camera_width = int(width * 0.7)
        self.camera_height = height  # Match full window height
        
        # Data logging
        self.data_log = []
        self.log_file = None
        self.start_time = None
        
        # UI State
        self.show_help = False
        self.show_debug = False
        self.recording = False
        self.paused = False
        
        # Performance
        self.clock = pygame.time.Clock()
        self.fps = 30
        self.last_update = 0
        
        # Data tracking
        self.speed_history = []
        self.acceleration_history = []
        self.max_history_length = 100
        
        # HUD elements
        self.hud_elements = []
        
        print("DWCL Interactive Simulator Initialized")
        
    def initialize_simulator(self, power_model_path: str):
        """Initialize CARLA simulation components"""
        try:
            # Load power model
            print("Loading power model...")
            power_model = load_model(power_model_path)

            # Initialize CARLA via the shared environment (single source of truth)
            print("Initializing CARLA environment...")
            sim_config = settings.SimulationSettings()
            # Ensure rendering in interactive play mode
            setattr(sim_config, "no_rendering_mode", False)

            self.env = CarlaEnv(sim_config)
            # Deterministic start similar to the old play script
            self.env.initialize_simulation(power_model, spawn_location=(20.0, 250.0, 0.0))

            # Bind references
            self.client = self.env.client
            self.world = self.env.world
            self.dwcl_manager = self.env.dwcl_manager
            self.traffic_manager = self.env.traffic_manager
            self.battery_manager = self.env.battery_manager
            self.vehicle = self.env.vehicle_manager.vehicle

            # Default to manual control (no autopilot)
            try:
                self.traffic_manager.set_autopilot(False)
            except Exception:
                pass
            
            # Attach camera
            print("Attaching camera...")
            self._attach_camera()
            
            # Set spectator
            self._set_spectator()
            
            # Initialize logging
            self._init_logging()
            
            print("Simulation initialized successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to initialize simulation: {e}")
            return False
            
    def _attach_camera(self):
        """Attach a camera to the vehicle"""
        # Camera blueprint
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.camera_width))
        camera_bp.set_attribute('image_size_y', str(self.camera_height))
        camera_bp.set_attribute('fov', '100')
        
        # Camera transform (behind and above vehicle)
        camera_transform = carla.Transform(
            carla.Location(x=-6.0, z=3.5),
            carla.Rotation(pitch=-15.0)
        )
        
        # Spawn camera
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )
        
        # Listen to camera
        self.camera.listen(lambda image: self._process_camera_image(image))
        
    def _process_camera_image(self, image):
        """Fixed camera processing"""
        # Get raw data
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        
        # Reshape to image dimensions
        array = array.reshape((image.height, image.width, 4))
        
        # Convert BGRA to RGB
        rgb_array = array[:, :, [2, 1, 0]]
        
        # Rotate for correct orientation
        rgb_array = np.rot90(rgb_array)
        
        # Create PyGame surface
        self.camera_image = pygame.surfarray.make_surface(rgb_array)
        
    def _set_spectator(self):
        """Set spectator view"""
        spectator = self.world.get_spectator()
        transform = self.vehicle.get_transform()
        transform.location.z += 10
        transform.rotation.pitch = -90
        spectator.set_transform(transform)
        
    def _init_logging(self):
        """Initialize data logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "data/logs"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"manual_session_{timestamp}.csv")
        
        # Write header
        with open(self.log_file, 'w') as f:
            f.write("timestamp,soc,required_soc,speed,acceleration,slope,"
                   "charging_power,alignment_factor,coupling_coeff,transfer_eff,"
                   "in_dwcl,remaining_distance,throttle,brake,steer\n")
            
        self.start_time = pygame.time.get_ticks()
        print(f"Logging data to: {self.log_file}")
        
    def handle_events(self):
        """Handle PyGame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.KEYDOWN:
                self.keys_pressed.add(event.key)
                self.last_key_time[event.key] = pygame.time.get_ticks()
                
                # Toggle manual/autopilot
                if event.key == pygame.K_m:
                    self.manual_mode = not self.manual_mode
                    if self.manual_mode:
                        print("Manual mode enabled")
                        self.traffic_manager.set_autopilot(False)
                    else:
                        print("Autopilot enabled")
                        self.traffic_manager.set_autopilot(True)
                        
                # Toggle help
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                    
                # Toggle debug
                elif event.key == pygame.K_d:
                    self.show_debug = not self.show_debug
                    
                # Toggle pause
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                    print(f"{'Paused' if self.paused else 'Resumed'}")
                    
                # Toggle recording
                elif event.key == pygame.K_r:
                    self.recording = not self.recording
                    print(f"{'Started' if self.recording else 'Stopped'} recording")
                    
                # Reset vehicle
                elif event.key == pygame.K_t:
                    start_transform = self.dwcl_manager.get_start_transform()
                    self.vehicle.set_transform(start_transform)
                    self.control = carla.VehicleControl()
                    print("Vehicle reset to start position")

                # Lane assist: enter/exit DWCL
                elif event.key == pygame.K_1:
                    self.request_enter_dwcl()

                elif event.key == pygame.K_2:
                    self.request_exit_dwcl()
                    
                # Take screenshot
                elif event.key == pygame.K_s:
                    self._take_screenshot()
                    
                # Quit
                elif event.key == pygame.K_ESCAPE:
                    return False
                    
            elif event.type == pygame.KEYUP:
                if event.key in self.keys_pressed:
                    self.keys_pressed.remove(event.key)
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check for button clicks
                mouse_pos = pygame.mouse.get_pos()
                self._handle_button_clicks(mouse_pos)
                
        return True
        
    def _handle_button_clicks(self, mouse_pos):
        """Handle button clicks in UI"""
        for element in self.hud_elements:
            if element.get('type') == 'button' and element.get('rect'):
                if element['rect'].collidepoint(mouse_pos):
                    callback = element.get('callback')
                    if callback:
                        callback()

    def request_enter_dwcl(self) -> None:
        """High-level command: temporarily hand off control to the behavior agent to enter the DWCL."""
        if self.env is None or self.battery_manager is None:
            return
        try:
            if bool(self.battery_manager.is_on_dwcl()):
                return  # already on DWCL
        except Exception:
            pass

        self.lane_assist_active = True
        self.lane_assist_target_in_dwcl = True

        # Avoid controller conflicts: disable manual input and traffic autopilot
        self.manual_mode = False
        try:
            self.traffic_manager.set_autopilot(False)
        except Exception:
            pass

        try:
            self.env.move_to_dwcl()
            print("Lane assist: entering DWCL")
        except Exception as e:
            print(f"Lane assist: failed to enter DWCL ({e})")
            self.lane_assist_active = False
            self.manual_mode = True

    def request_exit_dwcl(self) -> None:
        """High-level command: temporarily hand off control to the behavior agent to exit the DWCL."""
        if self.env is None or self.battery_manager is None:
            return
        try:
            if not bool(self.battery_manager.is_on_dwcl()):
                return  # already off DWCL
        except Exception:
            pass

        self.lane_assist_active = True
        self.lane_assist_target_in_dwcl = False

        self.manual_mode = False
        try:
            self.traffic_manager.set_autopilot(False)
        except Exception:
            pass

        try:
            self.env.exit_dwcl()
            print("Lane assist: exiting DWCL")
        except Exception as e:
            print(f"Lane assist: failed to exit DWCL ({e})")
            self.lane_assist_active = False
            self.manual_mode = True

    def _update_lane_assist(self) -> None:
        """Return control back to the user when the lane-assist target has been reached."""
        if not self.lane_assist_active or self.lane_assist_target_in_dwcl is None:
            return

        try:
            in_dwcl = bool(self.battery_manager.is_on_dwcl())
        except Exception:
            return

        if in_dwcl == self.lane_assist_target_in_dwcl:
            self.lane_assist_active = False
            self.lane_assist_target_in_dwcl = None
            self.manual_mode = True
            print("Lane assist complete: manual control restored")
                        
    def update_controls(self):
        """Update vehicle controls based on key presses"""
        if self.paused:
            return

        # While lane assist is active, avoid applying manual controls to prevent
        # conflicts with the behavior agent.
        if self.lane_assist_active:
            return
            
        if self.manual_mode:
            # Reset control
            self.control.throttle = 0.0
            self.control.brake = 0.0
            self.control.steer = 0.0
            
            # Throttle
            if pygame.K_UP in self.keys_pressed:
                self.control.throttle = 0.7
                
            # Brake
            if pygame.K_DOWN in self.keys_pressed:
                self.control.brake = 1.0
                
            # Steering
            if pygame.K_LEFT in self.keys_pressed:
                self.control.steer = 0.4
            elif pygame.K_RIGHT in self.keys_pressed:
                self.control.steer = -0.4
                
            # Handbrake
            self.control.hand_brake = pygame.K_SPACE in self.keys_pressed
            
            # Reverse
            self.control.reverse = pygame.K_b in self.keys_pressed
            
            # Apply control
            self.vehicle.apply_control(self.control)
        else:
            # In autopilot mode, let traffic manager handle control
            pass
            
    def update_simulation(self):
        """Update simulation state"""
        if self.paused:
            return
            
        # Tick the world
        self.world.tick() # we use sync mode, the simulation gets updated here
        
        # Update battery
        current_time = self.world.get_snapshot().timestamp.elapsed_seconds
        delta_time = current_time - self.battery_manager.last_update_time
        battery_state = self.battery_manager.update(delta_time)

        # If lane assist is active, check whether the target lane has been reached
        self._update_lane_assist()
        #print(delta_time)
        
        # Update data history
        speed = self._get_speed_kmh()
        self.speed_history.append(speed)
        if len(self.speed_history) > self.max_history_length:
            self.speed_history.pop(0)
            
        acceleration = self._get_acceleration(delta_time)
        #print(acceleration)
        self.acceleration_history.append(acceleration)
        if len(self.acceleration_history) > self.max_history_length:
            self.acceleration_history.pop(0)
        
        # Log data if recording
        if self.recording:
            self._log_data(speed, acceleration)
            
    def _get_acceleration(self, delta_time = None) -> float:
        """Calculate current acceleration"""
        velocity = self.vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x**2 + velocity.y**2)
        
        if len(self.speed_history) > 1:
            # Simple differentiation for acceleration
            prev_speed_ms = self.speed_history[-2] / 3.6 if len(self.speed_history) >= 2 else 0
            current_speed_ms = speed_ms
            #delta_time = 0.1  # Assuming 20 Hz update
            if delta_time is None:
                delta_time = 0.1 # Default value used to plot acceleration when not provided

            #print("current_time:",current_time," last_update_time:",self.battery_manager.last_update_time)
            if delta_time > 0:
                return (current_speed_ms - prev_speed_ms) / delta_time
        return 0.0
    
    def _get_slope(self) -> float:
        """Get road slope in degrees"""
        rotation = self.vehicle.get_transform().rotation
        return rotation.pitch  # Already in degrees
        
    def _log_data(self, speed: float, acceleration: float):
        """Log current simulation data"""
        battery_state = self.battery_manager._get_state()
        
        timestamp = (pygame.time.get_ticks() - self.start_time) / 1000.0
        slope = self._get_slope()
        
        log_line = (f"{timestamp:.2f},"
                   f"{battery_state['soc']:.2f},"
                   f"{battery_state['soc_required']:.2f},"
                   f"{speed:.2f},"
                   f"{acceleration:.2f},"
                   f"{slope:.2f},"
                   f"{battery_state['charging_power']:.2f},"
                   f"{battery_state['alignment_factor']:.3f},"
                   f"{battery_state['coupling_coefficient']:.3f},"
                   f"{battery_state['transfer_efficiency']:.3f},"
                   f"{int(battery_state['in_dwcl'])},"
                   f"{battery_state['remaining_distance']:.2f},"
                   f"{self.control.throttle:.2f},"
                   f"{self.control.brake:.2f},"
                   f"{self.control.steer:.2f}\n")
        
        with open(self.log_file, 'a') as f:
            f.write(log_line)
            
    def _get_speed_kmh(self) -> float:
        """Get vehicle speed in km/h"""
        velocity = self.vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x**2 + velocity.y**2)
        return speed_ms * 3.6
        
    def render(self):
        """Render everything to PyGame screen"""
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw camera view - now full height
        if self.camera_image:
            camera_rect = pygame.Rect(0, 0, self.camera_width, self.camera_height)
            self.screen.blit(self.camera_image, camera_rect)
            
        # Draw HUD panels
        self._render_hud_panels()
        
        # Draw help if enabled
        if self.show_help:
            self._render_help_overlay()
            
        # Draw debug info if enabled
        if self.show_debug:
            self._render_debug_info()
            
        # Update display
        pygame.display.flip()
        
    def _render_hud_panels(self):
        """Render HUD panels with simulation data"""
        battery_state = self.battery_manager._get_state()
        speed = self._get_speed_kmh()
        acceleration = self._get_acceleration()
        slope = self._get_slope()
        
        # Clear previous HUD elements
        self.hud_elements = []
        
        # Panel dimensions - now exactly matches camera height
        panel_width = self.width - self.camera_width
        panel_x = self.camera_width
        panel_height = self.camera_height  # Match camera height
        
        # Draw main panel with gradient
        self._draw_gradient_panel(panel_x, 0, panel_width, panel_height)
        
        # Draw sections - now only 5 sections without header
        section_height = panel_height // 5
        
        # Section 1: Speed, Acceleration, Slope (start from 0)
        self._render_dynamics_section(panel_x, 0, panel_width, section_height, 
                                     speed, acceleration, slope)
        
        # Section 2: Battery status
        self._render_battery_section(panel_x, section_height, panel_width, section_height, 
                                    battery_state)
        
        # Section 3: Charging status
        self._render_charging_section(panel_x, section_height * 2, panel_width, section_height, 
                                     battery_state)
        
        # Section 4: Control status
        self._render_control_section(panel_x, section_height * 3, panel_width, section_height)
        
        # Section 5: Navigation info
        self._render_navigation_section(panel_x, section_height * 4, panel_width, section_height, 
                                       battery_state)
        
    def _draw_gradient_panel(self, x, y, width, height):
        """Draw a panel with gradient background"""
        # Create gradient surface
        gradient = pygame.Surface((width, height))
        
        # Create vertical gradient
        for i in range(height):
            # Calculate color based on position
            ratio = i / height
            r = int(self.colors['panel'][0] * (1 - ratio) + self.colors['panel_dark'][0] * ratio)
            g = int(self.colors['panel'][1] * (1 - ratio) + self.colors['panel_dark'][1] * ratio)
            b = int(self.colors['panel'][2] * (1 - ratio) + self.colors['panel_dark'][2] * ratio)
            
            pygame.draw.line(gradient, (r, g, b), (0, i), (width, i))
        
        # Draw gradient
        self.screen.blit(gradient, (x, y))
        
        # Draw border with accent
        pygame.draw.rect(self.screen, self.colors['border'], (x, y, width, height), 2)
        pygame.draw.rect(self.screen, self.colors['panel_accent'], (x, y, width, 4))
        
    def _render_dynamics_section(self, x, y, width, height, speed, acceleration, slope):
        """Render speed, acceleration, and slope"""
        # Section header
        header_rect = pygame.Rect(x, y, width, 30)
        pygame.draw.rect(self.screen, (40, 45, 60), header_rect)
        header = self.header_font.render("VEHICLE DYNAMICS", True, self.colors['text'])
        self.screen.blit(header, (x + 20, y + 5))
        
        # Content area - adjust for header
        content_y = y + 35
        content_height = height - 35
        
        # Divide into 3 columns
        col_width = width // 3
        
        # Speed gauge
        self._draw_gauge(x + 10, content_y, col_width - 20, content_height - 10,
                        speed, 0, 120, "km/h", "SPEED", self.colors['speed'])
        
        # Acceleration gauge
        self._draw_gauge(x + col_width + 10, content_y, col_width - 20, content_height - 10,
                        acceleration, -5, 5, "m/s²", "ACCELERATION", self.colors['acceleration'])
        
        # Slope gauge
        self._draw_gauge(x + 2 * col_width + 10, content_y, col_width - 20, content_height - 10,
                        slope, -15, 15, "°", "SLOPE", self.colors['slope'])
        
    def _draw_gauge(self, x, y, width, height, value, min_val, max_val, unit, label, color):
        """Draw a circular gauge"""
        # Background
        gauge_bg = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self.colors['gauge_bg'], gauge_bg, border_radius=8)
        
        # Calculate center and radius
        center_x = x + width // 2
        center_y = y + height // 2
        radius = min(width, height) // 2 - 10
        
        # Draw gauge arc
        pygame.draw.circle(self.screen, (50, 60, 70), (center_x, center_y), radius, 3)
        
        # Draw value arc
        value_ratio = (value - min_val) / (max_val - min_val)
        value_ratio = max(0, min(1, value_ratio))  # Clamp to 0-1
        angle = 180 + value_ratio * 180  # 180-360 degrees
        
        # Draw progress arc
        points = []
        for a in range(180, int(angle) + 1):
            rad = math.radians(a)
            px = center_x + radius * math.cos(rad)
            py = center_y + radius * math.sin(rad)
            points.append((px, py))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, 4)
        
        # Draw center value
        value_text = f"{value:.1f}"
        value_surface = self.value_font.render(value_text, True, color)
        value_rect = value_surface.get_rect(center=(center_x, center_y - 10))
        self.screen.blit(value_surface, value_rect)
        
        # Draw unit
        unit_surface = self.small_font.render(unit, True, self.colors['text_dim'])
        unit_rect = unit_surface.get_rect(center=(center_x, center_y + 15))
        self.screen.blit(unit_surface, unit_rect)
        
        # Draw label
        label_surface = self.small_font.render(label, True, self.colors['text'])
        label_rect = label_surface.get_rect(center=(center_x, y + height - 20))
        self.screen.blit(label_surface, label_rect)

    def _draw_button(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        text: str,
        callback,
        enabled: bool = True,
    ) -> None:
        """Draw a clickable UI button and register it in self.hud_elements."""
        rect = pygame.Rect(x, y, width, height)

        mouse_pos = pygame.mouse.get_pos()
        is_hover = rect.collidepoint(mouse_pos)

        if not enabled:
            base = (70, 75, 85)
            fg = self.colors['text_dim']
        else:
            base = self.colors['button_hover'] if is_hover else self.colors['button']
            fg = self.colors['text']

        pygame.draw.rect(self.screen, base, rect, border_radius=8)
        pygame.draw.rect(self.screen, self.colors['border_light'], rect, 1, border_radius=8)

        label = self.small_font.render(text, True, fg)
        label_rect = label.get_rect(center=rect.center)
        self.screen.blit(label, label_rect)

        self.hud_elements.append(
            {
                'type': 'button',
                'rect': rect,
                'callback': callback if enabled else None,
            }
        )
        
    def _render_battery_section(self, x, y, width, height, battery_state):
        """Render battery status section"""
        # Section header
        header_rect = pygame.Rect(x, y, width, 30)
        pygame.draw.rect(self.screen, (40, 45, 60), header_rect)
        header = self.header_font.render("BATTERY STATUS", True, self.colors['text'])
        self.screen.blit(header, (x + 20, y + 5))
        
        # Content area
        content_y = y + 35
        content_height = height - 35
        
        # SOC gauge
        soc = battery_state['soc']
        soc_color = self.colors['battery_low'] if soc < 20 else self.colors['battery_med'] if soc < 50 else self.colors['battery_high']
        
        # Large SOC display
        soc_text = f"{soc:.2f}%"
        soc_surface = self.title_font.render(soc_text, True, soc_color)
        soc_rect = soc_surface.get_rect(center=(x + width // 2, content_y + 30))
        self.screen.blit(soc_surface, soc_rect)
        
        # SOC bar
        bar_width = width - 40
        bar_height = 20
        bar_x = x + 20
        bar_y = content_y + 70
        
        # Background
        pygame.draw.rect(self.screen, self.colors['gauge_bg'], 
                        (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        
        # Fill
        fill_width = int(bar_width * (soc / 100))
        pygame.draw.rect(self.screen, soc_color, 
                        (bar_x, bar_y, fill_width, bar_height), border_radius=4)
        
        # Border
        pygame.draw.rect(self.screen, self.colors['border'], 
                        (bar_x, bar_y, bar_width, bar_height), 2, border_radius=4)
        
        # Battery info
        info_y = bar_y + 35
        info_items = [
            #("Required:", f"{battery_state['soc_required']:.2f}%"),
            ("Energy Used:", f"{battery_state['total_energy']:.3f} kWh"),
        ]
        
        for label, value in info_items:
            label_surface = self.data_font.render(label, True, self.colors['text_dim'])
            value_surface = self.value_font.render(value, True, self.colors['text'])
            self.screen.blit(label_surface, (x + 20, info_y))
            self.screen.blit(value_surface, (x + width - 120, info_y))
            info_y += 25
            
    def _render_charging_section(self, x, y, width, height, battery_state):
        """Render charging status section"""
        # Section header
        header_rect = pygame.Rect(x, y, width, 30)
        in_dwcl = battery_state['in_dwcl']
        header_color = self.colors['charging'] if in_dwcl else self.colors['text']
        header_text = "CHARGING ACTIVE" if in_dwcl else "NOT CHARGING"
        pygame.draw.rect(self.screen, (40, 45, 60), header_rect)
        header = self.header_font.render(header_text, True, header_color)
        self.screen.blit(header, (x + 20, y + 5))
        
        # Content area
        content_y = y + 35
        
        # Charging power with icon
        power = battery_state['charging_power']
        power_color = self.colors['power'] if power > 0 else self.colors['text_dim']
        power_text = f"{power:.2f} kW"
        power_surface = self.value_font.render(power_text, True, power_color)
        power_rect = power_surface.get_rect(center=(x + width // 2, content_y + 15))
        self.screen.blit(power_surface, power_rect)
        
        # Charging parameters
        param_y = content_y + 40
        param_items = [
            ("Alignment:", f"{battery_state['alignment_factor']:.3f}"),
            ("Coupling:", f"{battery_state['coupling_coefficient']:.3f}"),
            ("Efficiency:", f"{battery_state['transfer_efficiency']:.3f}"),
        ]
        
        for label, value in param_items:
            label_surface = self.data_font.render(label, True, self.colors['text_dim'])
            value_surface = self.value_font.render(value, True, self.colors['text'])
            self.screen.blit(label_surface, (x + 20, param_y))
            self.screen.blit(value_surface, (x + width - 120, param_y))
            param_y += 25
            
    def _render_control_section(self, x, y, width, height):
        """Render control status section"""
        # Section header
        header_rect = pygame.Rect(x, y, width, 30)
        pygame.draw.rect(self.screen, (40, 45, 60), header_rect)
        header = self.header_font.render("VEHICLE CONTROLS", True, self.colors['text'])
        self.screen.blit(header, (x + 20, y + 5))
        
        # Content area
        content_y = y + 35
        
        # Control bars
        controls = [
            ("Throttle", self.control.throttle, self.colors['success']),
            ("Brake", self.control.brake, self.colors['warning']),
            ("Steer", abs(self.control.steer), self.colors['speed']),
        ]
        
        for i, (label, value, color) in enumerate(controls):
            bar_y = content_y + i * 35
            bar_width = width - 100
            bar_height = 20
            
            # Label
            label_surface = self.data_font.render(label, True, self.colors['text_dim'])
            self.screen.blit(label_surface, (x + 20, bar_y))
            
            # Value
            value_surface = self.value_font.render(f"{value:.2f}", True, color)
            self.screen.blit(value_surface, (x + width - 50, bar_y))
            
            # Bar background
            pygame.draw.rect(self.screen, self.colors['gauge_bg'], 
                           (x + 20, bar_y + 20, bar_width, bar_height), border_radius=3)
            
            # Bar fill
            fill_width = int(bar_width * value)
            pygame.draw.rect(self.screen, color, 
                           (x + 20, bar_y + 20, fill_width, bar_height), border_radius=3)
            
            # Bar border
            pygame.draw.rect(self.screen, self.colors['border'], 
                           (x + 20, bar_y + 20, bar_width, bar_height), 1, border_radius=3)
            
        # Additional controls
        add_y = content_y + 105
        add_controls = [
            ("Hand Brake:", "ON" if self.control.hand_brake else "OFF"),
            ("Reverse:", "ON" if self.control.reverse else "OFF"),
        ]
        
        for label, value in add_controls:
            label_surface = self.data_font.render(label, True, self.colors['text_dim'])
            value_surface = self.value_font.render(value, True, self.colors['text'])
            self.screen.blit(label_surface, (x + 20, add_y))
            self.screen.blit(value_surface, (x + width - 50, add_y))
            add_y += 25
        """
        # Lane-assist buttons
        btn_y = y + height - 50
        btn_h = 34
        btn_w = (width - 60) // 2
        btn_x1 = x + 20
        btn_x2 = x + 40 + btn_w

        # Disable while lane assist is active (prevents stacked requests)
        assist_enabled = not self.lane_assist_active

        self._draw_button(
            btn_x1,
            btn_y,
            btn_w,
            btn_h,
            "Enter DWCL (1)",
            self.request_enter_dwcl,
            enabled=assist_enabled,
        )
        self._draw_button(
            btn_x2,
            btn_y,
            btn_w,
            btn_h,
            "Exit DWCL (2)",
            self.request_exit_dwcl,
            enabled=assist_enabled,
        )
        """           
    def _render_navigation_section(self, x, y, width, height, battery_state):
        """Render navigation info section"""
        # Section header
        header_rect = pygame.Rect(x, y, width, 30)
        pygame.draw.rect(self.screen, (40, 45, 60), header_rect)
        header = self.header_font.render("NAVIGATION", True, self.colors['text'])
        self.screen.blit(header, (x + 20, y + 5))
        
        # Content area
        content_y = y + 35
        
        # Large distance display
        distance = battery_state['remaining_distance']
        distance_text = f"{distance:.0f}"
        distance_surface = self.title_font.render(distance_text, True, self.colors['dwcl_active'])
        distance_rect = distance_surface.get_rect(center=(x + width // 2, content_y + 15))
        self.screen.blit(distance_surface, distance_rect)
        
        # Unit
        unit_surface = self.small_font.render("meters to destination", True, self.colors['text_dim'])
        unit_rect = unit_surface.get_rect(center=(x + width // 2, content_y + 45))
        self.screen.blit(unit_surface, unit_rect)
        
        # ETA and other info
        info_y = content_y + 70
        info_items = [
            ("ETA:", f"{battery_state['eta']:.1f}s"),
            ("Required SoC to Destination:", f"{-battery_state['soc_required']:.2f}%"),
            ("On DWCL:", "YES" if battery_state['in_dwcl'] else "NO"),
        ]
        
        for label, value in info_items:
            label_surface = self.data_font.render(label, True, self.colors['text_dim'])
            value_surface = self.value_font.render(value, True, self.colors['text'])
            self.screen.blit(label_surface, (x + 20, info_y))
            self.screen.blit(value_surface, (x + width - 120, info_y))
            info_y += 25
            
    def _render_help_overlay(self):
        """Render help overlay with controls"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 220))
        self.screen.blit(overlay, (0, 0))
        
        # Modern help box
        box_width = 600
        box_height = 500
        box_x = (self.width - box_width) // 2
        box_y = (self.height - box_height) // 2
        
        # Draw help box with gradient
        help_box = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
        pygame.draw.rect(help_box, (30, 40, 60, 240), (0, 0, box_width, box_height), border_radius=10)
        pygame.draw.rect(help_box, self.colors['dwcl_active'], (0, 0, box_width, box_height), 2, border_radius=10)
        self.screen.blit(help_box, (box_x, box_y))
        
        # Title
        title = self.title_font.render("CONTROLS HELP", True, self.colors['dwcl_active'])
        title_rect = title.get_rect(center=(self.width // 2, box_y + 30))
        self.screen.blit(title, title_rect)
        
        # Help items in two columns
        help_items_left = [
            ("Arrow Keys", "Drive vehicle"),
            ("↑", "Accelerate"),
            ("↓", "Brake"),
            ("← →", "Steer"),
            ("SPACE", "Hand brake"),
            ("B", "Reverse gear"),
        ]
        
        help_items_right = [
            ("M", "Manual/Autopilot"),
            ("1", "Lane assist: Enter DWCL"),
            ("2", "Lane assist: Exit DWCL"),
            ("P", "Pause/Resume"),
            ("R", "Recording"),
            ("T", "Reset vehicle"),
            ("H", "This help"),
            ("ESC", "Exit"),
        ]
        
        # Left column
        text_y = box_y + 80
        for key, description in help_items_left:
            key_surface = self.data_font.render(key, True, self.colors['charging'])
            desc_surface = self.data_font.render(description, True, self.colors['text'])
            self.screen.blit(key_surface, (box_x + 40, text_y))
            self.screen.blit(desc_surface, (box_x + 180, text_y))
            text_y += 35
            
        # Right column
        text_y = box_y + 80
        for key, description in help_items_right:
            key_surface = self.data_font.render(key, True, self.colors['charging'])
            desc_surface = self.data_font.render(description, True, self.colors['text'])
            self.screen.blit(key_surface, (box_x + box_width - 200, text_y))
            self.screen.blit(desc_surface, (box_x + box_width - 140, text_y))
            text_y += 35
            
        # Close hint
        close_hint = self.data_font.render("Press H to close help", True, self.colors['text_dim'])
        close_rect = close_hint.get_rect(center=(self.width // 2, box_y + box_height - 30))
        self.screen.blit(close_hint, close_rect)
        
    def _render_debug_info(self):
        """Render debug information overlay"""
        debug_y = 10
        debug_x = 10
        
        # Get debug data
        location = self.vehicle.get_location()
        rotation = self.vehicle.get_transform().rotation
        velocity = self.vehicle.get_velocity()
        speed = self._get_speed_kmh()
        
        debug_items = [
            f"Location: X={location.x:.2f}, Y={location.y:.2f}, Z={location.z:.2f}",
            f"Rotation: Pitch={rotation.pitch:.1f}, Yaw={rotation.yaw:.1f}",
            f"Velocity: X={velocity.x:.2f}, Y={velocity.y:.2f}",
            f"Speed: {speed:.2f} km/h",
            f"Simulation Time: {self.world.get_snapshot().timestamp.elapsed_seconds:.2f}s",
            f"FPS: {self.clock.get_fps():.1f}",
        ]
        
        # Draw debug box
        debug_height = len(debug_items) * 25 + 20
        debug_rect = pygame.Rect(debug_x, debug_y, 400, debug_height)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), debug_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.colors['border_light'], debug_rect, 1, border_radius=5)
        
        # Draw debug text
        for i, text in enumerate(debug_items):
            text_surface = self.small_font.render(text, True, (200, 220, 255))
            self.screen.blit(text_surface, (debug_x + 10, debug_y + 10 + i * 25))
            
    def _take_screenshot(self):
        """Take and save a screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_dir = "data/screenshots"
        os.makedirs(screenshot_dir, exist_ok=True)
        screenshot_path = os.path.join(screenshot_dir, f"screenshot_{timestamp}.png")
        
        pygame.image.save(self.screen, screenshot_path)
        print(f"Screenshot saved to: {screenshot_path}")
        
    def run(self):
        """Main simulation loop"""
        if not self.initialize_simulator("models/energy_model_best.hdf5"):
            print("Failed to initialize simulation. Exiting...")
            return
            
        print("\n" + "="*60)
        print("DWCL INTERACTIVE SIMULATOR READY")
        print("="*60)
        print("Controls:")
        print("  Arrow Keys: Drive vehicle")
        print("  M: Toggle Manual/Autopilot mode")
        print("  1: Lane assist - Enter DWCL")
        print("  2: Lane assist - Exit DWCL")
        print("  H: Show/Hide help")
        print("  P: Pause/Resume simulation")
        print("  ESC: Exit")
        print("="*60 + "\n")
        
        running = True
        while running:
            # Handle events
            running = self.handle_events()
            
            # Update controls
            self.update_controls()
            
            # Update simulation
            if not self.paused:
                self.update_simulation()
                
            # Render everything
            self.render()
            
            # Cap FPS
            self.clock.tick(self.fps)
            
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        
        # Stop recording
        if self.recording:
            print("Stopped recording")
            
        # Destroy camera
        if self.camera:
            self.camera.destroy()
            
        # Let the shared CARLA environment clean up its actors/settings
        if self.env is not None:
            try:
                self.env.cleanup()
            except Exception:
                pass
        elif self.vehicle:
            # Fallback
            try:
                self.vehicle.destroy()
            except Exception:
                pass
            
        # Close CARLA connection
        if self.client:
            print("Closing CARLA connection...")
            
        # Quit PyGame
        pygame.quit()
        
        print("Cleanup complete!")
        print("\nSession Summary:")
        print(f"  Log file: {self.log_file}")
        print(f"  Session duration: {(pygame.time.get_ticks() - self.start_time) / 1000:.1f}s")
        
        # List any screenshots taken
        screenshot_dir = "data/screenshots"
        if os.path.exists(screenshot_dir):
            screenshots = [f for f in os.listdir(screenshot_dir) if f.startswith("screenshot_")]
            if screenshots:
                print(f"  Screenshots taken: {len(screenshots)}")


def main():
    """Main entry point"""
    # Check if CARLA is available
    try:
        import carla
    except ImportError:
        print("CARLA Python API not found!")
        print("Please ensure CARLA is installed and added to your PYTHONPATH")
        print("Example: export PYTHONPATH=$PYTHONPATH:/path/to/carla/PythonAPI/carla/dist/carla-0.9.14-py3.8-linux-x86_64.egg")
        return
        
    # Check if power model exists
    power_model_path = os.path.join("models", "energy_model_best.hdf5")
    if not os.path.exists(power_model_path):
        print(f"Power model not found at: {power_model_path}")
        print("Please ensure the power model is available in the models/ directory")
        print("You can download it or train your own using the training script.")
        return
        
    # Create and run visualizer
    visualizer = DWCLPygameVisualizer()
    
    try:
        visualizer.run()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        visualizer.cleanup()
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        visualizer.cleanup()


if __name__ == "__main__":
    main()