"""
DWCL-RL Simulation Constants
"""

# CARLA Simulation
SIM_HOST = 'localhost'
SIM_PORT = 2000
TOWN = 'Town06'
SYNC_MODE = True
FIXED_DELTA_SECONDS = 0.1  # seconds

# Battery and Charging
SOC_INITIAL = 80  # Initial State of Charge (%)
ETA_CH = 0.92  # Battery charging efficiency
P_EV_MAX = 80  # Maximum Power Transfer (kW)
Q1 = 90  # Quality factor of primary coil
Q2 = 90  # Quality factor of secondary coil
K0 = 0.1  # Nominal coupling coefficient
D_MAX = 0.5  # Maximum lateral misalignment (meters)

# Vehicle Parameters
VEHICLE_MASS = 1680  # kg
GRAVITY = 9.81  # m/s²
ROLLING_RESISTANCE = 0.01
DRAG_COEFFICIENT = 0.28
FRONTAL_AREA = 1.93  # m²
AIR_DENSITY = 1.20  # kg/m³
BATTERY_CAPACITY = 24  # kWh
TRANSMISSION_EFFICIENCY = 0.90
REGENERATION_EFFICIENCY = 0.75
AUXILIARY_POWER = 0  # kW
MAX_SPEED = 70 / 3.6  # m/s
AVG_SPEED = 35 / 3.6  # m/s
AVG_SLOPE = 0.002
AVG_ACCELERATION = 0.28

# DWCL Parameters
DWCL_LENGTH = 569  # meters
DWCL_WIDTH = 3.5  # meters
COIL_WIDTH = 6  # meters
COIL_HEIGHT = 1  # meters
COIL_SPACING = 2  # meters

# Reinforcement Learning
STATE_DIM = 6  # [SoC, Required_SoC, ETA, Distance_to_DWCL_End, Lane_Type, Target_Speed]
ACTION_DIM = 6  # [0:Go_DWCL, 1:Leave_DWCL, 2:Accelerate, 3:Decelerate, 4:Maintain_Speed, 5:Stay_Out]
OBSERVATION_LOW = [0, 0, 0, 0, 0, 5]
OBSERVATION_HIGH = [100, 100, 1000, 1000, 1, 100]

# Training Parameters
GAMMA = 0.9
LEARNING_RATE = 0.001
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 10000
EPSILON_START = 1.0
EPSILON_DECAY = 0.99995
EPSILON_MIN = 0.05
TARGET_UPDATE_FREQ = 1000
CHECKPOINT_INTERVAL = 50
EPISODES = 1000

# Paths
POWER_MODEL_PATH = "models/energy_model_best.hdf5"
MODEL_SAVE_DIR = "models/saved_models/"
RESULTS_DIR = "data/results/"