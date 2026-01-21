"""
Simple training entry point for the DWCL-RL CARLA project.

Key points:
- Loads the DWCL power model from Keras HDF5 (.h5/.hdf5).
- Uses RLTrainer + DQNAgent + CarlaEnv from this codebase.
"""
from __future__ import annotations

import os
import random

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


"""
Simple training entry point for the DWCL-RL CARLA project.

Updates:
- No argparse / CLI parsing.
- Loads DWCL power model from Keras HDF5 (.h5/.hdf5).
- Auto-resume from the latest checkpoint (restores epsilon + episode number).
"""



import os
import random
import re
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import torch  # used for seeding / device detection; RL agent is torch

from tensorflow.keras.models import load_model  # Keras HDF5 loader

from src.carla_simulator.carla_env import CarlaEnv
from src.rl_agent.dqn_agent import DQNAgent
from src.rl_agent.trainer import RLTrainer
from config import constants as const


# ----------------------------
# Minimal config (no argparse)
# ----------------------------
@dataclass
class TrainConfig:
    # Training loop
    num_episodes: int = 500
    max_steps_per_episode: int = 600
    checkpoint_interval: int = 25

    # DQN hyperparameters (agent reads what it needs from this config)
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64])
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    target_update_freq: int = 200
    replay_buffer_size: int = 50_000

    # Exploration (per-episode decay)
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05

    # Simulation
    no_rendering_mode: bool = False


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_power_model_keras_hdf5(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Power model not found: {path}")
    # compile=False is recommended for inference-only use
    return load_model(path, compile=False)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return path to latest checkpoint_ep{N}.pth (or final/best) in checkpoint_dir."""
    if not os.path.isdir(checkpoint_dir):
        return None

    # Prefer regular checkpoints; allow final/best as fallbacks
    pattern = re.compile(r"(checkpoint_ep|final_model_ep|best_model_ep)(\d+).*(\.pth)$")

    best = None  # (episode, path)
    for name in os.listdir(checkpoint_dir):
        m = pattern.match(name)
        if not m:
            continue
        ep = int(m.group(2))
        path = os.path.join(checkpoint_dir, name)
        if best is None or ep > best[0]:
            best = (ep, path)

    return best[1] if best else None


def main(
    power_model_path: Optional[str] = None,
    resume_checkpoint_path: Optional[str] = None,
    seed: int = 42,
) -> None:
    set_seed(seed)

    if power_model_path is None:
        power_model_path = getattr(const, "POWER_MODEL_PATH", None)
    if not power_model_path:
        raise ValueError("Set POWER_MODEL_PATH at bottom of this file or define const.POWER_MODEL_PATH.")

    # Config (keep simple; constants may override)
    cfg = TrainConfig(
        num_episodes=getattr(const, "NUM_EPISODES", 500),
        max_steps_per_episode=getattr(const, "MAX_STEPS_PER_EPISODE", 600),
        checkpoint_interval=getattr(const, "CHECKPOINT_INTERVAL", 25),
        no_rendering_mode=getattr(const, "NO_RENDERING_MODE", False),
        learning_rate=getattr(const, "LEARNING_RATE", 1e-3),
        gamma=getattr(const, "GAMMA", 0.99),
        batch_size=getattr(const, "BATCH_SIZE", 64),
        target_update_freq=getattr(const, "TARGET_UPDATE_FREQ", 200),
        replay_buffer_size=getattr(const, "REPLAY_BUFFER_SIZE", 50_000),
        epsilon_start=getattr(const, "EPSILON_START", 1.0),
        epsilon_decay=getattr(const, "EPSILON_DECAY", 0.995),
        epsilon_min=getattr(const, "EPSILON_MIN", 0.05),
    )

    if hasattr(const, "HIDDEN_LAYERS") and const.HIDDEN_LAYERS:
        cfg.hidden_layers = list(const.HIDDEN_LAYERS)

    # Environment
    env = CarlaEnv(cfg)

    # Power model (Keras HDF5)
    power_model = load_power_model_keras_hdf5(power_model_path)

    # Initialize CARLA
    env.initialize_simulation(power_model)

    # Agent
    state_dim = getattr(const, "STATE_DIM", 6)
    action_dim = getattr(const, "ACTION_DIM", 6)
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, config=cfg)

    # Trainer
    trainer = RLTrainer(env=env, agent=agent, config=cfg)

    # Auto-resume if not provided
    if resume_checkpoint_path is None:
        ckpt_dir = os.path.join(getattr(const, "MODEL_SAVE_DIR", "models"), "checkpoints")
        resume_checkpoint_path = find_latest_checkpoint(ckpt_dir)

    if resume_checkpoint_path:
        print(f"🔁 Resuming from checkpoint: {resume_checkpoint_path}")
        trainer.load_checkpoint(resume_checkpoint_path)
    else:
        print("🆕 No checkpoint found. Starting fresh training.")

    # Train
    try:
        trainer.train(num_episodes=cfg.num_episodes)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C). Saving interrupt checkpoint...")
        try:
            # Save a normal checkpoint at the current episode index
            trainer._save_checkpoint(final=True)  # best-effort; uses agent.save + state pickle
        except Exception as e:
            print(f"Could not save interrupt checkpoint: {e}")
    finally:
        try:
            if hasattr(env, "cleanup"):
                env.cleanup()
            else:
                env.close()
        except Exception as e:
            print(f"Cleanup warning: {e}")


if __name__ == "__main__":
    # Keep it simple: edit these two lines if needed.
    POWER_MODEL_PATH = None  # e.g., r"models/power_model.h5"
    RESUME_CHECKPOINT = None  # Leave None to auto-resume from latest checkpoint

    main(power_model_path=POWER_MODEL_PATH, resume_checkpoint_path=RESUME_CHECKPOINT)
