"""
Training orchestration for DQN agent.

Key behavior:
- Epsilon decay is applied **per episode** (not per training step).
- Resume is exact: restores agent state (including epsilon) and trainer state,
  and continues from the next episode after the last saved one.
- Uses TrainingMetrics API from metrics.py (metrics.update / to_dict / from_dict).
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import pandas as pd

from src.rl_agent.metrics import TrainingMetrics
from config import constants as const


class RLTrainer:
    """Orchestrates RL training with checkpointing and metrics tracking."""

    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config

        # Metrics tracking (uses TrainingMetrics.update)
        self.metrics = TrainingMetrics()

        # Training state
        self.current_episode: int = 0
        self.best_reward: float = -float("inf")

        # Directories
        self.checkpoint_dir = os.path.join(getattr(const, "MODEL_SAVE_DIR", "models"), "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.results_dir = getattr(const, "RESULTS_DIR", "results")
        os.makedirs(self.results_dir, exist_ok=True)

    # ------------------------
    # Core training loop
    # ------------------------
    def train_episode(self) -> Dict:
        """Train for one episode and return episode metrics."""

        state = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0
        done = False

        losses = []

        episode_metrics = {
            "reward": 0.0,
            "steps": 0,
            "time": 0.0,
            "final_soc": 0.0,
            "arrived": False,
            "epsilon": float(self.agent.get_epsilon()),
            "losses": losses,
            "lane_switches": 0,
            "invalid_actions": 0,
        }

        start_time = (
            self.env.world.get_snapshot().timestamp.elapsed_seconds
            if getattr(self.env, "world", None)
            else 0.0
        )

        while not done and episode_steps < int(self.config.max_steps_per_episode):
            # Select action (epsilon-greedy inside agent)
            action = self.agent.select_action(state)

            # Transition (enter/exit DWCL manoeuvres are blocking inside env.step)
            next_state, reward, done, info = self.env.step(action)

            # Store + train
            self.agent.store_experience(state, action, reward, next_state, done)
            loss = self.agent.train()
            if loss is not None:
                try:
                    losses.append(float(loss))
                except Exception:
                    pass

            # Bookkeeping
            episode_reward += float(reward)
            episode_steps += 1
            state = next_state

            # Pull env performance counters if available
            episode_metrics["lane_switches"] = int(getattr(self.env, "lane_switch_counter", 0))
            episode_metrics["invalid_actions"] = int(getattr(self.env, "invalid_action_counter", 0))

            # SOC snapshot (best effort)
            try:
                soc = float(self.env.battery_manager.get_soc())
                episode_metrics["final_soc"] = soc
            except Exception:
                pass

        end_time = (
            self.env.world.get_snapshot().timestamp.elapsed_seconds
            if getattr(self.env, "world", None)
            else start_time
        )

        episode_metrics["time"] = float(end_time - start_time)
        episode_metrics["reward"] = float(episode_reward)
        episode_metrics["steps"] = int(episode_steps)
        episode_metrics["arrived"] = bool(getattr(self.env, "has_arrived", False))

        # ------------------------
        # Per-episode epsilon decay
        # ------------------------
        # NOTE: For true per-episode decay, remove epsilon decay from DQNAgent.train().
        try:
            self.agent.epsilon = max(
                float(self.agent.epsilon) * float(self.config.epsilon_decay),
                float(self.config.epsilon_min),
            )
        except Exception:
            pass

        # Log epsilon after decay (this is the value for the next episode)
        episode_metrics["epsilon"] = float(self.agent.get_epsilon())

        # Track metrics history (TrainingMetrics.update)
        self.metrics.update(episode_metrics)

        # Checkpointing
        if (int(self.current_episode) + 1) % int(self.config.checkpoint_interval) == 0:
            self._save_checkpoint()

        # Best model
        if episode_reward > self.best_reward:
            self.best_reward = float(episode_reward)
            self._save_checkpoint(best=True)

        return episode_metrics

    def train(self, num_episodes: Optional[int] = None) -> None:
        if num_episodes is None:
            num_episodes = int(self.config.num_episodes)

        print(f"🚀 Starting training for {num_episodes} episodes")
        print(f"📊 Checkpoint interval: {self.config.checkpoint_interval}")

        for episode in range(int(self.current_episode), int(num_episodes)):
            self.current_episode = int(episode)
            metrics = self.train_episode()
            self._log_episode(self.current_episode, metrics)

        self._save_checkpoint(final=True)

        print(f"✅ Training completed for {num_episodes} episodes")
        print(f"🏆 Best reward: {self.best_reward:.2f}")

    # ------------------------
    # Checkpointing
    # ------------------------
    def _save_checkpoint(self, best: bool = False, final: bool = False) -> None:
        if best:
            filename = f"best_model_ep{self.current_episode}_reward{self.best_reward:.2f}.pth"
        elif final:
            filename = f"final_model_ep{self.current_episode}.pth"
        else:
            filename = f"checkpoint_ep{self.current_episode}.pth"

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        # Save agent checkpoint (includes epsilon)
        self.agent.save(checkpoint_path)

        # Save trainer state next to it
        training_state = {
            "current_episode": int(self.current_episode),
            "best_reward": float(self.best_reward),
            "metrics": self.metrics.to_dict(),
        }
        state_path = checkpoint_path.replace(".pth", "_state.pkl")
        pd.to_pickle(training_state, state_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.agent.load(checkpoint_path)

        state_path = checkpoint_path.replace(".pth", "_state.pkl")
        if os.path.exists(state_path):
            training_state = pd.read_pickle(state_path)
            # Continue from NEXT episode (avoid repeating saved episode index)
            self.current_episode = int(training_state.get("current_episode", 0)) + 1
            self.best_reward = float(training_state.get("best_reward", -float("inf")))
            try:
                self.metrics.from_dict(training_state.get("metrics", {}))
            except Exception:
                pass
        else:
            self.current_episode = 0

        print(
            f"📂 Checkpoint loaded. Resuming at episode {self.current_episode} | "
            f"epsilon={self.agent.get_epsilon():.4f}"
        )

    # ------------------------
    # Logging
    # ------------------------
    def _log_episode(self, episode: int, metrics: Dict) -> None:
        print(
            f"Episode {episode + 1:4d} | "
            f"Reward: {metrics.get('reward', 0):7.2f} | "
            f"Steps: {metrics.get('steps', 0):3d} | "
            f"Time: {metrics.get('time', 0):6.2f}s | "
            f"Final SoC: {metrics.get('final_soc', 0):5.2f}% | "
            f"Epsilon: {self.agent.get_epsilon():.4f} | "
            f"Arrived: {'✓' if metrics.get('arrived', False) else '✗'}"
        )

    def get_metrics(self) -> TrainingMetrics:
        return self.metrics
