"""
Apex-RL Bot — RLBot BaseAgent that loads a trained PPO model
and plays in Rocket League using the same observation format as training.
"""

import os
import sys
import numpy as np
import torch

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

# Add src/ to path so we can import architecture
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC_DIR)

import architecture
import game_state_adapter
import importlib

importlib.reload(architecture)
importlib.reload(game_state_adapter)

from architecture import AttentionApexPolicy
from game_state_adapter import GameStateAdapter


class ApexBot(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.adapter = GameStateAdapter()
        self.model = None
        self.device = None

    def initialize_agent(self):
        """Called once when the bot starts. Load model and build boost map."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build boost pad index mapping
        field_info = self.get_field_info()
        self.adapter.build_boost_map(field_info)

        # Load model
        self.model = AttentionApexPolicy(
            input_shape=None,
            output_shape=16,
            layer_sizes=(256, 256, 256),
            device=self.device
        ).to(self.device)

        # Find the latest checkpoint
        checkpoint_dir = os.path.join(SRC_DIR, 'checkpoints')
        model_path = self._find_latest_checkpoint(checkpoint_dir)

        if model_path is not None:
            self.logger.info(f"Loading model from: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.logger.info("Model loaded successfully.")
        else:
            self.logger.warning("No checkpoint found! Bot will use random weights.")

        self.model.eval()

    def _find_latest_checkpoint(self, checkpoint_dir):
        """Find the most recent PPO_POLICY.pt in the nested checkpoints directory."""
        if not os.path.isdir(checkpoint_dir):
            return None

        latest_ts = -1
        latest_path = None

        for run_id in os.listdir(checkpoint_dir):
            run_dir = os.path.join(checkpoint_dir, run_id)
            if not os.path.isdir(run_dir): continue
            
            for step in os.listdir(run_dir):
                step_dir = os.path.join(run_dir, step)
                if not os.path.isdir(step_dir): continue
                
                policy_file = os.path.join(step_dir, 'PPO_POLICY.pt')
                if os.path.isfile(policy_file):
                    try:
                        ts = int(step)
                        if ts > latest_ts:
                            latest_ts = ts
                            latest_path = policy_file
                    except ValueError:
                        pass

        return latest_path

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """Called every game tick. Build obs, run model, return controls."""
        if self.model is None:
            return SimpleControllerState()

        # Build observation
        obs = self.adapter.build_obs(packet, self.index, self.team)

        # Run model inference
        with torch.no_grad():
            action, _ = self.model.get_action(obs, deterministic=True)
            action = action.cpu().numpy().flatten()

        # Store action for next observation
        self.adapter.set_previous_action(action)

        # Map 8-dim continuous action → SimpleControllerState
        # Action order matches ContinuousAction parser:
        # [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        controls = SimpleControllerState()
        controls.throttle = float(np.clip(action[0], -1, 1))
        controls.steer = float(np.clip(action[1], -1, 1))
        controls.pitch = float(np.clip(action[2], -1, 1))
        controls.yaw = float(np.clip(action[3], -1, 1))
        controls.roll = float(np.clip(action[4], -1, 1))
        controls.jump = action[5] > 0       # Threshold at 0
        controls.boost = action[6] > 0
        controls.handbrake = action[7] > 0

        return controls
