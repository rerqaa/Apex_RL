import torch
import numpy as np
from architecture import AttentionApexPolicy

class ApexRLAgent:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model with same params as training
        # Note: Action space 8 is standard for RLGym ContinuousAction
        # Layer sizes must match what was given to PPOLearner
        layer_sizes = (256, 256, 256)
        
        # We don't need the exact obs size because the encoder maps to a combined size
        self.model = AttentionApexPolicy(
            input_shape=None, 
            output_shape=16, 
            layer_sizes=layer_sizes, 
            device=self.device
        ).to(self.device)

        print(f"Loading model from: {model_path}")
        if str(model_path).endswith('.pt'):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        self.model.eval()

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            # The get_action method in MaskedContinuousPolicy handles [1, obs_size] shaping
            action, _ = self.model.get_action(obs, deterministic=True)
            return action.numpy()[0]