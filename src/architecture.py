import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


def init_weights(module, gain=np.sqrt(2)):
    """Apply Orthogonal Initialization to Linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class AttentionApexPolicy(nn.Module):
    def __init__(self, input_shape, output_shape, layer_sizes, device, var_min=0.1, var_max=1.0):
        super().__init__()
        self.device = device
        # Standard deviation is learned independently
        self.log_std = nn.Parameter(torch.full((output_shape // 2,), -0.5).to(self.device))
        
        # Must match obs.py: 24 features (20 physics + 4 one-hot) + 1 mask = 25
        self.ENTITY_FEAT_SIZE = 25
        
        # Self (1) + Allies (2) + Enemies (3) = 6 players
        self.NUM_PLAYERS = 6
        
        self.D_MODEL = 128
        
        # Shared Encoder for all entities (6 players + 1 ball)
        self.entity_encoder = nn.Sequential(
            nn.Linear(self.ENTITY_FEAT_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, self.D_MODEL),
            nn.ReLU()
        )
        
        # Transformer Encoder (Permutation-Invariant Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D_MODEL, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Global features: BoostPads(34) + Action(8) + RelBallPos(3) + RelBallVel(3) + ScoreDiff(1) + BallTouched(1) = 50
        self.global_feat_size = 50
        
        # Combined size: Self embedding (128) + 50 global = 178
        self.combined_size = self.D_MODEL + self.global_feat_size
        
        # The Head
        assert len(layer_sizes) > 0, "Layer sizes must be provided"
        layers = [nn.Linear(self.combined_size, layer_sizes[0]), nn.ReLU()]

        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(layer_sizes[-1], output_shape // 2))
        layers.append(nn.Tanh()) # ContinuousPolicy ends in Tanh
        
        self.head = nn.Sequential(*layers).to(self.device)
        self.entity_encoder = self.entity_encoder.to(self.device)
        self.transformer = self.transformer.to(self.device)

        # --- Orthogonal Initialization ---
        # Entity encoder: all Linear layers with gain=sqrt(2)
        for module in self.entity_encoder:
            init_weights(module, gain=np.sqrt(2))
        
        # Head: all Linear layers except the final output layer with gain=sqrt(2)
        head_linear_layers = [m for m in self.head if isinstance(m, nn.Linear)]
        for layer in head_linear_layers[:-1]:
            init_weights(layer, gain=np.sqrt(2))
        
        # Final layer of Policy: gain=0.01 (keeps Tanh input near 0 → centered actions)
        init_weights(head_linear_layers[-1], gain=0.01)
        
        # Transformer: leave PyTorch defaults (do NOT manually init)


    def _parse_obs(self, obs):
        """Parse flat observation into entity features and global features.
        
        Since obs.py now outputs all entities (including ball) in uniform
        25-feature format, we slice them identically.
        """
        # Total entities: 6 players + 1 ball = 7
        total_entities = self.NUM_PLAYERS + 1  # 7
        entity_end = total_entities * self.ENTITY_FEAT_SIZE  # 7 * 25 = 175
        
        entity_features = []
        start_idx = 0
        for _ in range(total_entities):
            end_idx = start_idx + self.ENTITY_FEAT_SIZE
            entity_features.append(obs[..., start_idx:end_idx])
            start_idx = end_idx
        
        # Global features are everything remaining after entities
        global_features = obs[..., entity_end:]
        
        return entity_features, global_features

    def get_output(self, obs):
        if type(obs) != torch.Tensor:
            if type(obs) != np.array:
                obs = np.asarray(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        entity_features, global_features = self._parse_obs(obs)

        # Stack entities: [Batch, 7, 25]
        entities = torch.stack(entity_features, dim=-2)
        is_present = entities[..., -1:]
        
        # Encode -> [Batch, 7, 128]
        encoded_entities = self.entity_encoder(entities) * is_present
        
        # Transformer -> [Batch, 7, 128]
        batch_size = encoded_entities.shape[0]
        padding_mask = (is_present.squeeze(-1) == 0.0)

        max_chunk = 32768
        if batch_size > max_chunk:
            chunks = []
            for i in range(0, batch_size, max_chunk):
                enc_chunk = encoded_entities[i:i+max_chunk]
                mask_chunk = padding_mask[i:i+max_chunk]
                chunks.append(self.transformer(enc_chunk, src_key_padding_mask=mask_chunk))
            attended_entities = torch.cat(chunks, dim=0)
        else:
            attended_entities = self.transformer(encoded_entities, src_key_padding_mask=padding_mask)
        
        # Self is the first entity (index 0)
        self_entity = attended_entities[..., 0, :]
        
        # Concatenate: [Batch, 128 + 50]
        combined_obs = torch.cat([self_entity, global_features], dim=-1)
        
        # Pass through head
        mean = self.head(combined_obs)
        std = self.log_std.clamp(-20, 0.5).exp().expand_as(mean)
        return mean, std

    def get_action(self, obs, summed_probs=True, deterministic=False):
        mean, std = self.get_output(obs)
        if deterministic:
            return mean.cpu(), 0

        distribution = Normal(loc=mean, scale=std)
        action = distribution.sample().clamp(min=-1, max=1)
        log_prob = distribution.log_prob(action)

        shape = log_prob.shape
        if summed_probs:
            if len(shape) > 1:
                log_prob = log_prob.sum(dim=-1)
            else:
                log_prob = log_prob.sum()

        return action.cpu(), log_prob.cpu()

    def get_backprop_data(self, obs, acts, summed_probs=True):
        mean, std = self.get_output(obs)
        distribution = Normal(loc=mean, scale=std)

        prob = distribution.log_prob(acts)
        if summed_probs:
            log_probs = prob.sum(dim=-1).to(self.device)
        else:
            log_probs = prob.to(self.device)

        entropy = distribution.entropy()
        entropy = entropy.mean().to(self.device)

        return log_probs, entropy


class AttentionApexValueEstimator(nn.Module):
    def __init__(self, input_shape, layer_sizes, device):
        super().__init__()
        self.device = device
        
        # Must match obs.py: 24 features (20 physics + 4 one-hot) + 1 mask = 25
        self.ENTITY_FEAT_SIZE = 25
        
        # Self (1) + Allies (2) + Enemies (3) = 6 players
        self.NUM_PLAYERS = 6
        
        self.D_MODEL = 128
        
        # Shared Encoder for all entities
        self.entity_encoder = nn.Sequential(
            nn.Linear(self.ENTITY_FEAT_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, self.D_MODEL),
            nn.ReLU()
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D_MODEL, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Global features: BoostPads(34) + Action(8) + RelBallPos(3) + RelBallVel(3) + ScoreDiff(1) + BallTouched(1) = 50
        self.global_feat_size = 50
        
        # Centralized Critic: mirrors policy input architecture
        self.combined_size = self.D_MODEL + self.global_feat_size

        assert len(layer_sizes) > 0, "AT LEAST ONE LAYER MUST BE SPECIFIED"
        layers = [nn.Linear(self.combined_size, layer_sizes[0]), nn.ReLU()]

        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(layer_sizes[-1], 1))
        
        self.head = nn.Sequential(*layers).to(self.device)
        self.entity_encoder = self.entity_encoder.to(self.device)
        self.transformer = self.transformer.to(self.device)

        # --- Orthogonal Initialization ---
        # Entity encoder: all Linear layers with gain=sqrt(2)
        for module in self.entity_encoder:
            init_weights(module, gain=np.sqrt(2))
        
        # Head: all Linear layers except the final output layer with gain=sqrt(2)
        head_linear_layers = [m for m in self.head if isinstance(m, nn.Linear)]
        for layer in head_linear_layers[:-1]:
            init_weights(layer, gain=np.sqrt(2))
        
        # Final layer of Critic: gain=1.0
        init_weights(head_linear_layers[-1], gain=1.0)
        
        # Transformer: leave PyTorch defaults (do NOT manually init)

    def _parse_obs(self, x):
        """Parse flat observation into entity features and global features."""
        total_entities = self.NUM_PLAYERS + 1  # 7
        entity_end = total_entities * self.ENTITY_FEAT_SIZE  # 7 * 25 = 175
        
        entity_features = []
        start_idx = 0
        for _ in range(total_entities):
            end_idx = start_idx + self.ENTITY_FEAT_SIZE
            entity_features.append(x[..., start_idx:end_idx])
            start_idx = end_idx
        
        global_features = x[..., entity_end:]
        
        return entity_features, global_features

    def forward(self, x):
        t = type(x)
        if t != torch.Tensor:
            if t != np.array:
                x = np.asarray(x)
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        entity_features, global_features = self._parse_obs(x)

        # Stack entities: [Batch, 7, 25]
        entities = torch.stack(entity_features, dim=-2)
        is_present = entities[..., -1:]
        
        # Encode
        encoded_entities = self.entity_encoder(entities) * is_present
        
        # Transformer
        batch_size = encoded_entities.shape[0]
        padding_mask = (is_present.squeeze(-1) == 0.0)

        max_chunk = 32768
        if batch_size > max_chunk:
            chunks = []
            for i in range(0, batch_size, max_chunk):
                enc_chunk = encoded_entities[i:i+max_chunk]
                mask_chunk = padding_mask[i:i+max_chunk]
                chunks.append(self.transformer(enc_chunk, src_key_padding_mask=mask_chunk))
            attended_entities = torch.cat(chunks, dim=0)
        else:
            attended_entities = self.transformer(encoded_entities, src_key_padding_mask=padding_mask)
        
        # Extract Self
        self_entity = attended_entities[..., 0, :]
        
        combined_obs = torch.cat([self_entity, global_features], dim=-1)
        
        return self.head(combined_obs)