import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import functools

class MapContinuousToAction(nn.Module):
    def __init__(self, range_min=0.1, range_max=1):
        super().__init__()
        tanh_range = [-1, 1]
        self.m = (range_max - range_min) / (tanh_range[1] - tanh_range[0])
        self.b = range_min - tanh_range[0] * self.m

    def forward(self, x):
        n = x.shape[-1] // 2
        return x[..., :n], x[..., n:] * self.m + self.b

class AttentionApexPolicy(nn.Module):
    def __init__(self, input_shape, output_shape, layer_sizes, device, var_min=0.1, var_max=1.0):
        super().__init__()
        self.device = device
        self.affine_map = MapContinuousToAction(range_min=var_min, range_max=var_max)
        
        # Must match obs.py: 20 features + 1 mask = 21
        self.ENTITY_FEAT_SIZE = 21
        
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

        layers.append(nn.Linear(layer_sizes[-1], output_shape))
        layers.append(nn.Tanh()) # ContinuousPolicy ends in Tanh
        
        self.head = nn.Sequential(*layers).to(self.device)
        self.entity_encoder = self.entity_encoder.to(self.device)
        self.transformer = self.transformer.to(self.device)



    def get_output(self, obs):
        if type(obs) != torch.Tensor:
            if type(obs) != np.array:
                obs = np.asarray(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        player_features = []
        start_idx = 0
        
        for _ in range(self.NUM_PLAYERS):
            end_idx = start_idx + self.ENTITY_FEAT_SIZE
            p_feat = obs[..., start_idx:end_idx]
            player_features.append(p_feat)
            start_idx = end_idx
            
        # Extract ball (9 features)
        ball_feat_raw = obs[..., start_idx:start_idx+9]
        start_idx += 9
        
        # Pad ball to 21 features (9 values + 11 zeros + 1 mask)
        padding = torch.zeros((*ball_feat_raw.shape[:-1], 11), device=self.device)
        mask_ones = torch.ones((*ball_feat_raw.shape[:-1], 1), device=self.device)
        ball_feat = torch.cat([ball_feat_raw, padding, mask_ones], dim=-1)
        player_features.append(ball_feat)
        
        # Global features are everything remaining
        global_features = obs[..., start_idx:]

        # Stack entities: [Batch, 7, 21]
        entities = torch.stack(player_features, dim=-2)
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
        policy_output = self.head(combined_obs)
        return self.affine_map(policy_output)

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
        
        # Must match obs.py: 20 features + 1 mask = 21
        self.ENTITY_FEAT_SIZE = 21
        
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
        
        # Centralized Critic: Can see everything, but for now mirrors policy input architecture
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

    def forward(self, x):
        t = type(x)
        if t != torch.Tensor:
            if t != np.array:
                x = np.asarray(x)
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        player_features = []
        start_idx = 0
        
        for _ in range(self.NUM_PLAYERS):
            end_idx = start_idx + self.ENTITY_FEAT_SIZE
            p_feat = x[..., start_idx:end_idx]
            player_features.append(p_feat)
            start_idx = end_idx
            
        # Extract ball
        ball_feat_raw = x[..., start_idx:start_idx+9]
        start_idx += 9
        
        # Pad ball to 21
        padding = torch.zeros((*ball_feat_raw.shape[:-1], 11), device=self.device)
        mask_ones = torch.ones((*ball_feat_raw.shape[:-1], 1), device=self.device)
        ball_feat = torch.cat([ball_feat_raw, padding, mask_ones], dim=-1)
        player_features.append(ball_feat)
        
        global_features = x[..., start_idx:]

        # Stack entities: [Batch, 7, 21]
        entities = torch.stack(player_features, dim=-2)
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