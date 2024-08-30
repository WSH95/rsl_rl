# -*- coding: utf-8 -*-
# Created by Shuhan Wang on 2024/8/30.
#

import torch
import torch.nn as nn
from rsl_rl.modules import build_mlp, get_activation
from typing import List


class StudentElevationMapExtractor(nn.Module):
    def __init__(self):
        super(StudentElevationMapExtractor, self).__init__()

    def forward(self, x):
        return None


class StudentPrivModelExtractor(nn.Module):
    def __init__(self):
        super(StudentPrivModelExtractor, self).__init__()

    def forward(self, x):
        return None


class StudentPrivStateExtractor(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 hidden_activation=nn.ELU,
                 output_dim: int = None,
                 output_activation=None):
        super(StudentPrivStateExtractor, self).__init__()
        self.input_dim = input_dim
        self.encoder = build_mlp(input_dim, hidden_dims, hidden_activation, output_dim, output_activation)

    def forward(self, x):
        return self.encoder(x[:, -self.input_dim:])


class StudentPropStateExtractor(nn.Module):
    def __init__(self):
        super(StudentPropStateExtractor, self).__init__()

    def forward(self, x):
        return None


class StudentExtractor(nn.Module):
    def __init__(self, encoder_cfg, obs_history_len, obs_dim):
        super(StudentExtractor, self).__init__()
        self.elevation_map_extractor = StudentElevationMapExtractor()
        priv_model_encoder_cfg = encoder_cfg["priv_model_encoder"]
        self.priv_model_extractor = ObsHistoryEncoder(
            obs_history_len=20,
            obs_dim=obs_dim,
            output_dim=priv_model_encoder_cfg["output_dim"],
            output_activation=get_activation(priv_model_encoder_cfg["output_activation"]))
        self.priv_state_extractor = StudentPrivStateExtractor(
            input_dim=obs_dim,
            hidden_dims=[128, 64],
            hidden_activation=nn.ELU,
            output_dim=3,
            output_activation=None)
        self.prop_state_extractor = StudentPropStateExtractor()

    def forward(self, x):
        elevation_map_extracted = self.elevation_map_extractor(x)
        priv_model_extracted = self.priv_model_extractor(x)
        priv_state_extracted = self.priv_state_extractor(x)
        prop_state_extracted = self.prop_state_extractor(x)
        return elevation_map_extracted, priv_model_extracted, priv_state_extracted, prop_state_extracted


class ObsHistoryEncoder(nn.Module):
    def __init__(self,
                 obs_history_len: int,
                 obs_dim: int,
                 output_dim: int,
                 output_activation: nn.Module = None):
        super(ObsHistoryEncoder, self).__init__()

        self.obs_history_len = obs_history_len
        self.obs_dim = obs_dim
        self.output_activation = output_activation

        channel_size = 10
        activation_fn = nn.ELU

        self.channel_regular_encoder = nn.Sequential(
            nn.Linear(obs_dim, 3*channel_size),
            activation_fn(),
        )

        self.conv_encoder = nn.Sequential(
            nn.Conv1d(in_channels=3*channel_size, out_channels=2*channel_size, kernel_size=6, stride=2),
            activation_fn(),
            nn.Conv1d(in_channels=2*channel_size, out_channels=channel_size, kernel_size=4, stride=2),
            activation_fn(),
        )
        num_conv_output = obs_history_len // 4 - 2
        if obs_history_len % 4 != 0:
            raise ValueError("obs_history_len must be divisible by 4")

        self.output_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_conv_output*channel_size, output_dim),
        )

    def forward(self, obs_history):
        num_batch = obs_history.shape[0]
        regular_obs = self.channel_regular_encoder(obs_history.reshape(-1, self.obs_dim))
        x = self.conv_encoder(regular_obs.reshape(num_batch, self.obs_history_len, -1).permute(0, 2, 1))
        x = self.output_mlp(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x

