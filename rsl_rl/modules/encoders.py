# -*- coding: utf-8 -*-
# Created by Shuhan Wang on 2024/8/25.
#

import torch
import torch.nn as nn
from rsl_rl.modules import build_mlp, get_activation


class TeacherElevationMapEncoder(nn.Module):
    def __init__(self,
                 input_height: int,
                 input_width: int,
                 output_dim: int,
                 output_activation: nn.Module = None):
        super(TeacherElevationMapEncoder, self).__init__()
        self.encoder = nn.Sequential(
            self.conv_block(1, 32, 3, 1, 0, nn.ELU),
            self.conv_block(32, 64, 3, 1, 0, nn.ELU),
            self.conv_block(64, 128, 3, 1, 0, nn.ELU),
        )
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (input_height - 6) * (input_width - 6), output_dim),
            # nn.ELU(),
            # nn.Linear(256, output_dim),
        )
        self.input_height = input_height
        self.input_width = input_width
        self.output_activation = output_activation

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding, activation, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                  nn.BatchNorm2d(out_channels),
                  activation()]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x.reshape(-1, 1, self.input_height, self.input_width))
        x = self.bottleneck(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


class TeacherEncoder(nn.Module):
    def __init__(self, encoder_cfg):
        super(TeacherEncoder, self).__init__()
        
        if encoder_cfg["if_elevation_map_encode"]:
            elevation_map_encoder_cfg = encoder_cfg["elevation_map_encoder"]
            self.teacher_elevation_map_encoder = TeacherElevationMapEncoder(
                input_height=elevation_map_encoder_cfg["input_height"],
                input_width=elevation_map_encoder_cfg["input_width"],
                output_dim=elevation_map_encoder_cfg["output_dim"],  # 128
                output_activation=get_activation(elevation_map_encoder_cfg["output_activation"]))
        else:
            self.teacher_elevation_map_encoder = nn.Identity()

        if encoder_cfg["if_priv_model_encode"]:
            priv_model_encoder_cfg = encoder_cfg["priv_model_encoder"]
            self.teacher_priv_model_encoder = build_mlp(
                input_dim=priv_model_encoder_cfg["input_dim"],
                hidden_dims=priv_model_encoder_cfg["hidden_dims"],  # [128, 128]
                hidden_activation=get_activation(priv_model_encoder_cfg["hidden_activation"]),  # nn.ELU
                output_dim=priv_model_encoder_cfg["output_dim"],  # 64
                output_activation=get_activation(priv_model_encoder_cfg["output_activation"]))  # None
        else:
            self.teacher_priv_model_encoder = nn.Identity()
        
        if encoder_cfg["if_priv_state_encode"]:
            priv_state_encoder_cfg = encoder_cfg["priv_state_encoder"]
            self.teacher_priv_state_encoder = build_mlp(
                input_dim=priv_state_encoder_cfg["input_dim"],
                hidden_dims=priv_state_encoder_cfg["hidden_dims"],  # [64, 64]
                hidden_activation=get_activation(priv_state_encoder_cfg["hidden_activation"]),  # nn.ELU
                output_dim=priv_state_encoder_cfg["output_dim"],  # 32
                output_activation=get_activation(priv_state_encoder_cfg["output_activation"]))  # None
        else:
            self.teacher_priv_state_encoder = nn.Identity()
        
        if encoder_cfg["if_prop_state_encode"]:
            prop_state_encoder_cfg = encoder_cfg["prop_state_encoder"]
            self.teacher_prop_state_encoder = build_mlp(
                input_dim=prop_state_encoder_cfg["input_dim"],
                hidden_dims=prop_state_encoder_cfg["hidden_dims"],  # [256, 128]
                hidden_activation=get_activation(prop_state_encoder_cfg["hidden_activation"]),  # nn.ELU
                output_dim=prop_state_encoder_cfg["output_dim"],  # 128
                output_activation=get_activation(prop_state_encoder_cfg["output_activation"]))  # None
        else:
            self.teacher_prop_state_encoder = nn.Identity()
    
    def forward(self, elevation_map, priv_model, priv_state, prop_state):
        elevation_map_encoded = self.teacher_elevation_map_encoder(elevation_map)
        priv_model_encoded = self.teacher_priv_model_encoder(priv_model)
        priv_state_encoded = self.teacher_priv_state_encoder(priv_state)
        prop_state_encoded = self.teacher_prop_state_encoder(prop_state)
        return elevation_map_encoded, priv_model_encoded, priv_state_encoded, prop_state_encoded



