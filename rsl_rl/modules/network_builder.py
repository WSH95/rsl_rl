# -*- coding: utf-8 -*-
# Created by Shuhan Wang on 2024/8/27.
#

import torch
import torch.nn as nn
from typing import List


def build_mlp(
        input_dim: int,
        hidden_dims: List[int],
        hidden_activation=nn.ELU,
        output_dim: int = None,
        output_activation=None) -> nn.Module:
    """
    Build a multi-layer perceptron (MLP) neural network
    :param input_dim: int: Dimension of the input
    :param hidden_dims: List[int]: List of dimensions of the hidden layers
    :param hidden_activation: nn.Module: Activation function for the hidden layers
    :param output_dim: int: Dimension of the output
    :param output_activation: nn.Module: Activation function for the output layer
    :return: nn.Module: MLP neural network
    """

    layers = []

    # Input layer
    layers.append(nn.Linear(input_dim, hidden_dims[0]))
    layers.append(hidden_activation())

    # Hidden layers
    for i in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        layers.append(hidden_activation())

    # Output layer
    if output_dim is not None:
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        if output_activation is not None:
            layers.append(output_activation())

    return nn.Sequential(*layers)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name is None:
        return None
    else:
        print("invalid activation function!")
        return None
