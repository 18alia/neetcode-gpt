import torch
import torch.nn as nn
import math
from typing import List
import numpy as np


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std= math.sqrt(2.0/(fan_in+fan_out))
        weights= torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std= math.sqrt(2.0/(fan_in))
        weights= torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        weights= [0]*num_layers
        stds= [0]*num_layers
        for i in range(num_layers):
            if i==0:
                fan_in= input_dim
            else:
                fan_in= hidden_dim
            fan_out= hidden_dim
            if init_type == 'xavier':
                std = math.sqrt(2.0 / (fan_in+fan_out))
            elif init_type == 'kaiming':
                std = math.sqrt(2.0 / fan_in)
            else:
                std = 1.0
            weights[i]= torch.randn(fan_out,fan_in) * std
        h= torch.randn(1,input_dim)
        for i in range(num_layers):
            z= h @ weights[i].T
            h= torch.relu(z)
            stds[i]= round(h.std().item(),2)
        return stds
