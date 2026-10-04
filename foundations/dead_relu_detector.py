import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_fractions= []
        with torch.no_grad():
            for module in model.children():
                x = module(x)
                if isinstance(module,nn.ReLU):
                    negatives= x==0
                    neuron_sums= negatives.sum(axis=0)
                    dead_neurons= torch.sum(neuron_sums==x.shape[0])
                    dead_fraction= dead_neurons/x.shape[1]
                    dead_fractions.append(torch.round(dead_fraction,decimals=4))
        return dead_fractions
                

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        n= len(dead_fractions)
        for dead_fraction in dead_fractions:
            if dead_fraction>0.5: return 'use_leaky_relu'
        if dead_fractions[0]>0.3: return 'reinitialize'
        for i in range(1,n):
            if dead_fractions[i]<dead_fractions[i-1]: break
            if i==(n-1) and dead_fractions[i]>0.1: return 'reduce_learning_rate'
        return 'healthy'
