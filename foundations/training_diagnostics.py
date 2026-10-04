import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats= []
        #print(model)
        with torch.no_grad():
            for module in model.children():
                print(module)
                x = module(x)
                if isinstance(module,nn.Linear):
                    negatives= x<=0
                    neuron_sums= negatives.sum(axis=0)
                    #print(neuron_sums)
                    dead_neurons= torch.sum(neuron_sums==x.shape[0])
                    #print(dead_neurons)
                    #print(x.shape)
                    dead_fraction= dead_neurons/x.shape[1]
                    layer_stats= {
                        'mean': round(torch.mean(x).item(),4),
                        'std': round(torch.std(x).item(),4),
                        'dead_fraction': round(dead_fraction.item(),4)
                    }
                    stats.append(layer_stats)
        #print(stats)
        return stats


    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        output= model(x)
        loss= nn.MSELoss()(output,y)
        loss.backward()
        stats=[]
        for module in model.children():
            if isinstance(module, nn.Linear):
                grad = module.weight.grad
                mean_val = round(grad.mean().item(), 4)
                std_val = round(grad.std().item(), 4)
                norm_val = round(torch.norm(grad).item(), 4)
                stats.append({'mean': mean_val, 'std': std_val, 'norm': norm_val})
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for layer in activation_stats:
            if layer['dead_fraction']>0.5:
                return 'dead_neurons'
        for layer in gradient_stats:
            if layer['norm']>1000:
                return 'exploding_gradients'
        for layer in gradient_stats:
            if layer['norm']<1e-5:
                return 'vanishing_gradients'
        for layer in activation_stats:
            if layer['std']<0.1: return 'vanishing_gradients'
            elif layer['std']>10.0: return 'exploding_gradients'
        return 'healthy'
