import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        if training:
            mu= np.mean(x, axis=0)
            var= np.var(x, axis=0)
            running_mean= [(1-momentum)*i for i in running_mean] + momentum*mu
            running_var= [(1-momentum)*i for i in running_var] + momentum*var
        else:
            mu= running_mean
            var= running_var
        x_hat= np.subtract(x,mu)/np.sqrt(np.add(var,eps))
        y= gamma * x_hat + beta
        return (np.round(y,4),np.round(running_mean,4),np.round(running_var,4))

