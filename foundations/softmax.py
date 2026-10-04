import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z-= np.max(z)
        e_z= np.exp(z)
        f_z= e_z/np.sum(e_z)
        return np.round(f_z,4)
