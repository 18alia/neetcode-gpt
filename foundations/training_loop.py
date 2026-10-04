import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        #print(X.shape)
        n_features= X.shape[1]
        n_samples= X.shape[0]
        w= np.zeros((n_features,1))
        b= 0
        y= np.reshape(y,(len(y),-1))
        for _ in range(epochs):
            y_hat= np.dot(X,w) + b
            #print(y_hat.shape)
            error= y_hat-y
            #print(error.shape)
            #loss= np.mean(error**2)
            #print(X.T.shape)
            dL_dw= 2/n_samples * np.dot(X.T,error)
            #print(dL_dw.shape)
            dL_db= 2/n_samples * np.sum(error)
            w = w - lr * dL_dw
            b = b - lr * dL_db
        w= w.flatten()
        return (np.round(w,5), round(b,5))
