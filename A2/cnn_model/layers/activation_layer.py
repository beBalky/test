import numpy as np

class ActivationLayer:
    def __init__(self, activation='relu'):
        self.activation = activation
    
    def forward(self, x):
        self.last_input = x
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
    
    def backward(self, d_out):
        if self.activation == 'relu':
            d_input = d_out.copy()
            d_input[self.last_input <= 0] = 0
            return d_input
        elif self.activation == 'sigmoid':
            sig = self.forward(self.last_input)
            return d_out * sig * (1 - sig)
        elif self.activation == 'tanh':
            tanh = self.forward(self.last_input)
            return d_out * (1 - tanh**2)


class ReLULayer:
    def forward(self, x):
        self.last_input = x
        return np.maximum(0, x)

    def backward(self, d_out):
        d_input = d_out.copy()
        d_input[self.last_input <= 0] = 0
        return d_input 