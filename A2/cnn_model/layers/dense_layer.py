import numpy as np
from ..utils.regularization import l1_regularization, l2_regularization


class DenseLayer:
    def __init__(self, input_size, output_size, learning_rate=0.01, lam_l1=0.01, lam_l2=0.01):
        # 使用He初始化以改善训练
        scale = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(input_size, output_size) * scale
        self.bias = np.zeros((output_size, 1))  # 初始化为0
        self.learning_rate = learning_rate
        self.lam_l1 = lam_l1
        self.lam_l2 = lam_l2
    
    def forward(self, x):
        self.last_input = x
        return np.dot(x, self.weights) + self.bias.T
    
    def backward(self, d_out):
        batch_size = d_out.shape[0]
        
        # 计算权重梯度，包括正则化
        d_weights = np.dot(self.last_input.T, d_out) / batch_size
        d_weights += l1_regularization(self.weights, self.lam_l1)
        d_weights += l2_regularization(self.weights, self.lam_l2)
        
        # 计算偏置梯度
        d_bias = np.sum(d_out, axis=0).reshape(-1, 1) / batch_size
        
        # 计算输入梯度
        d_input = np.dot(d_out, self.weights.T)
        
        # 更新权重和偏置
        self.weights -= self.learning_rate * d_weights
        self.bias -= self.learning_rate * d_bias

        return d_input, d_weights, d_bias 