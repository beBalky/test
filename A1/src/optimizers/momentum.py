<<<<<<< HEAD
import numpy as np
from optimizers.base_optimizer import BaseOptimizer

class MomentumOptimizer(BaseOptimizer):
    """
    动量优化器实现
    """
    
    def __init__(self, momentum=0.9):
        """
        初始化动量优化器
        
        参数:
            momentum (float): 动量参数，默认为0.9
        """
        super().__init__()
        self.momentum = momentum
        self.velocity = None
    
    def update(self, params, grads, learning_rate):
        """
        使用动量法更新参数
        
        参数:
            params (tuple): 参数元组 (W1, b1, W2, b2)
            grads (tuple): 梯度元组 (dW1, db1, dW2, db2)
            learning_rate (float): 学习率
        
        返回:
            tuple: 更新后的参数
        """
        W1, b1, W2, b2 = params
        dW1, db1, dW2, db2 = grads
        
        # 首次使用时初始化速度
        if self.velocity is None:
            self.velocity = {
                'W1': np.zeros_like(W1),
                'b1': np.zeros_like(b1),
                'W2': np.zeros_like(W2),
                'b2': np.zeros_like(b2)
            }
        
        # 更新速度矩阵
        self.velocity['W1'] = self.momentum * self.velocity['W1'] - learning_rate * dW1
        self.velocity['b1'] = self.momentum * self.velocity['b1'] - learning_rate * db1
        self.velocity['W2'] = self.momentum * self.velocity['W2'] - learning_rate * dW2
        self.velocity['b2'] = self.momentum * self.velocity['b2'] - learning_rate * db2
        
        # 使用速度更新参数
        W1 += self.velocity['W1']
        b1 += self.velocity['b1']
        W2 += self.velocity['W2']
        b2 += self.velocity['b2']
        
=======
import numpy as np
from optimizers.base_optimizer import BaseOptimizer

class MomentumOptimizer(BaseOptimizer):
    """
    动量优化器实现
    """
    
    def __init__(self, momentum=0.9):
        """
        初始化动量优化器
        
        参数:
            momentum (float): 动量参数，默认为0.9
        """
        super().__init__()
        self.momentum = momentum
        self.velocity = None
    
    def update(self, params, grads, learning_rate):
        """
        使用动量法更新参数
        
        参数:
            params (tuple): 参数元组 (W1, b1, W2, b2)
            grads (tuple): 梯度元组 (dW1, db1, dW2, db2)
            learning_rate (float): 学习率
        
        返回:
            tuple: 更新后的参数
        """
        W1, b1, W2, b2 = params
        dW1, db1, dW2, db2 = grads
        
        # 首次使用时初始化速度
        if self.velocity is None:
            self.velocity = {
                'W1': np.zeros_like(W1),
                'b1': np.zeros_like(b1),
                'W2': np.zeros_like(W2),
                'b2': np.zeros_like(b2)
            }
        
        # 更新速度矩阵
        self.velocity['W1'] = self.momentum * self.velocity['W1'] - learning_rate * dW1
        self.velocity['b1'] = self.momentum * self.velocity['b1'] - learning_rate * db1
        self.velocity['W2'] = self.momentum * self.velocity['W2'] - learning_rate * dW2
        self.velocity['b2'] = self.momentum * self.velocity['b2'] - learning_rate * db2
        
        # 使用速度更新参数
        W1 += self.velocity['W1']
        b1 += self.velocity['b1']
        W2 += self.velocity['W2']
        b2 += self.velocity['b2']
        
>>>>>>> fbfedc385406b6556c748886c8ab26c2e95c54a6
        return W1, b1, W2, b2 