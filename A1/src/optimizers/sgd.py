<<<<<<< HEAD
from optimizers.base_optimizer import BaseOptimizer

class SGDOptimizer(BaseOptimizer):
    """
    随机梯度下降优化器
    """

    def __init__(self, momentum=0.0):
        """
        初始化SGD优化器

        参数:
            momentum (float): 动量参数
        """
        super().__init__()
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads, learning_rate):
        """
        使用SGD更新参数

        参数:
            params (tuple): 参数元组 (W1, b1, W2, b2)
            grads (tuple): 梯度元组 (dW1, db1, dW2, db2)
            learning_rate (float): 学习率

        返回:
            tuple: 更新后的参数
        """
        W1, b1, W2, b2 = params
        dW1, db1, dW2, db2 = grads

        if self.momentum == 0.0:
            # 标准SGD
            return super().update(params, grads, learning_rate)
        else:
            # 带动量的SGD
            if self.velocity is None:
                # 初始化速度
                self.velocity = {
                    'W1': 0, 'b1': 0, 'W2': 0, 'b2': 0
                }

            # 更新速度
            self.velocity['W1'] = self.momentum * self.velocity['W1'] - learning_rate * dW1
            self.velocity['b1'] = self.momentum * self.velocity['b1'] - learning_rate * db1
            self.velocity['W2'] = self.momentum * self.velocity['W2'] - learning_rate * dW2
            self.velocity['b2'] = self.momentum * self.velocity['b2'] - learning_rate * db2

            # 更新参数
            W1 += self.velocity['W1']
            b1 += self.velocity['b1']
            W2 += self.velocity['W2']
            b2 += self.velocity['b2']

=======
from optimizers.base_optimizer import BaseOptimizer

class SGDOptimizer(BaseOptimizer):
    """
    随机梯度下降优化器
    """

    def __init__(self, momentum=0.0):
        """
        初始化SGD优化器

        参数:
            momentum (float): 动量参数
        """
        super().__init__()
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads, learning_rate):
        """
        使用SGD更新参数

        参数:
            params (tuple): 参数元组 (W1, b1, W2, b2)
            grads (tuple): 梯度元组 (dW1, db1, dW2, db2)
            learning_rate (float): 学习率

        返回:
            tuple: 更新后的参数
        """
        W1, b1, W2, b2 = params
        dW1, db1, dW2, db2 = grads

        if self.momentum == 0.0:
            # 标准SGD
            return super().update(params, grads, learning_rate)
        else:
            # 带动量的SGD
            if self.velocity is None:
                # 初始化速度
                self.velocity = {
                    'W1': 0, 'b1': 0, 'W2': 0, 'b2': 0
                }

            # 更新速度
            self.velocity['W1'] = self.momentum * self.velocity['W1'] - learning_rate * dW1
            self.velocity['b1'] = self.momentum * self.velocity['b1'] - learning_rate * db1
            self.velocity['W2'] = self.momentum * self.velocity['W2'] - learning_rate * dW2
            self.velocity['b2'] = self.momentum * self.velocity['b2'] - learning_rate * db2

            # 更新参数
            W1 += self.velocity['W1']
            b1 += self.velocity['b1']
            W2 += self.velocity['W2']
            b2 += self.velocity['b2']

>>>>>>> fbfedc385406b6556c748886c8ab26c2e95c54a6
            return W1, b1, W2, b2