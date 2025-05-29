<<<<<<< HEAD
import numpy as np
from optimizers.base_optimizer import BaseOptimizer

class AdamOptimizer(BaseOptimizer):
    """
    Adam优化器实现
    """

    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        初始化Adam优化器

        参数:
            beta1 (float): 一阶矩估计的指数衰减率
            beta2 (float): 二阶矩估计的指数衰减率
            epsilon (float): 防止除零的小常数
        """
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads, learning_rate):
        """
        使用Adam算法更新参数

        参数:
            params (tuple): 参数元组 (W1, b1, W2, b2)
            grads (tuple): 梯度元组 (dW1, db1, dW2, db2)
            learning_rate (float): 学习率

        返回:
            tuple: 更新后的参数
        """
        W1, b1, W2, b2 = params
        dW1, db1, dW2, db2 = grads

        if self.m is None:
            # 初始化一阶矩向量和二阶矩向量
            self.m = {
                'W1': np.zeros_like(W1),
                'b1': np.zeros_like(b1),
                'W2': np.zeros_like(W2),
                'b2': np.zeros_like(b2)
            }
            self.v = {
                'W1': np.zeros_like(W1),
                'b1': np.zeros_like(b1),
                'W2': np.zeros_like(W2),
                'b2': np.zeros_like(b2)
            }

        self.t += 1

        # 更新一阶矩估计
        self.m['W1'] = self.beta1 * self.m['W1'] + (1 - self.beta1) * dW1
        self.m['b1'] = self.beta1 * self.m['b1'] + (1 - self.beta1) * db1
        self.m['W2'] = self.beta1 * self.m['W2'] + (1 - self.beta1) * dW2
        self.m['b2'] = self.beta1 * self.m['b2'] + (1 - self.beta1) * db2

        # 更新二阶矩估计
        self.v['W1'] = self.beta2 * self.v['W1'] + (1 - self.beta2) * (dW1 ** 2)
        self.v['b1'] = self.beta2 * self.v['b1'] + (1 - self.beta2) * (db1 ** 2)
        self.v['W2'] = self.beta2 * self.v['W2'] + (1 - self.beta2) * (dW2 ** 2)
        self.v['b2'] = self.beta2 * self.v['b2'] + (1 - self.beta2) * (db2 ** 2)

        # 计算偏差修正后的一阶矩和二阶矩估计
        m_hat_W1 = self.m['W1'] / (1 - self.beta1 ** self.t)
        m_hat_b1 = self.m['b1'] / (1 - self.beta1 ** self.t)
        m_hat_W2 = self.m['W2'] / (1 - self.beta1 ** self.t)
        m_hat_b2 = self.m['b2'] / (1 - self.beta1 ** self.t)

        v_hat_W1 = self.v['W1'] / (1 - self.beta2 ** self.t)
        v_hat_b1 = self.v['b1'] / (1 - self.beta2 ** self.t)
        v_hat_W2 = self.v['W2'] / (1 - self.beta2 ** self.t)
        v_hat_b2 = self.v['b2'] / (1 - self.beta2 ** self.t)

        # 更新参数
        W1 -= learning_rate * m_hat_W1 / (np.sqrt(v_hat_W1) + self.epsilon)
        b1 -= learning_rate * m_hat_b1 / (np.sqrt(v_hat_b1) + self.epsilon)
        W2 -= learning_rate * m_hat_W2 / (np.sqrt(v_hat_W2) + self.epsilon)
        b2 -= learning_rate * m_hat_b2 / (np.sqrt(v_hat_b2) + self.epsilon)

=======
import numpy as np
from optimizers.base_optimizer import BaseOptimizer

class AdamOptimizer(BaseOptimizer):
    """
    Adam优化器实现
    """

    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        初始化Adam优化器

        参数:
            beta1 (float): 一阶矩估计的指数衰减率
            beta2 (float): 二阶矩估计的指数衰减率
            epsilon (float): 防止除零的小常数
        """
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads, learning_rate):
        """
        使用Adam算法更新参数

        参数:
            params (tuple): 参数元组 (W1, b1, W2, b2)
            grads (tuple): 梯度元组 (dW1, db1, dW2, db2)
            learning_rate (float): 学习率

        返回:
            tuple: 更新后的参数
        """
        W1, b1, W2, b2 = params
        dW1, db1, dW2, db2 = grads

        if self.m is None:
            # 初始化一阶矩向量和二阶矩向量
            self.m = {
                'W1': np.zeros_like(W1),
                'b1': np.zeros_like(b1),
                'W2': np.zeros_like(W2),
                'b2': np.zeros_like(b2)
            }
            self.v = {
                'W1': np.zeros_like(W1),
                'b1': np.zeros_like(b1),
                'W2': np.zeros_like(W2),
                'b2': np.zeros_like(b2)
            }

        self.t += 1

        # 更新一阶矩估计
        self.m['W1'] = self.beta1 * self.m['W1'] + (1 - self.beta1) * dW1
        self.m['b1'] = self.beta1 * self.m['b1'] + (1 - self.beta1) * db1
        self.m['W2'] = self.beta1 * self.m['W2'] + (1 - self.beta1) * dW2
        self.m['b2'] = self.beta1 * self.m['b2'] + (1 - self.beta1) * db2

        # 更新二阶矩估计
        self.v['W1'] = self.beta2 * self.v['W1'] + (1 - self.beta2) * (dW1 ** 2)
        self.v['b1'] = self.beta2 * self.v['b1'] + (1 - self.beta2) * (db1 ** 2)
        self.v['W2'] = self.beta2 * self.v['W2'] + (1 - self.beta2) * (dW2 ** 2)
        self.v['b2'] = self.beta2 * self.v['b2'] + (1 - self.beta2) * (db2 ** 2)

        # 计算偏差修正后的一阶矩和二阶矩估计
        m_hat_W1 = self.m['W1'] / (1 - self.beta1 ** self.t)
        m_hat_b1 = self.m['b1'] / (1 - self.beta1 ** self.t)
        m_hat_W2 = self.m['W2'] / (1 - self.beta1 ** self.t)
        m_hat_b2 = self.m['b2'] / (1 - self.beta1 ** self.t)

        v_hat_W1 = self.v['W1'] / (1 - self.beta2 ** self.t)
        v_hat_b1 = self.v['b1'] / (1 - self.beta2 ** self.t)
        v_hat_W2 = self.v['W2'] / (1 - self.beta2 ** self.t)
        v_hat_b2 = self.v['b2'] / (1 - self.beta2 ** self.t)

        # 更新参数
        W1 -= learning_rate * m_hat_W1 / (np.sqrt(v_hat_W1) + self.epsilon)
        b1 -= learning_rate * m_hat_b1 / (np.sqrt(v_hat_b1) + self.epsilon)
        W2 -= learning_rate * m_hat_W2 / (np.sqrt(v_hat_W2) + self.epsilon)
        b2 -= learning_rate * m_hat_b2 / (np.sqrt(v_hat_b2) + self.epsilon)

>>>>>>> fbfedc385406b6556c748886c8ab26c2e95c54a6
        return W1, b1, W2, b2