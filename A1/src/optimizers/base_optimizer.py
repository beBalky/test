<<<<<<< HEAD
class BaseOptimizer:
    """
    优化器的基类
    """

    def __init__(self):
        """
        初始化优化器
        """
        pass

    def update(self, params, grads, learning_rate):
        """
        更新参数

        参数:
            params (tuple): 参数元组 (W1, b1, W2, b2)
            grads (tuple): 梯度元组 (dW1, db1, dW2, db2)
            learning_rate (float): 学习率

        返回:
            tuple: 更新后的参数
        """
        W1, b1, W2, b2 = params
        dW1, db1, dW2, db2 = grads

        # 基础优化器使用简单的梯度下降
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

=======
class BaseOptimizer:
    """
    优化器的基类
    """

    def __init__(self):
        """
        初始化优化器
        """
        pass

    def update(self, params, grads, learning_rate):
        """
        更新参数

        参数:
            params (tuple): 参数元组 (W1, b1, W2, b2)
            grads (tuple): 梯度元组 (dW1, db1, dW2, db2)
            learning_rate (float): 学习率

        返回:
            tuple: 更新后的参数
        """
        W1, b1, W2, b2 = params
        dW1, db1, dW2, db2 = grads

        # 基础优化器使用简单的梯度下降
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

>>>>>>> fbfedc385406b6556c748886c8ab26c2e95c54a6
        return W1, b1, W2, b2