<<<<<<< HEAD
import numpy as np

class MLP:
    """
    多层感知机类，实现具有一个隐藏层的神经网络
    """

    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid'):
        """
        初始化MLP模型

        参数:
            input_size (int): 输入特征维度
            hidden_size (int): 隐藏层神经元数量
            output_size (int): 输出维度
            activation (str): 激活函数类型，可选 'sigmoid' 或 'relu'
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 选择激活函数
        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == 'relu':
            self.activation = self.relu
            self.activation_derivative = self.relu_derivative
        else:
            print(f"不支持的激活函数: {activation}，使用sigmoid")
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative

        # 使用He初始化权重
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        # 使用He初始化输出层权重
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # 缓存前向传播的中间值
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

        # 缓存权重梯度
        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None

    def sigmoid(self, x):
        """
        Sigmoid激活函数

        参数:
            x (numpy.ndarray): 输入

        返回:
            numpy.ndarray: Sigmoid(x)
        """
        # 使用clip避免溢出
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        """
        Sigmoid函数的导数

        参数:
            x (numpy.ndarray): 输入

        返回:
            numpy.ndarray: Sigmoid的导数在x处的值
        """
        s = self.sigmoid(x)
        return s * (1 - s)

    def relu(self, x):
        """
        ReLU激活函数

        参数:
            x (numpy.ndarray): 输入

        返回:
            numpy.ndarray: ReLU(x)
        """
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """
        ReLU函数的导数

        参数:
            x (numpy.ndarray): 输入

        返回:
            numpy.ndarray: ReLU的导数在x处的值
        """
        return np.where(x > 0, 1, 0)

    def forward(self, X):
        """
        前向传播

        参数:
            X (numpy.ndarray): 输入特征，形状为(batch_size, input_size)

        返回:
            numpy.ndarray: 模型预测值，形状为(batch_size, output_size)
        """
        # 输入层到隐藏层
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)

        # 隐藏层到输出层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # 回归问题使用线性输出

        return self.a2

    def backward(self, X, y, output, reg_lambda=0.0):
        """
        反向传播计算梯度

        参数:
            X (numpy.ndarray): 输入特征，形状为(batch_size, input_size)
            y (numpy.ndarray): 真实标签，形状为(batch_size, output_size)
            output (numpy.ndarray): 模型预测值，形状为(batch_size, output_size)
            reg_lambda (float): L2正则化参数

        返回:
            tuple: 包含所有梯度的元组 (dW1, db1, dW2, db2)
        """
        batch_size = X.shape[0]

        # 计算输出层的误差
        delta2 = (output - y) / batch_size  # MSE损失的导数

        # 计算隐藏层的误差
        delta1 = np.dot(delta2, self.W2.T) * self.activation_derivative(self.z1)

        # 计算权重和偏置的梯度
        self.dW2 = np.dot(self.a1.T, delta2) + reg_lambda * self.W2
        self.db2 = np.sum(delta2, axis=0, keepdims=True)
        self.dW1 = np.dot(X.T, delta1) + reg_lambda * self.W1
        self.db1 = np.sum(delta1, axis=0, keepdims=True)

        return self.dW1, self.db1, self.dW2, self.db2

    def get_params(self):
        """
        获取模型参数

        返回:
            tuple: 包含所有参数的元组 (W1, b1, W2, b2)
        """
        return self.W1, self.b1, self.W2, self.b2

    def set_params(self, params):
        """
        设置模型参数

        参数:
            params (tuple): 包含所有参数的元组 (W1, b1, W2, b2)
        """
        self.W1, self.b1, self.W2, self.b2 = params

    def get_gradients(self):
        """
        获取模型参数的梯度

        返回:
            tuple: 包含所有梯度的元组 (dW1, db1, dW2, db2)
        """
=======
import numpy as np

class MLP:
    """
    多层感知机类，实现具有一个隐藏层的神经网络
    """

    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid'):
        """
        初始化MLP模型

        参数:
            input_size (int): 输入特征维度
            hidden_size (int): 隐藏层神经元数量
            output_size (int): 输出维度
            activation (str): 激活函数类型，可选 'sigmoid' 或 'relu'
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 选择激活函数
        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == 'relu':
            self.activation = self.relu
            self.activation_derivative = self.relu_derivative
        else:
            print(f"不支持的激活函数: {activation}，使用sigmoid")
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative

        # 使用He初始化权重
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        # 使用He初始化输出层权重
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # 缓存前向传播的中间值
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

        # 缓存权重梯度
        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None

    def sigmoid(self, x):
        """
        Sigmoid激活函数

        参数:
            x (numpy.ndarray): 输入

        返回:
            numpy.ndarray: Sigmoid(x)
        """
        # 使用clip避免溢出
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        """
        Sigmoid函数的导数

        参数:
            x (numpy.ndarray): 输入

        返回:
            numpy.ndarray: Sigmoid的导数在x处的值
        """
        s = self.sigmoid(x)
        return s * (1 - s)

    def relu(self, x):
        """
        ReLU激活函数

        参数:
            x (numpy.ndarray): 输入

        返回:
            numpy.ndarray: ReLU(x)
        """
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """
        ReLU函数的导数

        参数:
            x (numpy.ndarray): 输入

        返回:
            numpy.ndarray: ReLU的导数在x处的值
        """
        return np.where(x > 0, 1, 0)

    def forward(self, X):
        """
        前向传播

        参数:
            X (numpy.ndarray): 输入特征，形状为(batch_size, input_size)

        返回:
            numpy.ndarray: 模型预测值，形状为(batch_size, output_size)
        """
        # 输入层到隐藏层
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)

        # 隐藏层到输出层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # 回归问题使用线性输出

        return self.a2

    def backward(self, X, y, output, reg_lambda=0.0):
        """
        反向传播计算梯度

        参数:
            X (numpy.ndarray): 输入特征，形状为(batch_size, input_size)
            y (numpy.ndarray): 真实标签，形状为(batch_size, output_size)
            output (numpy.ndarray): 模型预测值，形状为(batch_size, output_size)
            reg_lambda (float): L2正则化参数

        返回:
            tuple: 包含所有梯度的元组 (dW1, db1, dW2, db2)
        """
        batch_size = X.shape[0]

        # 计算输出层的误差
        delta2 = (output - y) / batch_size  # MSE损失的导数

        # 计算隐藏层的误差
        delta1 = np.dot(delta2, self.W2.T) * self.activation_derivative(self.z1)

        # 计算权重和偏置的梯度
        self.dW2 = np.dot(self.a1.T, delta2) + reg_lambda * self.W2
        self.db2 = np.sum(delta2, axis=0, keepdims=True)
        self.dW1 = np.dot(X.T, delta1) + reg_lambda * self.W1
        self.db1 = np.sum(delta1, axis=0, keepdims=True)

        return self.dW1, self.db1, self.dW2, self.db2

    def get_params(self):
        """
        获取模型参数

        返回:
            tuple: 包含所有参数的元组 (W1, b1, W2, b2)
        """
        return self.W1, self.b1, self.W2, self.b2

    def set_params(self, params):
        """
        设置模型参数

        参数:
            params (tuple): 包含所有参数的元组 (W1, b1, W2, b2)
        """
        self.W1, self.b1, self.W2, self.b2 = params

    def get_gradients(self):
        """
        获取模型参数的梯度

        返回:
            tuple: 包含所有梯度的元组 (dW1, db1, dW2, db2)
        """
>>>>>>> fbfedc385406b6556c748886c8ab26c2e95c54a6
        return self.dW1, self.db1, self.dW2, self.db2