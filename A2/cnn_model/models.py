import numpy as np
import time
import logging
from .layers import ConvLayer, PoolLayer, ActivationLayer, DenseLayer
from .utils.activation import softmax


class AlexNet:
    def __init__(self, learning_rate=0.001, verbose=False):
        self.learning_rate = learning_rate
        self.verbose = verbose

        # 优化网络结构，减少卷积层数和过滤器数量以加快训练
        self.conv1 = ConvLayer(input_channels=1, num_filters=16,
                               kernel_size=3, stride=1, padding=1, learning_rate=learning_rate)
        self.relu1 = ActivationLayer(activation='relu')
        self.pool1 = PoolLayer(pool_size=2, stride=2)

        self.conv2 = ConvLayer(input_channels=16, num_filters=32,
                               kernel_size=3, stride=1, padding=1, learning_rate=learning_rate)
        self.relu2 = ActivationLayer(activation='relu')
        self.pool2 = PoolLayer(pool_size=2, stride=2)

        # 减少第三层卷积
        self.fc1 = DenseLayer(input_size=32 * 7 * 7,
                              output_size=128, learning_rate=learning_rate)
        self.relu3 = ActivationLayer(activation='relu')

        self.fc2 = DenseLayer(
            input_size=128, output_size=10, learning_rate=learning_rate)

        self.layer_times = {'conv1': 0, 'relu1': 0, 'pool1': 0,
                            'conv2': 0, 'relu2': 0, 'pool2': 0,
                            'fc1': 0, 'relu3': 0, 'fc2': 0, 'softmax': 0}

        # 用于调试的形状跟踪
        self.shape_tracking = {}

    def forward(self, x):
        # 输入形状验证
        if self.verbose:
            print(f"\n输入形状: {x.shape}")
            # 检查输入维度
            if len(x.shape) != 4:
                print(
                    f"警告: 预期输入维度为4 (batch, channels, height, width)，实际为 {len(x.shape)}")

        # 记录每一层的前向传播时间
        start_time = time.time()
        x = self.conv1.forward(x)
        self.layer_times['conv1'] += time.time() - start_time
        if self.verbose:
            print(f"conv1输出形状: {x.shape}")

        start_time = time.time()
        x = self.relu1.forward(x)
        self.layer_times['relu1'] += time.time() - start_time

        start_time = time.time()
        x = self.pool1.forward(x)
        self.layer_times['pool1'] += time.time() - start_time
        if self.verbose:
            print(f"pool1输出形状: {x.shape}")

        start_time = time.time()
        x = self.conv2.forward(x)
        self.layer_times['conv2'] += time.time() - start_time

        start_time = time.time()
        x = self.relu2.forward(x)
        self.layer_times['relu2'] += time.time() - start_time

        start_time = time.time()
        x = self.pool2.forward(x)
        self.layer_times['pool2'] += time.time() - start_time
        if self.verbose:
            print(f"pool2输出形状: {x.shape}")

        # 保存展平前的形状以便反向传播
        self.feature_shape = x.shape

        # 展平特征图
        x_flat = x.reshape(x.shape[0], -1)
        if self.verbose:
            print(f"展平后形状: {x_flat.shape}")

        start_time = time.time()
        x = self.fc1.forward(x_flat)
        self.layer_times['fc1'] += time.time() - start_time
        if self.verbose:
            print(f"fc1输出形状: {x.shape}")

        start_time = time.time()
        x = self.relu3.forward(x)
        self.layer_times['relu3'] += time.time() - start_time

        start_time = time.time()
        x = self.fc2.forward(x)
        self.layer_times['fc2'] += time.time() - start_time
        if self.verbose:
            print(f"fc2输出形状: {x.shape}")

        start_time = time.time()
        output = softmax(x)
        self.layer_times['softmax'] += time.time() - start_time
        if self.verbose:
            print(f"最终输出形状: {output.shape}\n")

        return output

    def backward(self, d_out):
        if self.verbose:
            print(f"\n反向传播起始梯度形状: {d_out.shape}")

        d_out = self.fc2.backward(d_out)[0]
        if self.verbose:
            print(f"fc2反向传播后梯度形状: {d_out.shape}")

        d_out = self.relu3.backward(d_out)
        d_out = self.fc1.backward(d_out)[0]
        if self.verbose:
            print(f"fc1反向传播后梯度形状: {d_out.shape}")

        # 重塑为卷积层输出的形状
        d_out = d_out.reshape(self.feature_shape)
        if self.verbose:
            print(f"重塑为特征图形状: {d_out.shape}")

        d_out = self.pool2.backward(d_out)
        if self.verbose:
            print(f"pool2反向传播后梯度形状: {d_out.shape}")

        d_out = self.relu2.backward(d_out)
        d_out = self.conv2.backward(d_out)[0]
        if self.verbose:
            print(f"conv2反向传播后梯度形状: {d_out.shape}")

        d_out = self.pool1.backward(d_out)
        if self.verbose:
            print(f"pool1反向传播后梯度形状: {d_out.shape}")

        d_out = self.relu1.backward(d_out)
        d_out = self.conv1.backward(d_out)[0]
        if self.verbose:
            print(f"conv1反向传播后梯度形状: {d_out.shape}\n")

    def print_layer_times(self):
        """打印各层的计算时间，帮助识别性能瓶颈"""
        total_time = sum(self.layer_times.values())
        print("\n层计算时间分析:")
        for layer_name, layer_time in self.layer_times.items():
            if total_time > 0:
                percentage = (layer_time / total_time) * 100
                print(
                    f"  - {layer_name}: {layer_time:.4f}秒 ({percentage:.1f}%)")
            else:
                print(f"  - {layer_name}: {layer_time:.4f}秒")

    def save_state(self):
        """
        保存模型的状态（权重和偏置）
        
        返回:
            包含所有层参数的字典
        """
        state = {
            'conv1': {
                'weights': self.conv1.weights,
                'bias': self.conv1.bias
            },
            'conv2': {
                'weights': self.conv2.weights,
                'bias': self.conv2.bias
            },
            'fc1': {
                'weights': self.fc1.weights,
                'bias': self.fc1.bias
            },
            'fc2': {
                'weights': self.fc2.weights,
                'bias': self.fc2.bias
            },
            'learning_rate': self.learning_rate,
            'verbose': self.verbose
        }
        return state

    def load_state(self, state):
        """
        从保存的状态加载模型参数
        
        参数:
            state: 包含模型参数的字典
        """
        # 加载卷积层1的参数
        self.conv1.weights = state['conv1']['weights']
        self.conv1.bias = state['conv1']['bias']

        # 加载卷积层2的参数
        self.conv2.weights = state['conv2']['weights']
        self.conv2.bias = state['conv2']['bias']

        # 加载全连接层1的参数
        self.fc1.weights = state['fc1']['weights']
        self.fc1.bias = state['fc1']['bias']

        # 加载全连接层2的参数
        self.fc2.weights = state['fc2']['weights']
        self.fc2.bias = state['fc2']['bias']

        # 加载其他参数
        self.learning_rate = state.get('learning_rate', self.learning_rate)
        self.verbose = state.get('verbose', self.verbose)
