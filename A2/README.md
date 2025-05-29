<<<<<<< HEAD
# CNN从零实现

这个项目包含了一个使用NumPy从零实现的卷积神经网络(CNN)框架，基于AlexNet架构的简化版本。项目专注于实现CNN的核心组件，并应用于MNIST手写数字识别任务。

## 项目结构

```
A2/
├── cnn_model/              # CNN模型实现
│   ├── __init__.py
│   ├── models.py          # AlexNet模型定义
│   ├── data_loader.py     # 数据加载工具
│   │
│   ├── layers/            # 神经网络层实现
│   │   ├── __init__.py
│   │   ├── conv_layer.py  # 卷积层
│   │   ├── pool_layer.py  # 池化层
│   │   ├── activation_layer.py # 激活层
│   │   └── dense_layer.py # 全连接层
│   │
│   └── utils/             # 工具函数
│       ├── __init__.py
│       ├── regularization.py # 正则化函数
│       ├── loss.py        # 损失函数
│       └── activation.py  # 激活函数
│
├── data/                  # 数据集目录
├── checkpoints/          # 模型检查点
├── logs/                # 训练日志
├── main.py              # 主程序
└── README.md            # 项目文档
```

## 核心功能

### 网络架构
- 简化版AlexNet实现
- 支持可配置的网络结构
- 包含完整的前向传播和反向传播实现

### 层实现
1. 卷积层 (ConvLayer)
   - 支持可配置的卷积核大小
   - 实现填充（padding）和步长（stride）
   - 支持多通道输入和多卷积核

2. 池化层 (PoolLayer)
   - 最大池化操作
   - 可配置池化窗口大小和步长

3. 激活层 (ActivationLayer)
   - ReLU激活函数
   - Sigmoid激活函数
   - Tanh激活函数

4. 全连接层 (DenseLayer)
   - 支持He权重初始化
   - 实现L1和L2正则化
   - 可配置神经元数量

### 优化与训练
- 实现交叉熵损失函数
- 支持小批量随机梯度下降
- 包含学习率调整策略
- 实现早停机制
- 支持模型检查点保存和加载

### 性能优化
- 使用NumPy向量化操作提高性能
- 实现了高效的内存管理
- 支持批量处理以加速训练

## 环境要求

- Python 3.7+
- NumPy
- PyTorch (仅用于数据加载)
- tqdm (用于进度显示)
- matplotlib (用于可视化)

## 安装说明

```bash
pip install numpy torch tqdm matplotlib
```

## 使用方法

### 1. 训练模型

通过运行`main.py`文件来训练和测试模型：

```bash
python main.py
```

### 2. 配置参数

可以通过修改`main.py`中的参数来自定义训练过程：

```python
train_model(
    epochs=100,           # 训练轮数
    subset_size=1000,    # 训练数据子集大小
    data_dir='./data',  # 数据集目录
    verbose=True,      # 是否显示详细输出
    num_workers=4,      # 数据加载的并行工作线程数
    batch_size=256      # 批次大小
)
```

### 3. 模型结构

当前实现的简化版AlexNet结构：
- 输入层: 1x28x28 (MNIST图像)
- 第一卷积层: 16个3x3卷积核
- 第一池化层: 2x2最大池化
- 第二卷积层: 32个3x3卷积核
- 第二池化层: 2x2最大池化
- 全连接层1: 128个神经元
- 全连接层2: 10个神经元（输出层）

## 实现细节

### 卷积操作
- 使用im2col方法优化卷积计算
- 支持任意步长和填充大小
- 实现了梯度检查功能

### 正则化
- 支持L1正则化
- 支持L2正则化

### 损失函数
- 交叉熵损失

## 性能指标

在MNIST数据集上的表现：
- 快速评估准确率: 9.38%
- 测试集准确率：10.41%
- 单轮训练时间: ~2-3分钟 (取决于硬件配置)

## 注意事项

- 这是一个学习用途的实现，主要用于理解CNN的工作原理
- 大规模数据集训练建议使用PyTorch等深度学习框架
- 首次运行时会自动下载MNIST数据集
=======
# CNN从零实现

这个项目包含了一个使用NumPy从零实现的卷积神经网络(CNN)框架，基于AlexNet架构的简化版本。项目专注于实现CNN的核心组件，并应用于MNIST手写数字识别任务。

## 项目结构

```
A2/
├── cnn_model/              # CNN模型实现
│   ├── __init__.py
│   ├── models.py          # AlexNet模型定义
│   ├── data_loader.py     # 数据加载工具
│   │
│   ├── layers/            # 神经网络层实现
│   │   ├── __init__.py
│   │   ├── conv_layer.py  # 卷积层
│   │   ├── pool_layer.py  # 池化层
│   │   ├── activation_layer.py # 激活层
│   │   └── dense_layer.py # 全连接层
│   │
│   └── utils/             # 工具函数
│       ├── __init__.py
│       ├── regularization.py # 正则化函数
│       ├── loss.py        # 损失函数
│       └── activation.py  # 激活函数
│
├── data/                  # 数据集目录
├── checkpoints/          # 模型检查点
├── logs/                # 训练日志
├── main.py              # 主程序
└── README.md            # 项目文档
```

## 核心功能

### 网络架构
- 简化版AlexNet实现
- 支持可配置的网络结构
- 包含完整的前向传播和反向传播实现

### 层实现
1. 卷积层 (ConvLayer)
   - 支持可配置的卷积核大小
   - 实现填充（padding）和步长（stride）
   - 支持多通道输入和多卷积核

2. 池化层 (PoolLayer)
   - 最大池化操作
   - 可配置池化窗口大小和步长

3. 激活层 (ActivationLayer)
   - ReLU激活函数
   - Sigmoid激活函数
   - Tanh激活函数

4. 全连接层 (DenseLayer)
   - 支持He权重初始化
   - 实现L1和L2正则化
   - 可配置神经元数量

### 优化与训练
- 实现交叉熵损失函数
- 支持小批量随机梯度下降
- 包含学习率调整策略
- 实现早停机制
- 支持模型检查点保存和加载

### 性能优化
- 使用NumPy向量化操作提高性能
- 实现了高效的内存管理
- 支持批量处理以加速训练

## 环境要求

- Python 3.7+
- NumPy
- PyTorch (仅用于数据加载)
- tqdm (用于进度显示)
- matplotlib (用于可视化)

## 安装说明

```bash
pip install numpy torch tqdm matplotlib
```

## 使用方法

### 1. 训练模型

通过运行`main.py`文件来训练和测试模型：

```bash
python main.py
```

### 2. 配置参数

可以通过修改`main.py`中的参数来自定义训练过程：

```python
train_model(
    epochs=100,           # 训练轮数
    subset_size=1000,    # 训练数据子集大小
    data_dir='./data',  # 数据集目录
    verbose=True,      # 是否显示详细输出
    num_workers=4,      # 数据加载的并行工作线程数
    batch_size=256      # 批次大小
)
```

### 3. 模型结构

当前实现的简化版AlexNet结构：
- 输入层: 1x28x28 (MNIST图像)
- 第一卷积层: 16个3x3卷积核
- 第一池化层: 2x2最大池化
- 第二卷积层: 32个3x3卷积核
- 第二池化层: 2x2最大池化
- 全连接层1: 128个神经元
- 全连接层2: 10个神经元（输出层）

## 实现细节

### 卷积操作
- 使用im2col方法优化卷积计算
- 支持任意步长和填充大小
- 实现了梯度检查功能

### 正则化
- 支持L1正则化
- 支持L2正则化

### 损失函数
- 交叉熵损失

## 性能指标

在MNIST数据集上的表现：
- 快速评估准确率: 9.38%
- 测试集准确率：10.41%
- 单轮训练时间: ~2-3分钟 (取决于硬件配置)

## 注意事项

- 这是一个学习用途的实现，主要用于理解CNN的工作原理
- 大规模数据集训练建议使用PyTorch等深度学习框架
- 首次运行时会自动下载MNIST数据集
>>>>>>> 122082e17e4a59b84a279fae2b942a56c0dd3b0b
- 建议使用GPU加速训练（如果可用）