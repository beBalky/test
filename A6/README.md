<<<<<<< HEAD
# 基于注意力机制的神经机器翻译

这个项目实现了一个基于注意力机制的神经机器翻译系统，支持LSTM和GRU两种循环神经网络架构，以及Luong和Bahdanau两种注意力机制。项目使用PyTorch框架实现，专注于英德翻译任务。

## 项目结构

```
A6/
├── models/         # 模型实现目录
│   ├── __init__.py
│   ├── encoder.py  # 编码器实现
│   ├── decoder.py  # 解码器实现
│   ├── attention.py # 注意力机制实现
│   └── seq2seq.py  # 序列到序列模型
├── data/          # 数据集目录
│   ├── raw/       # 原始数据
│   └── processed/ # 预处理后的数据
├── checkpoints/   # 模型检查点保存目录
├── logs/          # 训练日志目录
├── config.py      # 配置文件
├── train.py       # 训练脚本
├── translate.py   # 翻译脚本
├── requirements.txt # 项目依赖
├── GRU-ATT.ipynb   # GRU模型训练notebook
└── LSTM-ATT.ipynb  # LSTM模型训练notebook
```

## 核心功能

### 模型架构
1. 编码器
   - 支持LSTM和GRU两种RNN架构
   - 双向RNN实现
   - 多层RNN支持
   - 词嵌入层

2. 解码器
   - 支持LSTM和GRU两种RNN架构
   - 注意力机制集成
   - 多层RNN支持
   - 输出层

3. 注意力机制
   - Luong注意力（点积、一般、拼接）
   - Bahdanau注意力
   - 可视化支持

### 训练特性
- 支持教师强制训练
- 实现梯度裁剪
- 学习率调度
- 早停机制
- 模型检查点
- 训练过程可视化

### 数据处理
- 支持多种数据预处理方法
- 实现数据清洗和标准化
- 支持批处理和动态批大小
- 使用spaCy进行分词
- 构建词汇表和词向量

## 配置说明

主要配置参数（在config.py中）：

```python
# 数据相关
BATCH_SIZE = 128
MIN_FREQ = 2

# 模型相关
EMBED_SIZE = 256
HIDDEN_SIZE = 512
N_LAYERS = 4
DROPOUT = 0.5

# 训练相关
EPOCHS = 10
LEARNING_RATE = 0.001
TEACHER_FORCING_RATIO = 0.5
CLIP_GRAD = 1.0

# 模型类型
MODEL_TYPE = 'lstm'  # 'lstm' or 'gru'
ATTENTION_TYPE = 'luong'  # 'luong' or 'bahdanau'
```

## 环境要求

- Python 3.7+
- PyTorch >= 2.0.0
- spaCy >= 3.0.0
- torchtext >= 0.14.0
- tqdm >= 4.65.0
- 德语和英语的spaCy语言模型

## 安装说明

1. 安装Python依赖：
```bash
pip install -r requirements.txt
```

2. 安装spaCy语言模型：
```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## 使用方法

### 训练模型

#### 使用脚本：
```bash
python train.py --model_type lstm --attention_type luong
```

## 模型性能

在WMT14英德翻译数据集上的表现：
- LSTM --> PPL: 33.676
=======
# 基于注意力机制的神经机器翻译

这个项目实现了一个基于注意力机制的神经机器翻译系统，支持LSTM和GRU两种循环神经网络架构，以及Luong和Bahdanau两种注意力机制。项目使用PyTorch框架实现，专注于英德翻译任务。

## 项目结构

```
A6/
├── models/         # 模型实现目录
│   ├── __init__.py
│   ├── encoder.py  # 编码器实现
│   ├── decoder.py  # 解码器实现
│   ├── attention.py # 注意力机制实现
│   └── seq2seq.py  # 序列到序列模型
├── data/          # 数据集目录
│   ├── raw/       # 原始数据
│   └── processed/ # 预处理后的数据
├── checkpoints/   # 模型检查点保存目录
├── logs/          # 训练日志目录
├── config.py      # 配置文件
├── train.py       # 训练脚本
├── translate.py   # 翻译脚本
├── requirements.txt # 项目依赖
├── GRU-ATT.ipynb   # GRU模型训练notebook
└── LSTM-ATT.ipynb  # LSTM模型训练notebook
```

## 核心功能

### 模型架构
1. 编码器
   - 支持LSTM和GRU两种RNN架构
   - 双向RNN实现
   - 多层RNN支持
   - 词嵌入层

2. 解码器
   - 支持LSTM和GRU两种RNN架构
   - 注意力机制集成
   - 多层RNN支持
   - 输出层

3. 注意力机制
   - Luong注意力（点积、一般、拼接）
   - Bahdanau注意力
   - 可视化支持

### 训练特性
- 支持教师强制训练
- 实现梯度裁剪
- 学习率调度
- 早停机制
- 模型检查点
- 训练过程可视化

### 数据处理
- 支持多种数据预处理方法
- 实现数据清洗和标准化
- 支持批处理和动态批大小
- 使用spaCy进行分词
- 构建词汇表和词向量

## 配置说明

主要配置参数（在config.py中）：

```python
# 数据相关
BATCH_SIZE = 128
MIN_FREQ = 2

# 模型相关
EMBED_SIZE = 256
HIDDEN_SIZE = 512
N_LAYERS = 4
DROPOUT = 0.5

# 训练相关
EPOCHS = 10
LEARNING_RATE = 0.001
TEACHER_FORCING_RATIO = 0.5
CLIP_GRAD = 1.0

# 模型类型
MODEL_TYPE = 'lstm'  # 'lstm' or 'gru'
ATTENTION_TYPE = 'luong'  # 'luong' or 'bahdanau'
```

## 环境要求

- Python 3.7+
- PyTorch >= 2.0.0
- spaCy >= 3.0.0
- torchtext >= 0.14.0
- tqdm >= 4.65.0
- 德语和英语的spaCy语言模型

## 安装说明

1. 安装Python依赖：
```bash
pip install -r requirements.txt
```

2. 安装spaCy语言模型：
```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## 使用方法

### 训练模型

#### 使用脚本：
```bash
python train.py --model_type lstm --attention_type luong
```

## 模型性能

在WMT14英德翻译数据集上的表现：
- LSTM --> PPL: 33.676
>>>>>>> 122082e17e4a59b84a279fae2b942a56c0dd3b0b
- GRU  --> PPL: 34.354