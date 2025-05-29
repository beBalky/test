<<<<<<< HEAD
import torch


class Config:
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

    # 设备相关
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型类型
    MODEL_TYPE = 'gru'  # 'lstm' or 'gru'

    # 注意力机制类型
    ATTENTION_TYPE = 'luong'  # 'luong' or 'bahdanau'

    # 路径相关
    MODEL_SAVE_PATH = './A6/checkpoints'

    # 翻译相关
    MAX_LENGTH = 100
=======
import torch


class Config:
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

    # 设备相关
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型类型
    MODEL_TYPE = 'gru'  # 'lstm' or 'gru'

    # 注意力机制类型
    ATTENTION_TYPE = 'luong'  # 'luong' or 'bahdanau'

    # 路径相关
    MODEL_SAVE_PATH = './A6/checkpoints'

    # 翻译相关
    MAX_LENGTH = 100
>>>>>>> 122082e17e4a59b84a279fae2b942a56c0dd3b0b
