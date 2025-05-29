<<<<<<< HEAD
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math
import time
import os
from typing import Tuple
from datetime import datetime

from config import Config
from data.dataset import TranslationDataset
from models.encoder import EncoderLSTM, EncoderGRU
from models.decoder import DecoderLSTM, DecoderGRU
from models.attention import LuongAttention, BahdanauAttention
from models.seq2seq import Seq2Seq


def setup_logger(log_dir='./A6/logs'):
    """
    设置日志记录器

    参数:
        log_dir: 日志文件保存目录
    """
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件名，包含时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )

    return logging.getLogger()


def epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
    """计算训练时间"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_epoch(model: nn.Module,
                iterator: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                clip: float) -> float:
    """训练一个epoch"""
    model.train()
    epoch_loss = 0

    for batch in tqdm(iterator, desc='Training'):
        src, trg = batch

        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module) -> float:
    """评估模型"""
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in tqdm(iterator, desc='Evaluating'):
            src, trg = batch

            output = model(src, trg, 0)  # 关闭 teacher forcing

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def main():
    """主训练函数"""
    # 创建保存模型的目录
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        os.makedirs(Config.MODEL_SAVE_PATH)

    # 加载数据集
    dataset = TranslationDataset(
        min_freq=Config.MIN_FREQ,
        batch_size=Config.BATCH_SIZE,
        device=Config.DEVICE
    )
    train_loader, valid_loader, test_loader = dataset.get_dataloaders()
    src_vocab_size, trg_vocab_size = dataset.get_vocab_sizes()

    logger = setup_logger()
    logger.info(f"训练开始，数据集大小: {len(train_loader)}")
    logger.info(f"验证集大小: {len(valid_loader)}")
    logger.info(f"测试集大小: {len(test_loader)}")
    logger.info(f"源词汇表大小: {src_vocab_size}")
    logger.info(f"目标词汇表大小: {trg_vocab_size}")
    logger.info(f"模型类型: {Config.MODEL_TYPE}")
    logger.info(f"注意力类型: {Config.ATTENTION_TYPE}")
    logger.info(f"批量大小: {Config.BATCH_SIZE}")
    logger.info(f"最小词频: {Config.MIN_FREQ}")
    logger.info(f"嵌入大小: {Config.EMBED_SIZE}")
    logger.info(f"隐藏大小: {Config.HIDDEN_SIZE}")
    logger.info(f"层数: {Config.N_LAYERS}")
    logger.info(f"dropout: {Config.DROPOUT}")
    logger.info(f"学习率: {Config.LEARNING_RATE}")
    logger.info(f"教师强制比例: {Config.TEACHER_FORCING_RATIO}")
    logger.info(f"梯度裁剪: {Config.CLIP_GRAD}")

    # 创建注意力层
    if Config.ATTENTION_TYPE == 'luong':
        attention = LuongAttention(Config.HIDDEN_SIZE)
    else:
        attention = BahdanauAttention(Config.HIDDEN_SIZE)

    # 创建编码器和解码器
    if Config.MODEL_TYPE == 'lstm':
        encoder = EncoderLSTM(
            vocab_size=src_vocab_size,
            embed_size=Config.EMBED_SIZE,
            hidden_size=Config.HIDDEN_SIZE,
            n_layers=Config.N_LAYERS,
            dropout=Config.DROPOUT
        )
        decoder = DecoderLSTM(
            vocab_size=trg_vocab_size,
            embed_size=Config.EMBED_SIZE,
            hidden_size=Config.HIDDEN_SIZE,
            n_layers=Config.N_LAYERS,
            attention=attention,
            dropout=Config.DROPOUT
        )
    else:
        encoder = EncoderGRU(
            vocab_size=src_vocab_size,
            embed_size=Config.EMBED_SIZE,
            hidden_size=Config.HIDDEN_SIZE,
            n_layers=Config.N_LAYERS,
            dropout=Config.DROPOUT
        )
        decoder = DecoderGRU(
            vocab_size=trg_vocab_size,
            embed_size=Config.EMBED_SIZE,
            hidden_size=Config.HIDDEN_SIZE,
            n_layers=Config.N_LAYERS,
            attention=attention,
            dropout=Config.DROPOUT
        )

    # 创建Seq2Seq模型
    model = Seq2Seq(encoder, decoder, Config.DEVICE).to(Config.DEVICE)

    # 初始化模型参数
    def init_weights(m: nn.Module):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
    model.apply(init_weights)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.src_vocab['<pad>'])

    # 训练循环
    best_valid_loss = float('inf')

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, Config.CLIP_GRAD)
        valid_loss = evaluate(model, valid_loader, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            logger.info(
                f'保存最佳模型: {Config.MODEL_TYPE}-best-model.pt, 验证损失: {valid_loss:.3f}')
            torch.save(model.state_dict(),
                       os.path.join(Config.MODEL_SAVE_PATH, f'{Config.MODEL_TYPE}-best-model.pt'))

        logging.info(
            f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        logging.info(
            f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        logging.info(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    # 在测试集上评估
    model.load_state_dict(torch.load(os.path.join(
        Config.MODEL_SAVE_PATH, f'{Config.MODEL_TYPE}-best-model.pt')))
    test_loss = evaluate(model, test_loader, criterion)
    logger.info(
        f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}')


if __name__ == '__main__':
    main()
=======
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math
import time
import os
from typing import Tuple
from datetime import datetime

from config import Config
from data.dataset import TranslationDataset
from models.encoder import EncoderLSTM, EncoderGRU
from models.decoder import DecoderLSTM, DecoderGRU
from models.attention import LuongAttention, BahdanauAttention
from models.seq2seq import Seq2Seq


def setup_logger(log_dir='./A6/logs'):
    """
    设置日志记录器

    参数:
        log_dir: 日志文件保存目录
    """
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件名，包含时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )

    return logging.getLogger()


def epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
    """计算训练时间"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_epoch(model: nn.Module,
                iterator: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                clip: float) -> float:
    """训练一个epoch"""
    model.train()
    epoch_loss = 0

    for batch in tqdm(iterator, desc='Training'):
        src, trg = batch

        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module) -> float:
    """评估模型"""
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in tqdm(iterator, desc='Evaluating'):
            src, trg = batch

            output = model(src, trg, 0)  # 关闭 teacher forcing

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def main():
    """主训练函数"""
    # 创建保存模型的目录
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        os.makedirs(Config.MODEL_SAVE_PATH)

    # 加载数据集
    dataset = TranslationDataset(
        min_freq=Config.MIN_FREQ,
        batch_size=Config.BATCH_SIZE,
        device=Config.DEVICE
    )
    train_loader, valid_loader, test_loader = dataset.get_dataloaders()
    src_vocab_size, trg_vocab_size = dataset.get_vocab_sizes()

    logger = setup_logger()
    logger.info(f"训练开始，数据集大小: {len(train_loader)}")
    logger.info(f"验证集大小: {len(valid_loader)}")
    logger.info(f"测试集大小: {len(test_loader)}")
    logger.info(f"源词汇表大小: {src_vocab_size}")
    logger.info(f"目标词汇表大小: {trg_vocab_size}")
    logger.info(f"模型类型: {Config.MODEL_TYPE}")
    logger.info(f"注意力类型: {Config.ATTENTION_TYPE}")
    logger.info(f"批量大小: {Config.BATCH_SIZE}")
    logger.info(f"最小词频: {Config.MIN_FREQ}")
    logger.info(f"嵌入大小: {Config.EMBED_SIZE}")
    logger.info(f"隐藏大小: {Config.HIDDEN_SIZE}")
    logger.info(f"层数: {Config.N_LAYERS}")
    logger.info(f"dropout: {Config.DROPOUT}")
    logger.info(f"学习率: {Config.LEARNING_RATE}")
    logger.info(f"教师强制比例: {Config.TEACHER_FORCING_RATIO}")
    logger.info(f"梯度裁剪: {Config.CLIP_GRAD}")

    # 创建注意力层
    if Config.ATTENTION_TYPE == 'luong':
        attention = LuongAttention(Config.HIDDEN_SIZE)
    else:
        attention = BahdanauAttention(Config.HIDDEN_SIZE)

    # 创建编码器和解码器
    if Config.MODEL_TYPE == 'lstm':
        encoder = EncoderLSTM(
            vocab_size=src_vocab_size,
            embed_size=Config.EMBED_SIZE,
            hidden_size=Config.HIDDEN_SIZE,
            n_layers=Config.N_LAYERS,
            dropout=Config.DROPOUT
        )
        decoder = DecoderLSTM(
            vocab_size=trg_vocab_size,
            embed_size=Config.EMBED_SIZE,
            hidden_size=Config.HIDDEN_SIZE,
            n_layers=Config.N_LAYERS,
            attention=attention,
            dropout=Config.DROPOUT
        )
    else:
        encoder = EncoderGRU(
            vocab_size=src_vocab_size,
            embed_size=Config.EMBED_SIZE,
            hidden_size=Config.HIDDEN_SIZE,
            n_layers=Config.N_LAYERS,
            dropout=Config.DROPOUT
        )
        decoder = DecoderGRU(
            vocab_size=trg_vocab_size,
            embed_size=Config.EMBED_SIZE,
            hidden_size=Config.HIDDEN_SIZE,
            n_layers=Config.N_LAYERS,
            attention=attention,
            dropout=Config.DROPOUT
        )

    # 创建Seq2Seq模型
    model = Seq2Seq(encoder, decoder, Config.DEVICE).to(Config.DEVICE)

    # 初始化模型参数
    def init_weights(m: nn.Module):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
    model.apply(init_weights)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.src_vocab['<pad>'])

    # 训练循环
    best_valid_loss = float('inf')

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, Config.CLIP_GRAD)
        valid_loss = evaluate(model, valid_loader, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            logger.info(
                f'保存最佳模型: {Config.MODEL_TYPE}-best-model.pt, 验证损失: {valid_loss:.3f}')
            torch.save(model.state_dict(),
                       os.path.join(Config.MODEL_SAVE_PATH, f'{Config.MODEL_TYPE}-best-model.pt'))

        logging.info(
            f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        logging.info(
            f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        logging.info(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    # 在测试集上评估
    model.load_state_dict(torch.load(os.path.join(
        Config.MODEL_SAVE_PATH, f'{Config.MODEL_TYPE}-best-model.pt')))
    test_loss = evaluate(model, test_loader, criterion)
    logger.info(
        f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}')


if __name__ == '__main__':
    main()
>>>>>>> 122082e17e4a59b84a279fae2b942a56c0dd3b0b
