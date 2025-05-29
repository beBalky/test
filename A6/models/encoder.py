<<<<<<< HEAD
import torch
import torch.nn as nn
from typing import Tuple


class EncoderLSTM(nn.Module):
    """LSTM编码器实现"""
    def __init__(self, 
                 vocab_size: int,
                 embed_size: int,
                 hidden_size: int,
                 n_layers: int,
                 dropout: float = 0.5):
        """
        初始化LSTM编码器
        
        Args:
            vocab_size: 词汇表大小
            embed_size: 词嵌入维度
            hidden_size: 隐藏状态维度
            n_layers: LSTM层数
            dropout: Dropout比率
        """
        super(EncoderLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=1)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            src: 源语言序列 [batch_size, seq_len]
            
        Returns:
            encoder_outputs: 所有时间步的隐藏状态 [batch_size, seq_len, hidden_size]
            (h_n, c_n): 最后时间步的隐藏状态和记忆单元
        """
        # [batch_size, seq_len, embed_size]
        embedded = self.dropout(self.embedding(src))
        
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        # h_n: [n_layers, batch_size, hidden_size]
        # c_n: [n_layers, batch_size, hidden_size]
        encoder_outputs, (h_n, c_n) = self.lstm(embedded)
        
        return encoder_outputs, (h_n, c_n)


class EncoderGRU(nn.Module):
    """GRU编码器实现"""
    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 hidden_size: int,
                 n_layers: int,
                 dropout: float = 0.5):
        """
        初始化GRU编码器
        
        Args:
            vocab_size: 词汇表大小
            embed_size: 词嵌入维度
            hidden_size: 隐藏状态维度
            n_layers: GRU层数
            dropout: Dropout比率
        """
        super(EncoderGRU, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=1)
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            src: 源语言序列 [batch_size, seq_len]
            
        Returns:
            encoder_outputs: 所有时间步的隐藏状态 [batch_size, seq_len, hidden_size]
            h_n: 最后时间步的隐藏状态 [n_layers, batch_size, hidden_size]
        """
        # [batch_size, seq_len, embed_size]
        embedded = self.dropout(self.embedding(src))
        
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        # h_n: [n_layers, batch_size, hidden_size]
        encoder_outputs, h_n = self.gru(embedded)
        
=======
import torch
import torch.nn as nn
from typing import Tuple


class EncoderLSTM(nn.Module):
    """LSTM编码器实现"""
    def __init__(self, 
                 vocab_size: int,
                 embed_size: int,
                 hidden_size: int,
                 n_layers: int,
                 dropout: float = 0.5):
        """
        初始化LSTM编码器
        
        Args:
            vocab_size: 词汇表大小
            embed_size: 词嵌入维度
            hidden_size: 隐藏状态维度
            n_layers: LSTM层数
            dropout: Dropout比率
        """
        super(EncoderLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=1)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            src: 源语言序列 [batch_size, seq_len]
            
        Returns:
            encoder_outputs: 所有时间步的隐藏状态 [batch_size, seq_len, hidden_size]
            (h_n, c_n): 最后时间步的隐藏状态和记忆单元
        """
        # [batch_size, seq_len, embed_size]
        embedded = self.dropout(self.embedding(src))
        
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        # h_n: [n_layers, batch_size, hidden_size]
        # c_n: [n_layers, batch_size, hidden_size]
        encoder_outputs, (h_n, c_n) = self.lstm(embedded)
        
        return encoder_outputs, (h_n, c_n)


class EncoderGRU(nn.Module):
    """GRU编码器实现"""
    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 hidden_size: int,
                 n_layers: int,
                 dropout: float = 0.5):
        """
        初始化GRU编码器
        
        Args:
            vocab_size: 词汇表大小
            embed_size: 词嵌入维度
            hidden_size: 隐藏状态维度
            n_layers: GRU层数
            dropout: Dropout比率
        """
        super(EncoderGRU, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=1)
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            src: 源语言序列 [batch_size, seq_len]
            
        Returns:
            encoder_outputs: 所有时间步的隐藏状态 [batch_size, seq_len, hidden_size]
            h_n: 最后时间步的隐藏状态 [n_layers, batch_size, hidden_size]
        """
        # [batch_size, seq_len, embed_size]
        embedded = self.dropout(self.embedding(src))
        
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        # h_n: [n_layers, batch_size, hidden_size]
        encoder_outputs, h_n = self.gru(embedded)
        
>>>>>>> 122082e17e4a59b84a279fae2b942a56c0dd3b0b
        return encoder_outputs, h_n 