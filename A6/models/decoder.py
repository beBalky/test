<<<<<<< HEAD
import torch
import torch.nn as nn
from typing import Tuple, Optional
from .attention import LuongAttention, BahdanauAttention


class DecoderLSTM(nn.Module):
    """LSTM解码器实现"""
    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 hidden_size: int,
                 n_layers: int,
                 attention: Optional[nn.Module] = None,
                 dropout: float = 0.5):
        """
        初始化LSTM解码器
        
        Args:
            vocab_size: 词汇表大小
            embed_size: 词嵌入维度
            hidden_size: 隐藏状态维度
            n_layers: LSTM层数
            attention: 注意力层
            dropout: Dropout比率
        """
        super(DecoderLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=1)
        self.attention = attention
        
        # 如果使用注意力机制，输入维度需要增加context vector的维度
        rnn_input_size = embed_size + hidden_size if attention else embed_size
        
        self.lstm = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # 输出层输入维度包含：词嵌入、RNN输出、context vector
        self.fc_out = nn.Linear(embed_size + hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                input: torch.Tensor,
                hidden: Tuple[torch.Tensor, torch.Tensor],
                encoder_outputs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            input: 目标语言输入 [batch_size]
            hidden: (h_n, c_n) 上一时间步的隐藏状态和记忆单元
            encoder_outputs: 编码器输出 [batch_size, src_len, hidden_size]
            
        Returns:
            output: 当前时间步的输出 [batch_size, vocab_size]
            hidden: 当前时间步的隐藏状态和记忆单元
        """
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, embed_size]
        
        if self.attention is not None:
            # 计算注意力
            context_vector, _ = self.attention(hidden[0][-1], encoder_outputs)
            # 拼接词嵌入和上下文向量
            rnn_input = torch.cat((embedded, context_vector.unsqueeze(1)), dim=2)
        else:
            rnn_input = embedded
            
        # 解码器RNN前向传播
        output, hidden = self.lstm(rnn_input, hidden)
        
        # 准备输出层的输入
        output = output.squeeze(1)  # [batch_size, hidden_size]
        embedded = embedded.squeeze(1)  # [batch_size, embed_size]
        if self.attention is not None:
            context_vector = context_vector  # [batch_size, hidden_size]
            output = self.fc_out(torch.cat((output, embedded, context_vector), dim=1))
        else:
            output = self.fc_out(torch.cat((output, embedded), dim=1))
            
        return output, hidden


class DecoderGRU(nn.Module):
    """GRU解码器实现"""
    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 hidden_size: int,
                 n_layers: int,
                 attention: Optional[nn.Module] = None,
                 dropout: float = 0.5):
        """
        初始化GRU解码器
        
        Args:
            vocab_size: 词汇表大小
            embed_size: 词嵌入维度
            hidden_size: 隐藏状态维度
            n_layers: GRU层数
            attention: 注意力层
            dropout: Dropout比率
        """
        super(DecoderGRU, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=1)
        self.attention = attention
        
        # 如果使用注意力机制，输入维度需要增加context vector的维度
        rnn_input_size = embed_size + hidden_size if attention else embed_size
        
        self.gru = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # 输出层输入维度包含：词嵌入、RNN输出、context vector
        self.fc_out = nn.Linear(embed_size + hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                input: torch.Tensor,
                hidden: torch.Tensor,
                encoder_outputs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            input: 目标语言输入 [batch_size]
            hidden: 上一时间步的隐藏状态
            encoder_outputs: 编码器输出 [batch_size, src_len, hidden_size]
            
        Returns:
            output: 当前时间步的输出 [batch_size, vocab_size]
            hidden: 当前时间步的隐藏状态
        """
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, embed_size]
        
        if self.attention is not None:
            # 计算注意力
            context_vector, _ = self.attention(hidden[-1], encoder_outputs)
            # 拼接词嵌入和上下文向量
            rnn_input = torch.cat((embedded, context_vector.unsqueeze(1)), dim=2)
        else:
            rnn_input = embedded
            
        # 解码器RNN前向传播
        output, hidden = self.gru(rnn_input, hidden)
        
        # 准备输出层的输入
        output = output.squeeze(1)  # [batch_size, hidden_size]
        embedded = embedded.squeeze(1)  # [batch_size, embed_size]
        if self.attention is not None:
            context_vector = context_vector  # [batch_size, hidden_size]
            output = self.fc_out(torch.cat((output, embedded, context_vector), dim=1))
        else:
            output = self.fc_out(torch.cat((output, embedded), dim=1))
            
=======
import torch
import torch.nn as nn
from typing import Tuple, Optional
from .attention import LuongAttention, BahdanauAttention


class DecoderLSTM(nn.Module):
    """LSTM解码器实现"""
    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 hidden_size: int,
                 n_layers: int,
                 attention: Optional[nn.Module] = None,
                 dropout: float = 0.5):
        """
        初始化LSTM解码器
        
        Args:
            vocab_size: 词汇表大小
            embed_size: 词嵌入维度
            hidden_size: 隐藏状态维度
            n_layers: LSTM层数
            attention: 注意力层
            dropout: Dropout比率
        """
        super(DecoderLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=1)
        self.attention = attention
        
        # 如果使用注意力机制，输入维度需要增加context vector的维度
        rnn_input_size = embed_size + hidden_size if attention else embed_size
        
        self.lstm = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # 输出层输入维度包含：词嵌入、RNN输出、context vector
        self.fc_out = nn.Linear(embed_size + hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                input: torch.Tensor,
                hidden: Tuple[torch.Tensor, torch.Tensor],
                encoder_outputs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            input: 目标语言输入 [batch_size]
            hidden: (h_n, c_n) 上一时间步的隐藏状态和记忆单元
            encoder_outputs: 编码器输出 [batch_size, src_len, hidden_size]
            
        Returns:
            output: 当前时间步的输出 [batch_size, vocab_size]
            hidden: 当前时间步的隐藏状态和记忆单元
        """
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, embed_size]
        
        if self.attention is not None:
            # 计算注意力
            context_vector, _ = self.attention(hidden[0][-1], encoder_outputs)
            # 拼接词嵌入和上下文向量
            rnn_input = torch.cat((embedded, context_vector.unsqueeze(1)), dim=2)
        else:
            rnn_input = embedded
            
        # 解码器RNN前向传播
        output, hidden = self.lstm(rnn_input, hidden)
        
        # 准备输出层的输入
        output = output.squeeze(1)  # [batch_size, hidden_size]
        embedded = embedded.squeeze(1)  # [batch_size, embed_size]
        if self.attention is not None:
            context_vector = context_vector  # [batch_size, hidden_size]
            output = self.fc_out(torch.cat((output, embedded, context_vector), dim=1))
        else:
            output = self.fc_out(torch.cat((output, embedded), dim=1))
            
        return output, hidden


class DecoderGRU(nn.Module):
    """GRU解码器实现"""
    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 hidden_size: int,
                 n_layers: int,
                 attention: Optional[nn.Module] = None,
                 dropout: float = 0.5):
        """
        初始化GRU解码器
        
        Args:
            vocab_size: 词汇表大小
            embed_size: 词嵌入维度
            hidden_size: 隐藏状态维度
            n_layers: GRU层数
            attention: 注意力层
            dropout: Dropout比率
        """
        super(DecoderGRU, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=1)
        self.attention = attention
        
        # 如果使用注意力机制，输入维度需要增加context vector的维度
        rnn_input_size = embed_size + hidden_size if attention else embed_size
        
        self.gru = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # 输出层输入维度包含：词嵌入、RNN输出、context vector
        self.fc_out = nn.Linear(embed_size + hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                input: torch.Tensor,
                hidden: torch.Tensor,
                encoder_outputs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            input: 目标语言输入 [batch_size]
            hidden: 上一时间步的隐藏状态
            encoder_outputs: 编码器输出 [batch_size, src_len, hidden_size]
            
        Returns:
            output: 当前时间步的输出 [batch_size, vocab_size]
            hidden: 当前时间步的隐藏状态
        """
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, embed_size]
        
        if self.attention is not None:
            # 计算注意力
            context_vector, _ = self.attention(hidden[-1], encoder_outputs)
            # 拼接词嵌入和上下文向量
            rnn_input = torch.cat((embedded, context_vector.unsqueeze(1)), dim=2)
        else:
            rnn_input = embedded
            
        # 解码器RNN前向传播
        output, hidden = self.gru(rnn_input, hidden)
        
        # 准备输出层的输入
        output = output.squeeze(1)  # [batch_size, hidden_size]
        embedded = embedded.squeeze(1)  # [batch_size, embed_size]
        if self.attention is not None:
            context_vector = context_vector  # [batch_size, hidden_size]
            output = self.fc_out(torch.cat((output, embedded, context_vector), dim=1))
        else:
            output = self.fc_out(torch.cat((output, embedded), dim=1))
            
>>>>>>> 122082e17e4a59b84a279fae2b942a56c0dd3b0b
        return output, hidden 