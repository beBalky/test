<<<<<<< HEAD
import torch
import torch.nn as nn


class LuongAttention(nn.Module):
    """Luong注意力机制实现"""
    def __init__(self, hidden_size: int, method: str = "dot"):
        """
        初始化Luong注意力层
        
        Args:
            hidden_size: 隐藏状态维度
            method: 注意力计算方法，可选 "dot" 或 "general"
        """
        super(LuongAttention, self).__init__()
        self.method = method
        if method == "general":
            self.W = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        """
        计算注意力权重和上下文向量
        
        Args:
            hidden: 解码器当前隐藏状态 [batch_size, hidden_size]
            encoder_outputs: 编码器所有隐藏状态 [batch_size, seq_len, hidden_size]
            
        Returns:
            context_vector: 上下文向量 [batch_size, hidden_size]
            attention_weights: 注意力权重 [batch_size, seq_len, 1]
        """
        if self.method == "dot":
            # [batch_size, seq_len, 1]
            score = torch.bmm(encoder_outputs, hidden.unsqueeze(2))
        elif self.method == "general":
            # [batch_size, seq_len, hidden_size]
            score = self.W(encoder_outputs)
            # [batch_size, seq_len, 1]
            score = torch.bmm(score, hidden.unsqueeze(2))
        
        attention_weights = torch.softmax(score, dim=1)
        context_vector = torch.bmm(encoder_outputs.transpose(1, 2), 
                                 attention_weights).squeeze(2)
        
        return context_vector, attention_weights


class BahdanauAttention(nn.Module):
    """Bahdanau注意力机制实现"""
    def __init__(self, hidden_size: int):
        """
        初始化Bahdanau注意力层
        
        Args:
            hidden_size: 隐藏状态维度
        """
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        """
        计算注意力权重和上下文向量
        
        Args:
            hidden: 解码器当前隐藏状态 [batch_size, hidden_size]
            encoder_outputs: 编码器所有隐藏状态 [batch_size, seq_len, hidden_size]
            
        Returns:
            context_vector: 上下文向量 [batch_size, hidden_size]
            attention_weights: 注意力权重 [batch_size, seq_len, 1]
        """
        # [batch_size, 1, hidden_size]
        hidden = hidden.unsqueeze(1)
        # [batch_size, seq_len, 1]
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(hidden)))
        attention_weights = torch.softmax(score, dim=1)
        
        # [batch_size, seq_len, hidden_size]
        context_vector = attention_weights * encoder_outputs
        # [batch_size, hidden_size]
        context_vector = context_vector.sum(1)
        
=======
import torch
import torch.nn as nn


class LuongAttention(nn.Module):
    """Luong注意力机制实现"""
    def __init__(self, hidden_size: int, method: str = "dot"):
        """
        初始化Luong注意力层
        
        Args:
            hidden_size: 隐藏状态维度
            method: 注意力计算方法，可选 "dot" 或 "general"
        """
        super(LuongAttention, self).__init__()
        self.method = method
        if method == "general":
            self.W = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        """
        计算注意力权重和上下文向量
        
        Args:
            hidden: 解码器当前隐藏状态 [batch_size, hidden_size]
            encoder_outputs: 编码器所有隐藏状态 [batch_size, seq_len, hidden_size]
            
        Returns:
            context_vector: 上下文向量 [batch_size, hidden_size]
            attention_weights: 注意力权重 [batch_size, seq_len, 1]
        """
        if self.method == "dot":
            # [batch_size, seq_len, 1]
            score = torch.bmm(encoder_outputs, hidden.unsqueeze(2))
        elif self.method == "general":
            # [batch_size, seq_len, hidden_size]
            score = self.W(encoder_outputs)
            # [batch_size, seq_len, 1]
            score = torch.bmm(score, hidden.unsqueeze(2))
        
        attention_weights = torch.softmax(score, dim=1)
        context_vector = torch.bmm(encoder_outputs.transpose(1, 2), 
                                 attention_weights).squeeze(2)
        
        return context_vector, attention_weights


class BahdanauAttention(nn.Module):
    """Bahdanau注意力机制实现"""
    def __init__(self, hidden_size: int):
        """
        初始化Bahdanau注意力层
        
        Args:
            hidden_size: 隐藏状态维度
        """
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        """
        计算注意力权重和上下文向量
        
        Args:
            hidden: 解码器当前隐藏状态 [batch_size, hidden_size]
            encoder_outputs: 编码器所有隐藏状态 [batch_size, seq_len, hidden_size]
            
        Returns:
            context_vector: 上下文向量 [batch_size, hidden_size]
            attention_weights: 注意力权重 [batch_size, seq_len, 1]
        """
        # [batch_size, 1, hidden_size]
        hidden = hidden.unsqueeze(1)
        # [batch_size, seq_len, 1]
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(hidden)))
        attention_weights = torch.softmax(score, dim=1)
        
        # [batch_size, seq_len, hidden_size]
        context_vector = attention_weights * encoder_outputs
        # [batch_size, hidden_size]
        context_vector = context_vector.sum(1)
        
>>>>>>> 122082e17e4a59b84a279fae2b942a56c0dd3b0b
        return context_vector, attention_weights 