<<<<<<< HEAD
import torch
import torch.nn as nn
import random
from typing import Optional, Tuple
from .encoder import EncoderLSTM, EncoderGRU
from .decoder import DecoderLSTM, DecoderGRU


class Seq2Seq(nn.Module):
    """序列到序列模型实现"""
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        """
        初始化Seq2Seq模型
        
        Args:
            encoder: 编码器模型
            decoder: 解码器模型
            device: 计算设备
        """
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self,
                src: torch.Tensor,
                trg: torch.Tensor,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        前向传播
        
        Args:
            src: 源语言序列 [batch_size, src_len]
            trg: 目标语言序列 [batch_size, trg_len]
            teacher_forcing_ratio: 教师强制概率
            
        Returns:
            outputs: 所有时间步的输出 [batch_size, trg_len, vocab_size]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features
        
        # 存储所有时间步的输出
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 编码器前向传播
        encoder_outputs, hidden = self.encoder(src)
        
        # 第一个解码器输入是<sos>标记
        decoder_input = trg[:, 0]
        
        for t in range(1, trg_len):
            # 解码器前向传播
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            
            # 保存当前时间步的预测
            outputs[:, t, :] = output
            
            # 决定是否使用教师强制
            teacher_force = random.random() < teacher_forcing_ratio
            
            # 获取最可能的词
            top1 = output.argmax(1)
            
            # 如果使用教师强制，则使用实际目标词作为下一个输入
            # 否则使用模型的预测作为下一个输入
            decoder_input = trg[:, t] if teacher_force else top1
            
        return outputs
    
    def translate(self,
                 src: torch.Tensor,
                 max_length: int,
                 sos_idx: int,
                 eos_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用模型进行翻译
        
        Args:
            src: 源语言序列 [batch_size, src_len]
            max_length: 生成序列的最大长度
            sos_idx: 起始符号的索引
            eos_idx: 结束符号的索引
            
        Returns:
            predictions: 预测的序列
            attentions: 注意力权重 (如果使用注意力机制)
        """
        batch_size = src.shape[0]
        attentions = []
        
        # 编码器前向传播
        encoder_outputs, hidden = self.encoder(src)
        
        # 第一个解码器输入是<sos>标记
        decoder_input = torch.tensor([sos_idx] * batch_size).to(self.device)
        
        predictions = torch.zeros(batch_size, max_length).to(self.device)
        
        for t in range(max_length):
            # 解码器前向传播
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            
            # 获取最可能的词
            top1 = output.argmax(1)
            
            # 保存预测
            predictions[:, t] = top1
            
            # 如果所有序列都预测到了<eos>，则停止生成
            if all(top1 == eos_idx):
                break
                
            # 使用当前预测作为下一个时间步的输入
            decoder_input = top1
            
=======
import torch
import torch.nn as nn
import random
from typing import Optional, Tuple
from .encoder import EncoderLSTM, EncoderGRU
from .decoder import DecoderLSTM, DecoderGRU


class Seq2Seq(nn.Module):
    """序列到序列模型实现"""
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        """
        初始化Seq2Seq模型
        
        Args:
            encoder: 编码器模型
            decoder: 解码器模型
            device: 计算设备
        """
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self,
                src: torch.Tensor,
                trg: torch.Tensor,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        前向传播
        
        Args:
            src: 源语言序列 [batch_size, src_len]
            trg: 目标语言序列 [batch_size, trg_len]
            teacher_forcing_ratio: 教师强制概率
            
        Returns:
            outputs: 所有时间步的输出 [batch_size, trg_len, vocab_size]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features
        
        # 存储所有时间步的输出
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 编码器前向传播
        encoder_outputs, hidden = self.encoder(src)
        
        # 第一个解码器输入是<sos>标记
        decoder_input = trg[:, 0]
        
        for t in range(1, trg_len):
            # 解码器前向传播
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            
            # 保存当前时间步的预测
            outputs[:, t, :] = output
            
            # 决定是否使用教师强制
            teacher_force = random.random() < teacher_forcing_ratio
            
            # 获取最可能的词
            top1 = output.argmax(1)
            
            # 如果使用教师强制，则使用实际目标词作为下一个输入
            # 否则使用模型的预测作为下一个输入
            decoder_input = trg[:, t] if teacher_force else top1
            
        return outputs
    
    def translate(self,
                 src: torch.Tensor,
                 max_length: int,
                 sos_idx: int,
                 eos_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用模型进行翻译
        
        Args:
            src: 源语言序列 [batch_size, src_len]
            max_length: 生成序列的最大长度
            sos_idx: 起始符号的索引
            eos_idx: 结束符号的索引
            
        Returns:
            predictions: 预测的序列
            attentions: 注意力权重 (如果使用注意力机制)
        """
        batch_size = src.shape[0]
        attentions = []
        
        # 编码器前向传播
        encoder_outputs, hidden = self.encoder(src)
        
        # 第一个解码器输入是<sos>标记
        decoder_input = torch.tensor([sos_idx] * batch_size).to(self.device)
        
        predictions = torch.zeros(batch_size, max_length).to(self.device)
        
        for t in range(max_length):
            # 解码器前向传播
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            
            # 获取最可能的词
            top1 = output.argmax(1)
            
            # 保存预测
            predictions[:, t] = top1
            
            # 如果所有序列都预测到了<eos>，则停止生成
            if all(top1 == eos_idx):
                break
                
            # 使用当前预测作为下一个时间步的输入
            decoder_input = top1
            
>>>>>>> 122082e17e4a59b84a279fae2b942a56c0dd3b0b
        return predictions, torch.stack(attentions).permute(1, 0, 2) if attentions else None 