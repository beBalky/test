<<<<<<< HEAD
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import spacy
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import os


class TranslationDataset:
    """机器翻译数据集处理类"""
    def __init__(self,
                 min_freq: int = 2,
                 batch_size: int = 128,
                 device: torch.device = torch.device('cpu')):
        """
        初始化数据集
        
        Args:
            min_freq: 最小词频
            batch_size: 批次大小
            device: 计算设备
        """
        self.min_freq = min_freq
        self.batch_size = batch_size
        self.device = device
        
        # 加载分词器
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            print("请先安装spacy语言模型：")
            print("python -m spacy download de_core_news_sm")
            print("python -m spacy download en_core_web_sm")
            raise
            
        # 创建分词函数
        self.tokenize_de = lambda text: [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]
        self.tokenize_en = lambda text: [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]
        
        # 特殊标记
        self.special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']
        
        # 加载数据集
        self.train_data = self._load_data('train')
        self.valid_data = self._load_data('val')
        self.test_data = self._load_data('test')
        
        # 构建词汇表
        self.src_vocab = self._build_vocab(self.train_data, self.tokenize_de, 'de')
        self.trg_vocab = self._build_vocab(self.train_data, self.tokenize_en, 'en')
        
    def _load_data(self, split: str) -> List[Tuple[str, str]]:
        """加载数据集"""
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        de_path = os.path.join(data_dir, f'{split}.de')
        en_path = os.path.join(data_dir, f'{split}.en')
        
        if not os.path.exists(de_path) or not os.path.exists(en_path):
            raise FileNotFoundError(
                f"数据文件不存在。请确保在 {data_dir} 目录下有以下文件：\n"
                f"- {split}.de\n"
                f"- {split}.en"
            )
        
        with open(de_path, 'r', encoding='utf-8') as f_de, \
             open(en_path, 'r', encoding='utf-8') as f_en:
            de_lines = f_de.readlines()
            en_lines = f_en.readlines()
            
        return list(zip(de_lines, en_lines))
    
    def _build_vocab(self,
                    data: List[Tuple[str, str]],
                    tokenize_fn,
                    lang: str) -> Dict[str, int]:
        """
        构建词汇表
        
        Args:
            data: 数据集
            tokenize_fn: 分词函数
            lang: 语言标识
            
        Returns:
            vocab: 词汇表
        """
        def yield_tokens(data_iter, tokenize_fn, lang_idx):
            for item in data_iter:
                yield tokenize_fn(item[lang_idx])
                
        lang_idx = 0 if lang == 'de' else 1
        vocab = build_vocab_from_iterator(
            yield_tokens(data, tokenize_fn, lang_idx),
            min_freq=self.min_freq,
            specials=self.special_tokens
        )
        vocab.set_default_index(vocab['<unk>'])
        return vocab
    
    def _process_batch(self,
                      batch: List[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理批次数据
        
        Args:
            batch: 批次数据
            
        Returns:
            src_tensor: 源语言张量
            trg_tensor: 目标语言张量
        """
        de_batch, en_batch = [], []
        for de_sent, en_sent in batch:
            # 分词
            de_tokens = ['<sos>'] + self.tokenize_de(de_sent.strip()) + ['<eos>']
            en_tokens = ['<sos>'] + self.tokenize_en(en_sent.strip()) + ['<eos>']
            
            # 转换为索引
            de_indices = [self.src_vocab[token] for token in de_tokens]
            en_indices = [self.trg_vocab[token] for token in en_tokens]
            
            de_batch.append(de_indices)
            en_batch.append(en_indices)
            
        # 填充到相同长度
        src_tensor = self._pad_sequence(de_batch)
        trg_tensor = self._pad_sequence(en_batch)
        
        return src_tensor.to(self.device), trg_tensor.to(self.device)
    
    def _pad_sequence(self, sequences: List[List[int]]) -> torch.Tensor:
        """
        将序列填充到相同长度
        
        Args:
            sequences: 序列列表
            
        Returns:
            padded_sequences: 填充后的序列张量
        """
        max_len = max(len(seq) for seq in sequences)
        pad_idx = self.src_vocab['<pad>']
        
        padded_sequences = []
        for seq in sequences:
            padded_seq = seq + [pad_idx] * (max_len - len(seq))
            padded_sequences.append(padded_seq)
            
        return torch.tensor(padded_sequences)
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        获取数据加载器
        
        Returns:
            train_loader: 训练集数据加载器
            valid_loader: 验证集数据加载器
            test_loader: 测试集数据加载器
        """
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._process_batch
        )
        
        valid_loader = DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._process_batch
        )
        
        test_loader = DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._process_batch
        )
        
        return train_loader, valid_loader, test_loader
    
    def get_vocab_sizes(self) -> Tuple[int, int]:
        """
        获取词汇表大小
        
        Returns:
            src_vocab_size: 源语言词汇表大小
            trg_vocab_size: 目标语言词汇表大小
        """
=======
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import spacy
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import os


class TranslationDataset:
    """机器翻译数据集处理类"""
    def __init__(self,
                 min_freq: int = 2,
                 batch_size: int = 128,
                 device: torch.device = torch.device('cpu')):
        """
        初始化数据集
        
        Args:
            min_freq: 最小词频
            batch_size: 批次大小
            device: 计算设备
        """
        self.min_freq = min_freq
        self.batch_size = batch_size
        self.device = device
        
        # 加载分词器
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            print("请先安装spacy语言模型：")
            print("python -m spacy download de_core_news_sm")
            print("python -m spacy download en_core_web_sm")
            raise
            
        # 创建分词函数
        self.tokenize_de = lambda text: [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]
        self.tokenize_en = lambda text: [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]
        
        # 特殊标记
        self.special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']
        
        # 加载数据集
        self.train_data = self._load_data('train')
        self.valid_data = self._load_data('val')
        self.test_data = self._load_data('test')
        
        # 构建词汇表
        self.src_vocab = self._build_vocab(self.train_data, self.tokenize_de, 'de')
        self.trg_vocab = self._build_vocab(self.train_data, self.tokenize_en, 'en')
        
    def _load_data(self, split: str) -> List[Tuple[str, str]]:
        """加载数据集"""
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        de_path = os.path.join(data_dir, f'{split}.de')
        en_path = os.path.join(data_dir, f'{split}.en')
        
        if not os.path.exists(de_path) or not os.path.exists(en_path):
            raise FileNotFoundError(
                f"数据文件不存在。请确保在 {data_dir} 目录下有以下文件：\n"
                f"- {split}.de\n"
                f"- {split}.en"
            )
        
        with open(de_path, 'r', encoding='utf-8') as f_de, \
             open(en_path, 'r', encoding='utf-8') as f_en:
            de_lines = f_de.readlines()
            en_lines = f_en.readlines()
            
        return list(zip(de_lines, en_lines))
    
    def _build_vocab(self,
                    data: List[Tuple[str, str]],
                    tokenize_fn,
                    lang: str) -> Dict[str, int]:
        """
        构建词汇表
        
        Args:
            data: 数据集
            tokenize_fn: 分词函数
            lang: 语言标识
            
        Returns:
            vocab: 词汇表
        """
        def yield_tokens(data_iter, tokenize_fn, lang_idx):
            for item in data_iter:
                yield tokenize_fn(item[lang_idx])
                
        lang_idx = 0 if lang == 'de' else 1
        vocab = build_vocab_from_iterator(
            yield_tokens(data, tokenize_fn, lang_idx),
            min_freq=self.min_freq,
            specials=self.special_tokens
        )
        vocab.set_default_index(vocab['<unk>'])
        return vocab
    
    def _process_batch(self,
                      batch: List[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理批次数据
        
        Args:
            batch: 批次数据
            
        Returns:
            src_tensor: 源语言张量
            trg_tensor: 目标语言张量
        """
        de_batch, en_batch = [], []
        for de_sent, en_sent in batch:
            # 分词
            de_tokens = ['<sos>'] + self.tokenize_de(de_sent.strip()) + ['<eos>']
            en_tokens = ['<sos>'] + self.tokenize_en(en_sent.strip()) + ['<eos>']
            
            # 转换为索引
            de_indices = [self.src_vocab[token] for token in de_tokens]
            en_indices = [self.trg_vocab[token] for token in en_tokens]
            
            de_batch.append(de_indices)
            en_batch.append(en_indices)
            
        # 填充到相同长度
        src_tensor = self._pad_sequence(de_batch)
        trg_tensor = self._pad_sequence(en_batch)
        
        return src_tensor.to(self.device), trg_tensor.to(self.device)
    
    def _pad_sequence(self, sequences: List[List[int]]) -> torch.Tensor:
        """
        将序列填充到相同长度
        
        Args:
            sequences: 序列列表
            
        Returns:
            padded_sequences: 填充后的序列张量
        """
        max_len = max(len(seq) for seq in sequences)
        pad_idx = self.src_vocab['<pad>']
        
        padded_sequences = []
        for seq in sequences:
            padded_seq = seq + [pad_idx] * (max_len - len(seq))
            padded_sequences.append(padded_seq)
            
        return torch.tensor(padded_sequences)
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        获取数据加载器
        
        Returns:
            train_loader: 训练集数据加载器
            valid_loader: 验证集数据加载器
            test_loader: 测试集数据加载器
        """
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._process_batch
        )
        
        valid_loader = DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._process_batch
        )
        
        test_loader = DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._process_batch
        )
        
        return train_loader, valid_loader, test_loader
    
    def get_vocab_sizes(self) -> Tuple[int, int]:
        """
        获取词汇表大小
        
        Returns:
            src_vocab_size: 源语言词汇表大小
            trg_vocab_size: 目标语言词汇表大小
        """
>>>>>>> 122082e17e4a59b84a279fae2b942a56c0dd3b0b
        return len(self.src_vocab), len(self.trg_vocab) 