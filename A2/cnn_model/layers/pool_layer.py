import numpy as np


class PoolLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}
    
    def forward(self, x):
        """
        向量化的最大池化前向传播
        """
        self.last_input = x
        batch_size, channels, height, width = x.shape
        
        # 计算输出尺寸
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        # 初始化输出
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        # 存储用于反向传播的索引
        self.max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=np.int32)
        
        # 批量计算最大池化
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    h_start = h * self.stride
                    h_end = min(h_start + self.pool_size, height)  # 确保不超出边界
                    
                    for w in range(out_width):
                        w_start = w * self.stride
                        w_end = min(w_start + self.pool_size, width)  # 确保不超出边界
                        
                        # 获取当前窗口
                        pool_region = x[b, c, h_start:h_end, w_start:w_end]
                        
                        # 确保窗口非空
                        if pool_region.size > 0:
                            # 找到最大值及其位置
                            output[b, c, h, w] = np.max(pool_region)
                            
                            # 存储最大值的位置，用于反向传播
                            max_idx = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                            self.max_indices[b, c, h, w] = max_idx
        
        return output

    def backward(self, d_out):
        """
        向量化的最大池化反向传播
        """
        batch_size, channels, out_height, out_width = d_out.shape
        _, _, in_height, in_width = self.last_input.shape
        
        # 初始化输入梯度
        d_input = np.zeros_like(self.last_input)
        
        # 分配梯度到正确的位置
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        # 获取最大值位置
                        h_max, w_max = self.max_indices[b, c, h, w]
                        
                        # 计算原始数据中的位置
                        h_orig = h * self.stride + h_max
                        w_orig = w * self.stride + w_max
                        
                        # 确保索引在有效范围内
                        if h_orig < in_height and w_orig < in_width:
                            # 将梯度分配给最大值位置
                            d_input[b, c, h_orig, w_orig] += d_out[b, c, h, w]
        
        return d_input 