import numpy as np


class ConvLayer:
    def __init__(self, input_channels, num_filters, kernel_size, stride=1, padding=0, learning_rate=0.01):
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # 使用He初始化以改善训练
        scale = np.sqrt(2.0 / (input_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(num_filters, input_channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros((num_filters, 1))  # 初始化为0而非随机值
        self.learning_rate = learning_rate
        
    def im2col(self, x, h_out, w_out):
        """
        将输入图像转换为列矩阵，用于高效卷积
        """
        batch_size, channels, height, width = x.shape
        k = self.kernel_size
        
        # 初始化输出矩阵 - 形状为 (batch_size, channels*k*k, h_out*w_out)
        cols = np.zeros((batch_size, channels * k * k, h_out * w_out))
        
        # 填充结果矩阵
        for b in range(batch_size):
            col_idx = 0
            for i in range(h_out):
                for j in range(w_out):
                    y_start = i * self.stride
                    x_start = j * self.stride
                    
                    # 确保提取的窗口在有效范围内
                    if y_start + k <= height + 2 * self.padding and x_start + k <= width + 2 * self.padding:
                        # 提取窗口
                        window = x[b, :, y_start:y_start+k, x_start:x_start+k]
                        
                        # 只有当窗口形状正确时才添加
                        if window.shape == (channels, k, k):
                            cols[b, :, col_idx] = window.reshape(-1)
                            col_idx += 1
                    
        return cols
        
    def forward(self, x):
        """
        向量化的前向传播实现
        """
        self.last_input = x
        batch_size, channels, height, width = x.shape
        
        # 添加padding到输入
        if self.padding > 0:
            padded_x = np.zeros((batch_size, channels, height + 2 * self.padding, width + 2 * self.padding))
            padded_x[:, :, self.padding:self.padding+height, self.padding:self.padding+width] = x
        else:
            padded_x = x
            
        padded_height, padded_width = padded_x.shape[2], padded_x.shape[3]
        
        # 计算输出尺寸
        h_out = (padded_height - self.kernel_size) // self.stride + 1
        w_out = (padded_width - self.kernel_size) // self.stride + 1
        
        # 保存这些值以便反向传播
        self.h_out, self.w_out = h_out, w_out
        self.padded_x = padded_x
        
        # 将权重重塑为 (num_filters, input_channels*kernel_size*kernel_size)
        w_reshaped = self.weights.reshape(self.num_filters, -1)
        
        # 将输入转换为列矩阵
        x_cols = self.im2col(padded_x, h_out, w_out)
        self.x_cols = x_cols  # 保存以便反向传播
        
        # 批量矩阵乘法计算卷积
        output = np.zeros((batch_size, self.num_filters, h_out, w_out))
        for b in range(batch_size):
            # 矩阵乘法：(num_filters, channels*k*k) x (channels*k*k, h_out*w_out)
            out = np.dot(w_reshaped, x_cols[b]) + self.bias
            # 重塑结果
            output[b] = out.reshape(self.num_filters, h_out, w_out)
            
        return output
    
    def col2im(self, dcol, x_shape):
        """
        列矩阵转回图像格式，用于反向传播
        """
        batch_size, channels, height, width = x_shape
        padded_h = height + 2 * self.padding
        padded_w = width + 2 * self.padding
        
        dx_padded = np.zeros((batch_size, channels, padded_h, padded_w))
        k = self.kernel_size
        
        # 遍历每个batch
        for b in range(batch_size):
            # 确保形状正确
            h_out = self.h_out
            w_out = self.w_out
            
            # 只使用有效的列（防止索引越界）
            valid_cols = min(dcol.shape[2], h_out * w_out)
            
            # 重构梯度
            for col_idx in range(valid_cols):
                # 计算在特征图中的位置
                i = col_idx // w_out
                j = col_idx % w_out
                
                y_start = i * self.stride
                x_start = j * self.stride
                
                # 确保在有效范围内
                if y_start + k <= padded_h and x_start + k <= padded_w:
                    # 获取当前窗口的梯度并正确重塑
                    window_grad = dcol[b, :, col_idx].reshape(channels, k, k)
                    
                    # 累加到对应的位置
                    dx_padded[b, :, y_start:y_start+k, x_start:x_start+k] += window_grad
                    
        # 移除padding
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded
            
        return dx
    
    def backward(self, d_out):
        """
        向量化的反向传播实现
        """
        batch_size, num_filters, out_height, out_width = d_out.shape
        
        # 确保 self.h_out 和 self.w_out 与d_out形状匹配
        if out_height != self.h_out or out_width != self.w_out:
            print(f"警告: 输出梯度形状({out_height},{out_width})与前向传播保存的形状({self.h_out},{self.w_out})不匹配")
            self.h_out, self.w_out = out_height, out_width
        
        # 将输出梯度重塑为二维矩阵 (num_filters, h_out*w_out)
        d_out_reshaped = d_out.reshape(batch_size, num_filters, -1)
        
        # 计算偏置梯度
        d_bias = np.sum(d_out_reshaped, axis=(0, 2)).reshape(num_filters, 1)
        
        # 初始化权重梯度
        d_weights = np.zeros_like(self.weights)
        
        # 将权重重塑为二维矩阵
        w_reshaped = self.weights.reshape(num_filters, -1)  # (num_filters, input_channels*k*k)
        
        # 初始化输入梯度
        d_x_cols = np.zeros_like(self.x_cols)
        
        for b in range(batch_size):
            # 对每个过滤器计算权重梯度
            for f in range(num_filters):
                for i in range(self.h_out * self.w_out):
                    # 确保索引在有效范围内
                    if i < d_out_reshaped.shape[2] and i < self.x_cols.shape[2]:
                        # 计算权重梯度贡献
                        window_grad = d_out_reshaped[b, f, i] * self.x_cols[b, :, i]
                        d_weights[f] += window_grad.reshape(self.input_channels, self.kernel_size, self.kernel_size)
            
            # 计算输入梯度
            for i in range(self.h_out * self.w_out):
                # 确保索引在有效范围内
                if i < d_out_reshaped.shape[2] and i < d_x_cols.shape[2]:
                    for f in range(num_filters):
                        d_x_cols[b, :, i] += d_out_reshaped[b, f, i] * w_reshaped[f]
        
        # 将列矩阵转回图像格式
        d_input = self.col2im(d_x_cols, self.last_input.shape)
        
        # 更新权重和偏置
        self.weights -= self.learning_rate * d_weights
        self.bias -= self.learning_rate * d_bias
        
        return d_input, d_weights, d_bias 