<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class Trainer:
    """
    基础训练器类
    """
    
    def __init__(self, model, optimizer=None):
        """
        初始化训练器
        
        参数:
            model: 要训练的模型
            optimizer: 优化器，如果为None则使用简单的梯度下降
        """
        self.model = model
        self.optimizer = optimizer
        self.train_losses = []
    
    def train(self, X_train, y_train, learning_rate=0.01, epochs=100, batch_size=32, verbose=True):
        """
        训练模型
        
        参数:
            X_train (numpy.ndarray): 训练特征
            y_train (numpy.ndarray): 训练标签
            learning_rate (float): 学习率
            epochs (int): 训练轮数
            batch_size (int): 批量大小
            verbose (bool): 是否显示训练进度
        
        返回:
            list: 训练损失历史
        """
        n_samples = X_train.shape[0]
        self.train_losses = []
        
        if verbose:
            iterator = tqdm(range(epochs), desc="Training")
        else:
            iterator = range(epochs)
        
        for epoch in iterator:
            epoch_loss = 0
            # 生成小批量数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            num_batches = int(np.ceil(n_samples / batch_size))
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # 前向传播
                predictions = self.model.forward(X_batch)
                
                # 计算损失（均方误差）
                loss = np.mean((predictions - y_batch) ** 2)
                epoch_loss += loss * (end_idx - start_idx) / n_samples
                
                # 反向传播
                self.model.backward(X_batch, y_batch, predictions)
                
                # 获取当前参数和梯度
                params = self.model.get_params()
                grads = self.model.get_gradients()
                
                # 更新参数
                if self.optimizer is not None:
                    new_params = self.optimizer.update(params, grads, learning_rate)
                else:
                    # 如果没有指定优化器，使用简单的梯度下降
                    new_params = (
                        params[0] - learning_rate * grads[0],
                        params[1] - learning_rate * grads[1],
                        params[2] - learning_rate * grads[2],
                        params[3] - learning_rate * grads[3]
                    )
                
                # 更新模型参数
                self.model.set_params(new_params)
            
            self.train_losses.append(epoch_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")
        
        return self.train_losses
    
    def evaluate(self, X, y):
        """
        评估模型
        
        参数:
            X (numpy.ndarray): 特征数据
            y (numpy.ndarray): 目标数据
        
        返回:
            float: 均方误差
        """
        predictions = self.model.forward(X)
        mse = np.mean((predictions - y) ** 2)
        return mse
    
    def predict(self, X):
        """
        使用模型进行预测
        
        参数:
            X (numpy.ndarray): 输入特征
        
        返回:
            numpy.ndarray: 预测值
        """
        return self.model.forward(X)
    
    def plot_losses(self):
        """
        绘制训练损失曲线
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
=======
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class Trainer:
    """
    基础训练器类
    """
    
    def __init__(self, model, optimizer=None):
        """
        初始化训练器
        
        参数:
            model: 要训练的模型
            optimizer: 优化器，如果为None则使用简单的梯度下降
        """
        self.model = model
        self.optimizer = optimizer
        self.train_losses = []
    
    def train(self, X_train, y_train, learning_rate=0.01, epochs=100, batch_size=32, verbose=True):
        """
        训练模型
        
        参数:
            X_train (numpy.ndarray): 训练特征
            y_train (numpy.ndarray): 训练标签
            learning_rate (float): 学习率
            epochs (int): 训练轮数
            batch_size (int): 批量大小
            verbose (bool): 是否显示训练进度
        
        返回:
            list: 训练损失历史
        """
        n_samples = X_train.shape[0]
        self.train_losses = []
        
        if verbose:
            iterator = tqdm(range(epochs), desc="Training")
        else:
            iterator = range(epochs)
        
        for epoch in iterator:
            epoch_loss = 0
            # 生成小批量数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            num_batches = int(np.ceil(n_samples / batch_size))
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # 前向传播
                predictions = self.model.forward(X_batch)
                
                # 计算损失（均方误差）
                loss = np.mean((predictions - y_batch) ** 2)
                epoch_loss += loss * (end_idx - start_idx) / n_samples
                
                # 反向传播
                self.model.backward(X_batch, y_batch, predictions)
                
                # 获取当前参数和梯度
                params = self.model.get_params()
                grads = self.model.get_gradients()
                
                # 更新参数
                if self.optimizer is not None:
                    new_params = self.optimizer.update(params, grads, learning_rate)
                else:
                    # 如果没有指定优化器，使用简单的梯度下降
                    new_params = (
                        params[0] - learning_rate * grads[0],
                        params[1] - learning_rate * grads[1],
                        params[2] - learning_rate * grads[2],
                        params[3] - learning_rate * grads[3]
                    )
                
                # 更新模型参数
                self.model.set_params(new_params)
            
            self.train_losses.append(epoch_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")
        
        return self.train_losses
    
    def evaluate(self, X, y):
        """
        评估模型
        
        参数:
            X (numpy.ndarray): 特征数据
            y (numpy.ndarray): 目标数据
        
        返回:
            float: 均方误差
        """
        predictions = self.model.forward(X)
        mse = np.mean((predictions - y) ** 2)
        return mse
    
    def predict(self, X):
        """
        使用模型进行预测
        
        参数:
            X (numpy.ndarray): 输入特征
        
        返回:
            numpy.ndarray: 预测值
        """
        return self.model.forward(X)
    
    def plot_losses(self):
        """
        绘制训练损失曲线
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
>>>>>>> fbfedc385406b6556c748886c8ab26c2e95c54a6
        plt.show() 