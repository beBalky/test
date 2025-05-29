import numpy as np

def mse_loss(y_true, y_pred):
    """
    均方误差损失函数
    
    参数:
        y_true: 真实标签
        y_pred: 预测值
        
    返回:
        均方误差损失值
    """
    return ((y_true - y_pred) ** 2).mean()

def mse_loss_grad(y_true, y_pred):
    """
    均方误差损失函数的梯度
    
    参数:
        y_true: 真实标签
        y_pred: 预测值
        
    返回:
        均方误差损失函数的梯度
    """
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy_loss(y_true, y_pred):
    """
    交叉熵损失函数
    
    参数:
        y_true: 真实标签（one-hot编码）
        y_pred: 预测值（概率分布）
        
    返回:
        交叉熵损失值
    """
    n_samples = y_true.shape[0]
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    logp = - np.log(y_pred[np.arange(n_samples), np.argmax(y_true, axis=1)])
    loss = np.sum(logp) / n_samples
    return loss

def cross_entropy_loss_grad(y_true, y_pred):
    """
    交叉熵损失函数的梯度
    
    参数:
        y_true: 真实标签（one-hot编码）
        y_pred: 预测值（概率分布）
        
    返回:
        交叉熵损失函数对预测值的梯度
    """
    n_samples = y_true.shape[0]
    grad = y_pred.copy()
    grad[np.arange(n_samples), np.argmax(y_true, axis=1)] -= 1
    grad = grad / n_samples
    return grad 