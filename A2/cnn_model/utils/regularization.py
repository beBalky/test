<<<<<<< HEAD
import numpy as np

def l1_regularization(weights, lam=0.01):
    """
    L1正则化函数
    
    参数:
        weights: 权重矩阵
        lam: 正则化系数
        
    返回:
        L1正则化梯度
    """
    return lam * np.sign(weights)

def l2_regularization(weights, lam=0.01):
    """
    L2正则化函数
    
    参数:
        weights: 权重矩阵
        lam: 正则化系数
        
    返回:
        L2正则化梯度
    """
=======
import numpy as np

def l1_regularization(weights, lam=0.01):
    """
    L1正则化函数
    
    参数:
        weights: 权重矩阵
        lam: 正则化系数
        
    返回:
        L1正则化梯度
    """
    return lam * np.sign(weights)

def l2_regularization(weights, lam=0.01):
    """
    L2正则化函数
    
    参数:
        weights: 权重矩阵
        lam: 正则化系数
        
    返回:
        L2正则化梯度
    """
>>>>>>> fbfedc385406b6556c748886c8ab26c2e95c54a6
    return lam * weights 