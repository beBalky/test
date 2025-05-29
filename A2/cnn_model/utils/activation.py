<<<<<<< HEAD
import numpy as np

def softmax(x):
    """
    Softmax激活函数
    
    参数:
        x: 输入数据
        
    返回:
        应用softmax后的概率分布
    """
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
=======
import numpy as np

def softmax(x):
    """
    Softmax激活函数
    
    参数:
        x: 输入数据
        
    返回:
        应用softmax后的概率分布
    """
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
>>>>>>> fbfedc385406b6556c748886c8ab26c2e95c54a6
    return exps / np.sum(exps, axis=1, keepdims=True) 