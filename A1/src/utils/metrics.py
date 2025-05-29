<<<<<<< HEAD
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Metrics:
    """
    评估指标计算工具类
    """

    @staticmethod
    def mse(y_true, y_pred):
        """
        计算均方误差 (Mean Squared Error)

        参数:
            y_true (numpy.ndarray): 真实值
            y_pred (numpy.ndarray): 预测值

        返回:
            float: MSE值
        """
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def rmse(y_true, y_pred):
        """
        计算均方根误差 (Root Mean Squared Error)

        参数:
            y_true (numpy.ndarray): 真实值
            y_pred (numpy.ndarray): 预测值

        返回:
            float: RMSE值
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(y_true, y_pred):
        """
        计算平均绝对误差 (Mean Absolute Error)

        参数:
            y_true (numpy.ndarray): 真实值
            y_pred (numpy.ndarray): 预测值

        返回:
            float: MAE值
        """
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def r2(y_true, y_pred):
        """
        计算R²决定系数

        参数:
            y_true (numpy.ndarray): 真实值
            y_pred (numpy.ndarray): 预测值

        返回:
            float: R²值
        """
        return r2_score(y_true, y_pred)

    @staticmethod
    def print_metrics(y_true, y_pred):
        """
        打印所有评估指标

        参数:
            y_true (numpy.ndarray): 真实值
            y_pred (numpy.ndarray): 预测值
        """
        print(f"MSE: {Metrics.mse(y_true, y_pred):.6f}")
        print(f"RMSE: {Metrics.rmse(y_true, y_pred):.6f}")
        print(f"MAE: {Metrics.mae(y_true, y_pred):.6f}")
        print(f"R²: {Metrics.r2(y_true, y_pred):.6f}")
=======
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Metrics:
    """
    评估指标计算工具类
    """

    @staticmethod
    def mse(y_true, y_pred):
        """
        计算均方误差 (Mean Squared Error)

        参数:
            y_true (numpy.ndarray): 真实值
            y_pred (numpy.ndarray): 预测值

        返回:
            float: MSE值
        """
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def rmse(y_true, y_pred):
        """
        计算均方根误差 (Root Mean Squared Error)

        参数:
            y_true (numpy.ndarray): 真实值
            y_pred (numpy.ndarray): 预测值

        返回:
            float: RMSE值
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(y_true, y_pred):
        """
        计算平均绝对误差 (Mean Absolute Error)

        参数:
            y_true (numpy.ndarray): 真实值
            y_pred (numpy.ndarray): 预测值

        返回:
            float: MAE值
        """
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def r2(y_true, y_pred):
        """
        计算R²决定系数

        参数:
            y_true (numpy.ndarray): 真实值
            y_pred (numpy.ndarray): 预测值

        返回:
            float: R²值
        """
        return r2_score(y_true, y_pred)

    @staticmethod
    def print_metrics(y_true, y_pred):
        """
        打印所有评估指标

        参数:
            y_true (numpy.ndarray): 真实值
            y_pred (numpy.ndarray): 预测值
        """
        print(f"MSE: {Metrics.mse(y_true, y_pred):.6f}")
        print(f"RMSE: {Metrics.rmse(y_true, y_pred):.6f}")
        print(f"MAE: {Metrics.mae(y_true, y_pred):.6f}")
        print(f"R²: {Metrics.r2(y_true, y_pred):.6f}")
>>>>>>> fbfedc385406b6556c748886c8ab26c2e95c54a6
