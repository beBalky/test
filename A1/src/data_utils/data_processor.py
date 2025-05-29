<<<<<<< HEAD
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt


class DataProcessor:
    """
    数据处理类，用于加载、清洗、处理数据集
    """

    def __init__(self, data_path):
        """
        初始化数据处理器

        参数:
            data_path (str): 数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_scaler = None
        self.target_scaler = None

    def load_data(self):
        """加载数据"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"原始数据行数: {len(self.data)}")
            print(f"原始数据中NaN值数量: {self.data.isna().sum().sum()}")
            print("\n各列缺失值数量:")
            print(self.data.isna().sum())
            return self.data
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return None

    def clean_data(self):
        """清洗数据，处理缺失值等"""
        if self.data is None:
            print("请先加载数据")
            return None

        # 删除包含缺失值的行
        self.data = self.data.dropna()
        print(f"\n清理后的数据行数: {len(self.data)}")
        print(f"清理后的数据中NaN值数量: {self.data.isna().sum().sum()}")

        return self.data

    def split_features_target(self, target_col):
        """
        分离特征和目标变量

        参数:
            target_col (str): 目标列的名称
        """
        if self.data is None:
            print("请先加载和清洗数据")
            return None, None

        self.X = self.data.drop(columns=[target_col]).values
        self.y = self.data[target_col].values.reshape(-1, 1)

        print("\nX的统计信息:")
        print(f"形状: {self.X.shape}")
        print(
            f"最小值: {np.min(self.X)} 最大值: {np.max(self.X)} 均值: {np.mean(self.X)}")

        print("\ny的统计信息:")
        print(f"形状: {self.y.shape}")
        print(
            f"最小值: {np.min(self.y)} 最大值: {np.max(self.y)} 均值: {np.mean(self.y)}")

        return self.X, self.y

    def train_test_split(self, test_size=0.2, random_state=42):
        """
        划分训练集和测试集

        参数:
            test_size (float): 测试集比例
            random_state (int): 随机种子
        """
        if self.X is None or self.y is None:
            print("请先分离特征和目标变量")
            return None, None, None, None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        print(f"\n训练集大小: {self.X_train.shape} 测试集大小: {self.X_test.shape}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def normalize_data(self, scaler_type='standard'):
        """
        标准化/归一化数据

        参数:
            scaler_type (str): 标准化方法，可选 'standard', 'minmax', 'robust'

        返回:
            归一化后的训练集和测试集
        """
        if self.X_train is None:
            print("请先划分训练集和测试集")
            return None, None, None, None

        # 特征标准化
        if scaler_type == 'standard':
            self.feature_scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.feature_scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.feature_scaler = RobustScaler()
        else:
            print(f"不支持的标准化方法: {scaler_type}，使用StandardScaler")
            self.feature_scaler = StandardScaler()

        X_train_norm = self.feature_scaler.fit_transform(self.X_train)
        X_test_norm = self.feature_scaler.transform(self.X_test)

        # 目标变量标准化
        if scaler_type == 'standard':
            self.target_scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.target_scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.target_scaler = RobustScaler()
        else:
            self.target_scaler = StandardScaler()

        y_train_norm = self.target_scaler.fit_transform(self.y_train)
        y_test_norm = self.target_scaler.transform(self.y_test)

        return X_train_norm, X_test_norm, y_train_norm, y_test_norm

    def inverse_transform_y(self, y_norm):
        """
        将标准化的目标变量转换回原始尺度

        参数:
            y_norm (numpy.ndarray): 标准化的目标变量

        返回:
            numpy.ndarray: 原始尺度的目标变量
        """
        if self.target_scaler is None:
            print("请先标准化数据")
            return None

        return self.target_scaler.inverse_transform(y_norm)

    def create_batches(self, X, y, batch_size):
        """
        创建小批量数据

        参数:
            X (numpy.ndarray): 特征数据
            y (numpy.ndarray): 目标数据
            batch_size (int): 批量大小

        返回:
            list: 包含(X_batch, y_batch)元组的列表
        """
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        num_batches = int(np.ceil(n_samples / batch_size))
        batches = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            batches.append((X_batch, y_batch))

        return batches

    def visualize_data(self, columns=None):
        """
        可视化数据，展示特征分布和相关性

        参数:
            columns (list): 要可视化的列名，如果为None则可视化所有列
        """
        if self.data is None:
            print("请先加载数据")
            return

        if columns is None:
            columns = self.data.columns

        # 绘制直方图查看分布
        self.data[columns].hist(figsize=(15, 10), bins=20)
        plt.tight_layout()
        plt.show()

        # 相关性矩阵
        correlation = self.data[columns].corr()
        plt.figure(figsize=(12, 10))
        plt.title('特征相关性矩阵')
        plt.imshow(correlation, cmap='coolwarm', interpolation='none')
        plt.colorbar()
        plt.xticks(range(len(columns)), columns, rotation=90)
        plt.yticks(range(len(columns)), columns)
        plt.show()

        # 如果有目标变量，绘制与目标变量的关系
        if 'MEDV' in columns:
            target = 'MEDV'
            features = [col for col in columns if col != target]

            fig, axes = plt.subplots(
                nrows=len(features)//3 + 1, ncols=3, figsize=(18, len(features)*2))
            axes = axes.flatten()

            for i, feature in enumerate(features):
                axes[i].scatter(self.data[feature],
                                self.data[target], alpha=0.5)
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel(target)
                axes[i].set_title(f'{feature} vs {target}')

            plt.tight_layout()
            plt.show()
=======
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt


class DataProcessor:
    """
    数据处理类，用于加载、清洗、处理数据集
    """

    def __init__(self, data_path):
        """
        初始化数据处理器

        参数:
            data_path (str): 数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_scaler = None
        self.target_scaler = None

    def load_data(self):
        """加载数据"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"原始数据行数: {len(self.data)}")
            print(f"原始数据中NaN值数量: {self.data.isna().sum().sum()}")
            print("\n各列缺失值数量:")
            print(self.data.isna().sum())
            return self.data
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return None

    def clean_data(self):
        """清洗数据，处理缺失值等"""
        if self.data is None:
            print("请先加载数据")
            return None

        # 删除包含缺失值的行
        self.data = self.data.dropna()
        print(f"\n清理后的数据行数: {len(self.data)}")
        print(f"清理后的数据中NaN值数量: {self.data.isna().sum().sum()}")

        return self.data

    def split_features_target(self, target_col):
        """
        分离特征和目标变量

        参数:
            target_col (str): 目标列的名称
        """
        if self.data is None:
            print("请先加载和清洗数据")
            return None, None

        self.X = self.data.drop(columns=[target_col]).values
        self.y = self.data[target_col].values.reshape(-1, 1)

        print("\nX的统计信息:")
        print(f"形状: {self.X.shape}")
        print(
            f"最小值: {np.min(self.X)} 最大值: {np.max(self.X)} 均值: {np.mean(self.X)}")

        print("\ny的统计信息:")
        print(f"形状: {self.y.shape}")
        print(
            f"最小值: {np.min(self.y)} 最大值: {np.max(self.y)} 均值: {np.mean(self.y)}")

        return self.X, self.y

    def train_test_split(self, test_size=0.2, random_state=42):
        """
        划分训练集和测试集

        参数:
            test_size (float): 测试集比例
            random_state (int): 随机种子
        """
        if self.X is None or self.y is None:
            print("请先分离特征和目标变量")
            return None, None, None, None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        print(f"\n训练集大小: {self.X_train.shape} 测试集大小: {self.X_test.shape}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def normalize_data(self, scaler_type='standard'):
        """
        标准化/归一化数据

        参数:
            scaler_type (str): 标准化方法，可选 'standard', 'minmax', 'robust'

        返回:
            归一化后的训练集和测试集
        """
        if self.X_train is None:
            print("请先划分训练集和测试集")
            return None, None, None, None

        # 特征标准化
        if scaler_type == 'standard':
            self.feature_scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.feature_scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.feature_scaler = RobustScaler()
        else:
            print(f"不支持的标准化方法: {scaler_type}，使用StandardScaler")
            self.feature_scaler = StandardScaler()

        X_train_norm = self.feature_scaler.fit_transform(self.X_train)
        X_test_norm = self.feature_scaler.transform(self.X_test)

        # 目标变量标准化
        if scaler_type == 'standard':
            self.target_scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.target_scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.target_scaler = RobustScaler()
        else:
            self.target_scaler = StandardScaler()

        y_train_norm = self.target_scaler.fit_transform(self.y_train)
        y_test_norm = self.target_scaler.transform(self.y_test)

        return X_train_norm, X_test_norm, y_train_norm, y_test_norm

    def inverse_transform_y(self, y_norm):
        """
        将标准化的目标变量转换回原始尺度

        参数:
            y_norm (numpy.ndarray): 标准化的目标变量

        返回:
            numpy.ndarray: 原始尺度的目标变量
        """
        if self.target_scaler is None:
            print("请先标准化数据")
            return None

        return self.target_scaler.inverse_transform(y_norm)

    def create_batches(self, X, y, batch_size):
        """
        创建小批量数据

        参数:
            X (numpy.ndarray): 特征数据
            y (numpy.ndarray): 目标数据
            batch_size (int): 批量大小

        返回:
            list: 包含(X_batch, y_batch)元组的列表
        """
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        num_batches = int(np.ceil(n_samples / batch_size))
        batches = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            batches.append((X_batch, y_batch))

        return batches

    def visualize_data(self, columns=None):
        """
        可视化数据，展示特征分布和相关性

        参数:
            columns (list): 要可视化的列名，如果为None则可视化所有列
        """
        if self.data is None:
            print("请先加载数据")
            return

        if columns is None:
            columns = self.data.columns

        # 绘制直方图查看分布
        self.data[columns].hist(figsize=(15, 10), bins=20)
        plt.tight_layout()
        plt.show()

        # 相关性矩阵
        correlation = self.data[columns].corr()
        plt.figure(figsize=(12, 10))
        plt.title('特征相关性矩阵')
        plt.imshow(correlation, cmap='coolwarm', interpolation='none')
        plt.colorbar()
        plt.xticks(range(len(columns)), columns, rotation=90)
        plt.yticks(range(len(columns)), columns)
        plt.show()

        # 如果有目标变量，绘制与目标变量的关系
        if 'MEDV' in columns:
            target = 'MEDV'
            features = [col for col in columns if col != target]

            fig, axes = plt.subplots(
                nrows=len(features)//3 + 1, ncols=3, figsize=(18, len(features)*2))
            axes = axes.flatten()

            for i, feature in enumerate(features):
                axes[i].scatter(self.data[feature],
                                self.data[target], alpha=0.5)
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel(target)
                axes[i].set_title(f'{feature} vs {target}')

            plt.tight_layout()
            plt.show()
>>>>>>> fbfedc385406b6556c748886c8ab26c2e95c54a6
