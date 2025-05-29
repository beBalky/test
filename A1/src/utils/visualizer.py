<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


class Visualizer:
    """
    Data Visualization Tool Class
    """

    @staticmethod
    def plot_loss_history(losses, title="Training Loss Curve", figsize=(10, 6), save_path=None):
        """
        Plot loss history curve

        Parameters:
            losses (list): Loss history data
            title (str): Chart title
            figsize (tuple): Chart size
            save_path (str): Path to save the image
        """
        plt.figure(figsize=figsize)
        plt.plot(losses)
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)

        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_prediction_vs_actual(y_true, y_pred, title="Prediction vs True Value", figsize=(10, 6), save_path=None):
        """
        Plot prediction vs actual value comparison

        Parameters:
            y_true (numpy.ndarray): True values
            y_pred (numpy.ndarray): Predicted values
            title (str): Chart title
            figsize (tuple): Chart size
            save_path (str): Path to save the image
        """
        plt.figure(figsize=figsize)

        # 使用蓝色点表示预测值与真实值的关系，增大点的大小，并添加标签
        plt.scatter(y_true, y_pred, color='blue',
                    alpha=0.7, s=50, label='Predictions')

        # 添加理想预测线（对角线）
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--',
                 linewidth=2, label='Perfect Prediction (y=x)')

        plt.title(title, fontsize=14)
        plt.xlabel('True Value', fontsize=12)
        plt.ylabel('Predicted Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        # 添加文字说明
        plt.annotate('Blue points: Each point represents a prediction\nRed line: Perfect prediction line (y=x)',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3",
                               fc="white", ec="gray", alpha=0.8),
                     fontsize=10, ha='left', va='top')

        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_residuals(y_true, y_pred, title="Residual Plot", figsize=(10, 6), save_path=None, hist_save_path=None):
        """
        Plot residuals

        Parameters:
            y_true (numpy.ndarray): True values
            y_pred (numpy.ndarray): Predicted values
            title (str): Chart title
            figsize (tuple): Chart size
            save_path (str): Path to save the residual scatter plot
            hist_save_path (str): Path to save the residual histogram
        """
        residuals = y_true - y_pred

        plt.figure(figsize=figsize)
        plt.scatter(y_pred, residuals, color='blue',
                    alpha=0.7, s=50, label='Residuals')
        plt.axhline(y=0, color='r', linestyle='--',
                    linewidth=2, label='Zero Error Line')
        plt.title(title, fontsize=14)
        plt.xlabel('Predicted Value', fontsize=12)
        plt.ylabel('Residual (True value - Prediction)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        # 添加文字说明
        plt.annotate('Blue points: Residual errors\nRed line: Zero error line',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3",
                               fc="white", ec="gray", alpha=0.8),
                     fontsize=10, ha='left', va='top')

        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Plot residual histogram
        plt.figure(figsize=figsize)
        sns.histplot(residuals, kde=True, color='blue')
        plt.title('Residual Distribution', fontsize=14)
        plt.xlabel('Residuals', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 添加文字说明
        plt.annotate('Distribution of prediction errors',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3",
                               fc="white", ec="gray", alpha=0.8),
                     fontsize=10, ha='left', va='top')

        if hist_save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(hist_save_path), exist_ok=True)
            plt.savefig(hist_save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_feature_importance(feature_names, weights, title="Feature Importance", figsize=(12, 8), save_path=None):
        """
        Plot feature importance

        Parameters:
            feature_names (list): List of feature names
            weights (numpy.ndarray): Feature weights
            title (str): Chart title
            figsize (tuple): Chart size
            save_path (str): Path to save the image
        """
        # Convert to absolute values and sort
        importance = np.abs(weights)
        indices = np.argsort(importance)

        plt.figure(figsize=figsize)
        bars = plt.barh(range(len(indices)),
                        importance[indices], color='steelblue')
        plt.yticks(range(len(indices)), [feature_names[i]
                   for i in indices], fontsize=10)
        plt.title(title, fontsize=14)
        plt.xlabel('Absolute Weight Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 为每个条形添加数值标签
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                     f'{importance[indices[i]]:.4f}',
                     va='center', fontsize=9)

        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
=======
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


class Visualizer:
    """
    Data Visualization Tool Class
    """

    @staticmethod
    def plot_loss_history(losses, title="Training Loss Curve", figsize=(10, 6), save_path=None):
        """
        Plot loss history curve

        Parameters:
            losses (list): Loss history data
            title (str): Chart title
            figsize (tuple): Chart size
            save_path (str): Path to save the image
        """
        plt.figure(figsize=figsize)
        plt.plot(losses)
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)

        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_prediction_vs_actual(y_true, y_pred, title="Prediction vs True Value", figsize=(10, 6), save_path=None):
        """
        Plot prediction vs actual value comparison

        Parameters:
            y_true (numpy.ndarray): True values
            y_pred (numpy.ndarray): Predicted values
            title (str): Chart title
            figsize (tuple): Chart size
            save_path (str): Path to save the image
        """
        plt.figure(figsize=figsize)

        # 使用蓝色点表示预测值与真实值的关系，增大点的大小，并添加标签
        plt.scatter(y_true, y_pred, color='blue',
                    alpha=0.7, s=50, label='Predictions')

        # 添加理想预测线（对角线）
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--',
                 linewidth=2, label='Perfect Prediction (y=x)')

        plt.title(title, fontsize=14)
        plt.xlabel('True Value', fontsize=12)
        plt.ylabel('Predicted Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        # 添加文字说明
        plt.annotate('Blue points: Each point represents a prediction\nRed line: Perfect prediction line (y=x)',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3",
                               fc="white", ec="gray", alpha=0.8),
                     fontsize=10, ha='left', va='top')

        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_residuals(y_true, y_pred, title="Residual Plot", figsize=(10, 6), save_path=None, hist_save_path=None):
        """
        Plot residuals

        Parameters:
            y_true (numpy.ndarray): True values
            y_pred (numpy.ndarray): Predicted values
            title (str): Chart title
            figsize (tuple): Chart size
            save_path (str): Path to save the residual scatter plot
            hist_save_path (str): Path to save the residual histogram
        """
        residuals = y_true - y_pred

        plt.figure(figsize=figsize)
        plt.scatter(y_pred, residuals, color='blue',
                    alpha=0.7, s=50, label='Residuals')
        plt.axhline(y=0, color='r', linestyle='--',
                    linewidth=2, label='Zero Error Line')
        plt.title(title, fontsize=14)
        plt.xlabel('Predicted Value', fontsize=12)
        plt.ylabel('Residual (True value - Prediction)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        # 添加文字说明
        plt.annotate('Blue points: Residual errors\nRed line: Zero error line',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3",
                               fc="white", ec="gray", alpha=0.8),
                     fontsize=10, ha='left', va='top')

        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Plot residual histogram
        plt.figure(figsize=figsize)
        sns.histplot(residuals, kde=True, color='blue')
        plt.title('Residual Distribution', fontsize=14)
        plt.xlabel('Residuals', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 添加文字说明
        plt.annotate('Distribution of prediction errors',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3",
                               fc="white", ec="gray", alpha=0.8),
                     fontsize=10, ha='left', va='top')

        if hist_save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(hist_save_path), exist_ok=True)
            plt.savefig(hist_save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_feature_importance(feature_names, weights, title="Feature Importance", figsize=(12, 8), save_path=None):
        """
        Plot feature importance

        Parameters:
            feature_names (list): List of feature names
            weights (numpy.ndarray): Feature weights
            title (str): Chart title
            figsize (tuple): Chart size
            save_path (str): Path to save the image
        """
        # Convert to absolute values and sort
        importance = np.abs(weights)
        indices = np.argsort(importance)

        plt.figure(figsize=figsize)
        bars = plt.barh(range(len(indices)),
                        importance[indices], color='steelblue')
        plt.yticks(range(len(indices)), [feature_names[i]
                   for i in indices], fontsize=10)
        plt.title(title, fontsize=14)
        plt.xlabel('Absolute Weight Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 为每个条形添加数值标签
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                     f'{importance[indices[i]]:.4f}',
                     va='center', fontsize=9)

        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
>>>>>>> fbfedc385406b6556c748886c8ab26c2e95c54a6
