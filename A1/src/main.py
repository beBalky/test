<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import datetime

from data_utils.data_processor import DataProcessor
from models.mlp import MLP
from optimizers.sgd import SGDOptimizer
from optimizers.momentum import MomentumOptimizer
from optimizers.adam import AdamOptimizer
from trainers.trainer import Trainer
from utils.metrics import Metrics
from utils.visualizer import Visualizer


class Logger:
    """
    Logger class to redirect console output to both console and a file
    """
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    """
    Main function to run the entire process
    """
    # Create output directories
    output_dir = 'A1/output'
    plots_dir = os.path.join(output_dir, 'plots')
    logs_dir = os.path.join(output_dir, 'logs')
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up logging to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f'training_log_{timestamp}.txt')
    sys.stdout = Logger(log_file)
    
    print(f"Training started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")

    # Set random seed for reproducibility
    np.random.seed(42)
    print("Random seed set to: 42")

    # Specify data path
    data_path = 'A1/data/boston.csv'
    print(f"Data path: {data_path}")

    # Data processing
    data_processor = DataProcessor(data_path)
    data = data_processor.load_data()

    # Print column names for debugging
    if data is not None:
        print("Dataset columns:", data.columns.tolist())
    else:
        print("Data loading failed")
        return

    data_processor.clean_data()
    data_processor.split_features_target(target_col='MEDV')
    data_processor.train_test_split(test_size=0.2)
    X_train, X_test, y_train, y_test = data_processor.normalize_data(
        scaler_type='standard')

    # Print shapes and statistics of features and target variables
    print(
        f"Training features shape: {X_train.shape}, Test features shape: {X_test.shape}")
    print(
        f"Training target shape: {y_train.shape}, Test target shape: {y_test.shape}")

    # Create model
    input_size = X_train.shape[1]
    hidden_size = 32
    output_size = 1
    model = MLP(input_size, hidden_size, output_size, activation='relu')
    print(f"Model architecture: Input({input_size}) -> Hidden({hidden_size}) -> Output({output_size})")
    print(f"Activation function: relu")

    # Create different optimizers
    optimizers = {
        'SGD': SGDOptimizer(),
        'Momentum': MomentumOptimizer(momentum=0.9),
        'Adam': AdamOptimizer(beta1=0.9, beta2=0.999)
    }
    print(f"Optimizers to be evaluated: {', '.join(optimizers.keys())}")

    # Store training results for each optimizer
    results = {}
    eval_results = {}
    # Store metrics for all optimizers
    optimizer_metrics = {}

    # Train the model with different optimizers
    for opt_name, optimizer in optimizers.items():
        print(f"\n{'='*50}")
        print(f"Training model with {opt_name} optimizer...")
        print(f"{'='*50}")

        # Reinitialize model to ensure fair comparison
        model = MLP(input_size, hidden_size, output_size, activation='relu')

        # Create trainer
        trainer = Trainer(model, optimizer)

        # Total epochs
        total_epochs = 1000
        # Evaluation interval
        eval_interval = 100
        
        print(f"Total epochs: {total_epochs}")
        print(f"Evaluation interval: {eval_interval}")

        losses = []
        eval_epochs = []
        eval_mses = []

        # Segment training and evaluation
        for i in range(0, total_epochs, eval_interval):
            end_epoch = min(i + eval_interval, total_epochs)
            # Train model
            batch_losses = trainer.train(
                X_train,
                y_train,
                learning_rate=0.001,
                epochs=eval_interval,
                batch_size=32,
                verbose=True
            )
            losses.extend(batch_losses)

            # Evaluate current model
            current_epoch = i + eval_interval
            test_mse = trainer.evaluate(X_test, y_test)
            print(
                f"Epoch {current_epoch}/{total_epochs}, {opt_name} Test MSE: {test_mse:.6f}")

            eval_epochs.append(current_epoch)
            eval_mses.append(test_mse)

        # Make predictions and calculate metrics for this optimizer
        y_pred = trainer.predict(X_test)
        y_test_orig = data_processor.inverse_transform_y(y_test)
        y_pred_orig = data_processor.inverse_transform_y(y_pred)

        # Store metrics for comparison table
        mae = Metrics.mae(y_test_orig, y_pred_orig)
        mse = Metrics.mse(y_test_orig, y_pred_orig)
        rmse = Metrics.rmse(y_test_orig, y_pred_orig)
        r2 = Metrics.r2(y_test_orig, y_pred_orig)

        optimizer_metrics[opt_name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }

        # Final evaluation
        final_test_mse = trainer.evaluate(X_test, y_test)
        print(f"{opt_name} Final Test MSE: {final_test_mse:.6f}")
        print(f"Performance metrics on raw scale:")
        print(f"  MAE: {mae:.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R²: {r2:.6f}")

        # Store results
        results[opt_name] = {
            'losses': losses,
            'test_mse': final_test_mse,
            'predictions': y_pred
        }

        eval_results[opt_name] = {
            'epochs': eval_epochs,
            'mses': eval_mses
        }

    # Save training history to CSV
    training_history_file = os.path.join(logs_dir, 'training_history.csv')
    history_data = []
    for opt_name, result in results.items():
        for epoch, loss in enumerate(result['losses']):
            history_data.append({
                'optimizer': opt_name,
                'epoch': epoch + 1,
                'loss': loss
            })
    
    pd.DataFrame(history_data).to_csv(training_history_file, index=False)
    print(f"\nTraining history saved to: {training_history_file}")

    # Visualize training loss curves
    plt.figure(figsize=(12, 8))
    for opt_name, result in results.items():
        plt.plot(result['losses'],
                 label=f"{opt_name} (MSE: {result['test_mse']:.6f})")

    plt.title('Training Loss Curves for Different Optimizers')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(plots_dir, 'training_loss_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training loss curves to: {save_path}")
    plt.show()

    # Visualize evaluation results every 100 epochs
    plt.figure(figsize=(12, 8))
    for opt_name, result in eval_results.items():
        plt.plot(result['epochs'], result['mses'],
                 marker='o', label=f"{opt_name}")

    plt.title('Test MSE Every 100 Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Test MSE')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(plots_dir, 'test_mse_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved test MSE comparison to: {save_path}")
    plt.show()

    # Display optimizer performance comparison table
    print("\n优化器性能对比总结:")
    print("-" * 70)
    print(f"{'优化器':<12}{'MAE':<12}{'MSE':<12}{'RMSE':<12}{'R²':<12}")
    print("-" * 70)
    for opt_name, metrics in optimizer_metrics.items():
        print(
            f"{opt_name:<12}{metrics['MAE']:<12.4f}{metrics['MSE']:<12.4f}{metrics['RMSE']:<12.4f}{metrics['R2']:<12.4f}")
    print("-" * 70)

    # Save optimizer performance comparison to CSV
    metrics_file = os.path.join(logs_dir, 'optimizer_metrics.csv')
    metrics_data = []
    for opt_name, metrics in optimizer_metrics.items():
        metrics_data.append({
            'optimizer': opt_name,
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'RMSE': metrics['RMSE'],
            'R2': metrics['R2']
        })
    
    pd.DataFrame(metrics_data).to_csv(metrics_file, index=False)
    print(f"Optimizer metrics saved to: {metrics_file}")

    # Use the best performing optimizer to retrain the model
    best_opt_name = min(results, key=lambda x: results[x]['test_mse'])
    print(f"\nBest optimizer: {best_opt_name}")

    print(f"\n{'='*50}")
    print(f"Retraining with best optimizer: {best_opt_name}")
    print(f"{'='*50}")

    # Retrain using the best optimizer
    best_model = MLP(input_size, hidden_size, output_size, activation='relu')
    best_trainer = Trainer(best_model, optimizers[best_opt_name])

    best_losses = []
    best_eval_epochs = []
    best_eval_mses = []

    # Segment training and evaluation for the best model
    for i in range(0, total_epochs, eval_interval):
        end_epoch = min(i + eval_interval, total_epochs)
        # Train model
        batch_losses = best_trainer.train(
            X_train,
            y_train,
            learning_rate=0.001,
            epochs=eval_interval,
            batch_size=32,
            verbose=True
        )
        best_losses.extend(batch_losses)

        # Evaluate current model
        current_epoch = i + eval_interval
        test_mse = best_trainer.evaluate(X_test, y_test)
        print(
            f"Best optimizer {best_opt_name} - Epoch {current_epoch}/{total_epochs}, MSE: {test_mse:.6f}")

        best_eval_epochs.append(current_epoch)
        best_eval_mses.append(test_mse)

    # Make predictions on the test set
    y_pred = best_trainer.predict(X_test)

    # Convert normalized results back to original scale
    y_test_orig = data_processor.inverse_transform_y(y_test)
    y_pred_orig = data_processor.inverse_transform_y(y_pred)

    # Calculate and print evaluation metrics
    print("\nModel Evaluation Metrics (Raw Scale):")
    Metrics.print_metrics(y_test_orig, y_pred_orig)

    # Save predictions to CSV
    predictions_file = os.path.join(logs_dir, 'best_model_predictions.csv')
    pd.DataFrame({
        'true_value': y_test_orig.flatten(),
        'predicted_value': y_pred_orig.flatten(),
        'error': (y_test_orig - y_pred_orig).flatten()
    }).to_csv(predictions_file, index=False)
    print(f"Best model predictions saved to: {predictions_file}")

    # Visualize prediction results
    Visualizer.plot_prediction_vs_actual(
        y_test_orig,
        y_pred_orig,
        title="Prediction vs True Value",
        save_path=os.path.join(plots_dir, 'prediction_vs_actual.png')
    )
    print(f"Saved prediction vs actual plot to: {os.path.join(plots_dir, 'prediction_vs_actual.png')}")

    Visualizer.plot_residuals(
        y_test_orig,
        y_pred_orig,
        title="Residual Plot",
        save_path=os.path.join(plots_dir, 'residual_plot.png'),
        hist_save_path=os.path.join(plots_dir, 'residual_distribution.png')
    )
    print(f"Saved residual plots to: {os.path.join(plots_dir, 'residual_plot.png')} and {os.path.join(plots_dir, 'residual_distribution.png')}")

    # Get feature importance
    feature_importance = np.abs(best_model.W1)
    feature_importance = np.sum(feature_importance, axis=1)

    # Feature names
    if data_processor.data is not None and hasattr(data_processor.data, 'columns'):
        feature_names = list(data_processor.data.columns[:-1])
    else:
        feature_names = [f'Feature {i+1}' for i in range(input_size)]

    # Save feature importance to CSV
    feature_importance_file = os.path.join(logs_dir, 'feature_importance.csv')
    pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).to_csv(feature_importance_file, index=False)
    print(f"Feature importance saved to: {feature_importance_file}")

    # Visualize feature importance
    Visualizer.plot_feature_importance(
        feature_names,
        feature_importance,
        title="Feature Importance",
        save_path=os.path.join(plots_dir, 'feature_importance.png')
    )
    print(f"Saved feature importance plot to: {os.path.join(plots_dir, 'feature_importance.png')}")

    # Visualize best optimizer evaluation results every 100 epochs
    plt.figure(figsize=(10, 6))
    plt.plot(best_eval_epochs, best_eval_mses,
             marker='o', label=f"{best_opt_name}")
    plt.title(f'Best Optimizer ({best_opt_name}) Test MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Test MSE')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(plots_dir, 'best_optimizer_performance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved best optimizer performance plot to: {save_path}")
    plt.show()

    print("\nTraining completed!")
    print(f"All results have been saved to: {output_dir}")
    print(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Restore original stdout
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
=======
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import datetime

from data_utils.data_processor import DataProcessor
from models.mlp import MLP
from optimizers.sgd import SGDOptimizer
from optimizers.momentum import MomentumOptimizer
from optimizers.adam import AdamOptimizer
from trainers.trainer import Trainer
from utils.metrics import Metrics
from utils.visualizer import Visualizer


class Logger:
    """
    Logger class to redirect console output to both console and a file
    """
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    """
    Main function to run the entire process
    """
    # Create output directories
    output_dir = 'A1/output'
    plots_dir = os.path.join(output_dir, 'plots')
    logs_dir = os.path.join(output_dir, 'logs')
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up logging to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f'training_log_{timestamp}.txt')
    sys.stdout = Logger(log_file)
    
    print(f"Training started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")

    # Set random seed for reproducibility
    np.random.seed(42)
    print("Random seed set to: 42")

    # Specify data path
    data_path = 'A1/data/boston.csv'
    print(f"Data path: {data_path}")

    # Data processing
    data_processor = DataProcessor(data_path)
    data = data_processor.load_data()

    # Print column names for debugging
    if data is not None:
        print("Dataset columns:", data.columns.tolist())
    else:
        print("Data loading failed")
        return

    data_processor.clean_data()
    data_processor.split_features_target(target_col='MEDV')
    data_processor.train_test_split(test_size=0.2)
    X_train, X_test, y_train, y_test = data_processor.normalize_data(
        scaler_type='standard')

    # Print shapes and statistics of features and target variables
    print(
        f"Training features shape: {X_train.shape}, Test features shape: {X_test.shape}")
    print(
        f"Training target shape: {y_train.shape}, Test target shape: {y_test.shape}")

    # Create model
    input_size = X_train.shape[1]
    hidden_size = 32
    output_size = 1
    model = MLP(input_size, hidden_size, output_size, activation='relu')
    print(f"Model architecture: Input({input_size}) -> Hidden({hidden_size}) -> Output({output_size})")
    print(f"Activation function: relu")

    # Create different optimizers
    optimizers = {
        'SGD': SGDOptimizer(),
        'Momentum': MomentumOptimizer(momentum=0.9),
        'Adam': AdamOptimizer(beta1=0.9, beta2=0.999)
    }
    print(f"Optimizers to be evaluated: {', '.join(optimizers.keys())}")

    # Store training results for each optimizer
    results = {}
    eval_results = {}
    # Store metrics for all optimizers
    optimizer_metrics = {}

    # Train the model with different optimizers
    for opt_name, optimizer in optimizers.items():
        print(f"\n{'='*50}")
        print(f"Training model with {opt_name} optimizer...")
        print(f"{'='*50}")

        # Reinitialize model to ensure fair comparison
        model = MLP(input_size, hidden_size, output_size, activation='relu')

        # Create trainer
        trainer = Trainer(model, optimizer)

        # Total epochs
        total_epochs = 1000
        # Evaluation interval
        eval_interval = 100
        
        print(f"Total epochs: {total_epochs}")
        print(f"Evaluation interval: {eval_interval}")

        losses = []
        eval_epochs = []
        eval_mses = []

        # Segment training and evaluation
        for i in range(0, total_epochs, eval_interval):
            end_epoch = min(i + eval_interval, total_epochs)
            # Train model
            batch_losses = trainer.train(
                X_train,
                y_train,
                learning_rate=0.001,
                epochs=eval_interval,
                batch_size=32,
                verbose=True
            )
            losses.extend(batch_losses)

            # Evaluate current model
            current_epoch = i + eval_interval
            test_mse = trainer.evaluate(X_test, y_test)
            print(
                f"Epoch {current_epoch}/{total_epochs}, {opt_name} Test MSE: {test_mse:.6f}")

            eval_epochs.append(current_epoch)
            eval_mses.append(test_mse)

        # Make predictions and calculate metrics for this optimizer
        y_pred = trainer.predict(X_test)
        y_test_orig = data_processor.inverse_transform_y(y_test)
        y_pred_orig = data_processor.inverse_transform_y(y_pred)

        # Store metrics for comparison table
        mae = Metrics.mae(y_test_orig, y_pred_orig)
        mse = Metrics.mse(y_test_orig, y_pred_orig)
        rmse = Metrics.rmse(y_test_orig, y_pred_orig)
        r2 = Metrics.r2(y_test_orig, y_pred_orig)

        optimizer_metrics[opt_name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }

        # Final evaluation
        final_test_mse = trainer.evaluate(X_test, y_test)
        print(f"{opt_name} Final Test MSE: {final_test_mse:.6f}")
        print(f"Performance metrics on raw scale:")
        print(f"  MAE: {mae:.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R²: {r2:.6f}")

        # Store results
        results[opt_name] = {
            'losses': losses,
            'test_mse': final_test_mse,
            'predictions': y_pred
        }

        eval_results[opt_name] = {
            'epochs': eval_epochs,
            'mses': eval_mses
        }

    # Save training history to CSV
    training_history_file = os.path.join(logs_dir, 'training_history.csv')
    history_data = []
    for opt_name, result in results.items():
        for epoch, loss in enumerate(result['losses']):
            history_data.append({
                'optimizer': opt_name,
                'epoch': epoch + 1,
                'loss': loss
            })
    
    pd.DataFrame(history_data).to_csv(training_history_file, index=False)
    print(f"\nTraining history saved to: {training_history_file}")

    # Visualize training loss curves
    plt.figure(figsize=(12, 8))
    for opt_name, result in results.items():
        plt.plot(result['losses'],
                 label=f"{opt_name} (MSE: {result['test_mse']:.6f})")

    plt.title('Training Loss Curves for Different Optimizers')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(plots_dir, 'training_loss_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training loss curves to: {save_path}")
    plt.show()

    # Visualize evaluation results every 100 epochs
    plt.figure(figsize=(12, 8))
    for opt_name, result in eval_results.items():
        plt.plot(result['epochs'], result['mses'],
                 marker='o', label=f"{opt_name}")

    plt.title('Test MSE Every 100 Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Test MSE')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(plots_dir, 'test_mse_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved test MSE comparison to: {save_path}")
    plt.show()

    # Display optimizer performance comparison table
    print("\n优化器性能对比总结:")
    print("-" * 70)
    print(f"{'优化器':<12}{'MAE':<12}{'MSE':<12}{'RMSE':<12}{'R²':<12}")
    print("-" * 70)
    for opt_name, metrics in optimizer_metrics.items():
        print(
            f"{opt_name:<12}{metrics['MAE']:<12.4f}{metrics['MSE']:<12.4f}{metrics['RMSE']:<12.4f}{metrics['R2']:<12.4f}")
    print("-" * 70)

    # Save optimizer performance comparison to CSV
    metrics_file = os.path.join(logs_dir, 'optimizer_metrics.csv')
    metrics_data = []
    for opt_name, metrics in optimizer_metrics.items():
        metrics_data.append({
            'optimizer': opt_name,
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'RMSE': metrics['RMSE'],
            'R2': metrics['R2']
        })
    
    pd.DataFrame(metrics_data).to_csv(metrics_file, index=False)
    print(f"Optimizer metrics saved to: {metrics_file}")

    # Use the best performing optimizer to retrain the model
    best_opt_name = min(results, key=lambda x: results[x]['test_mse'])
    print(f"\nBest optimizer: {best_opt_name}")

    print(f"\n{'='*50}")
    print(f"Retraining with best optimizer: {best_opt_name}")
    print(f"{'='*50}")

    # Retrain using the best optimizer
    best_model = MLP(input_size, hidden_size, output_size, activation='relu')
    best_trainer = Trainer(best_model, optimizers[best_opt_name])

    best_losses = []
    best_eval_epochs = []
    best_eval_mses = []

    # Segment training and evaluation for the best model
    for i in range(0, total_epochs, eval_interval):
        end_epoch = min(i + eval_interval, total_epochs)
        # Train model
        batch_losses = best_trainer.train(
            X_train,
            y_train,
            learning_rate=0.001,
            epochs=eval_interval,
            batch_size=32,
            verbose=True
        )
        best_losses.extend(batch_losses)

        # Evaluate current model
        current_epoch = i + eval_interval
        test_mse = best_trainer.evaluate(X_test, y_test)
        print(
            f"Best optimizer {best_opt_name} - Epoch {current_epoch}/{total_epochs}, MSE: {test_mse:.6f}")

        best_eval_epochs.append(current_epoch)
        best_eval_mses.append(test_mse)

    # Make predictions on the test set
    y_pred = best_trainer.predict(X_test)

    # Convert normalized results back to original scale
    y_test_orig = data_processor.inverse_transform_y(y_test)
    y_pred_orig = data_processor.inverse_transform_y(y_pred)

    # Calculate and print evaluation metrics
    print("\nModel Evaluation Metrics (Raw Scale):")
    Metrics.print_metrics(y_test_orig, y_pred_orig)

    # Save predictions to CSV
    predictions_file = os.path.join(logs_dir, 'best_model_predictions.csv')
    pd.DataFrame({
        'true_value': y_test_orig.flatten(),
        'predicted_value': y_pred_orig.flatten(),
        'error': (y_test_orig - y_pred_orig).flatten()
    }).to_csv(predictions_file, index=False)
    print(f"Best model predictions saved to: {predictions_file}")

    # Visualize prediction results
    Visualizer.plot_prediction_vs_actual(
        y_test_orig,
        y_pred_orig,
        title="Prediction vs True Value",
        save_path=os.path.join(plots_dir, 'prediction_vs_actual.png')
    )
    print(f"Saved prediction vs actual plot to: {os.path.join(plots_dir, 'prediction_vs_actual.png')}")

    Visualizer.plot_residuals(
        y_test_orig,
        y_pred_orig,
        title="Residual Plot",
        save_path=os.path.join(plots_dir, 'residual_plot.png'),
        hist_save_path=os.path.join(plots_dir, 'residual_distribution.png')
    )
    print(f"Saved residual plots to: {os.path.join(plots_dir, 'residual_plot.png')} and {os.path.join(plots_dir, 'residual_distribution.png')}")

    # Get feature importance
    feature_importance = np.abs(best_model.W1)
    feature_importance = np.sum(feature_importance, axis=1)

    # Feature names
    if data_processor.data is not None and hasattr(data_processor.data, 'columns'):
        feature_names = list(data_processor.data.columns[:-1])
    else:
        feature_names = [f'Feature {i+1}' for i in range(input_size)]

    # Save feature importance to CSV
    feature_importance_file = os.path.join(logs_dir, 'feature_importance.csv')
    pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).to_csv(feature_importance_file, index=False)
    print(f"Feature importance saved to: {feature_importance_file}")

    # Visualize feature importance
    Visualizer.plot_feature_importance(
        feature_names,
        feature_importance,
        title="Feature Importance",
        save_path=os.path.join(plots_dir, 'feature_importance.png')
    )
    print(f"Saved feature importance plot to: {os.path.join(plots_dir, 'feature_importance.png')}")

    # Visualize best optimizer evaluation results every 100 epochs
    plt.figure(figsize=(10, 6))
    plt.plot(best_eval_epochs, best_eval_mses,
             marker='o', label=f"{best_opt_name}")
    plt.title(f'Best Optimizer ({best_opt_name}) Test MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Test MSE')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(plots_dir, 'best_optimizer_performance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved best optimizer performance plot to: {save_path}")
    plt.show()

    print("\nTraining completed!")
    print(f"All results have been saved to: {output_dir}")
    print(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Restore original stdout
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
>>>>>>> fbfedc385406b6556c748886c8ab26c2e95c54a6
