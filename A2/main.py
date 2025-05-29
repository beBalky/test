import numpy as np
import os
import time
import logging
import json
from datetime import datetime
from cnn_model import AlexNet
from cnn_model.data_loader import load_mnist_torch
from cnn_model.utils.loss import cross_entropy_loss, cross_entropy_loss_grad


def setup_logger(log_dir='./A2/logs'):
    """
    设置日志记录器

    参数:
        log_dir: 日志文件保存目录
    """
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件名，包含时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )

    return logging.getLogger()


def train_model(epochs=3, subset_size=500, data_dir='./data', verbose=False, num_workers=4, batch_size=64):
    """
    训练AlexNet模型

    参数:
        epochs: 训练轮数
        subset_size: 数据子集大小，用于快速训练
        data_dir: 数据集所在目录
        verbose: 是否显示详细输出
        num_workers: 数据加载的并行工作线程数
        batch_size: 批次大小，较大的批次可以提高性能
    """
    # 设置日志记录器
    logger = setup_logger()

    # 创建检查点目录
    checkpoint_dir = os.path.join(
        './A2/checkpoints', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"检查点将保存在: {checkpoint_dir}")

    # 记录训练配置
    config = {
        "epochs": epochs,
        "subset_size": subset_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "data_dir": os.path.abspath(data_dir),
        "timestamp": datetime.now().isoformat()
    }

    with open(os.path.join(checkpoint_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    logger.info(f"正在从 {os.path.abspath(data_dir)} 加载数据...")
    start_time = time.time()

    # 加载数据，使用更大的批次大小和更多工作线程
    trainloader, testloader = load_mnist_torch(
        subset_size=subset_size, data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)

    # 加载数据完成后的日志
    logger.info(f"数据加载完成，耗时: {time.time() - start_time:.2f}秒")
    logger.info(
        f"训练样本数: {len(trainloader.dataset)}, 测试样本数: {len(testloader.dataset)}")

    # 检查输入图像的形状，确保它是正确的
    for inputs, _ in trainloader:
        # 应该是[batch_size, channels, height, width]
        logger.info(f"输入图像形状: {inputs.shape}")
        break

    # 初始化模型，仅在最后一轮启用详细输出以减少日志
    model = AlexNet(learning_rate=0.001, verbose=False)
    logger.info("模型初始化完成，开始训练...")

    # 初始化最佳模型跟踪
    best_accuracy = 0.0
    best_epoch = -1
    best_model_path = None
    patience = 10  # 早停的耐心值
    no_improvement = 0  # 没有改善的轮数计数

    # 训练模型
    total_start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_loss = 0.0

        logger.info(f"轮次 {epoch + 1}/{epochs} 开始训练...")

        # 在最后一轮启用详细输出以查看模型分析
        if epoch == epochs - 1:
            model.verbose = verbose

        for i, data in enumerate(trainloader, 0):
            # 只在关键点记录进度
            if i % 10 == 0:
                logger.info(
                    f"批次进度: {i}/{len(trainloader)} ({i/len(trainloader)*100:.1f}%)")

            inputs, labels = data

            # 将数据转换为 numpy 数组，确保形状正确
            inputs = inputs.numpy()
            # 确保输入是[batch, channels, height, width]形式
            if inputs.shape[1] != 1:
                logger.warning(f"警告: 预期输入通道数为1，实际为 {inputs.shape[1]}")

            labels = np.eye(10)[labels.numpy()]  # 转换为 one-hot 编码

            # 前向传播
            y_pred = model.forward(inputs)

            # 计算损失和梯度
            loss = cross_entropy_loss(labels, y_pred)
            loss_grad = cross_entropy_loss_grad(labels, y_pred)

            # 反向传播和更新参数
            model.backward(loss_grad)

            running_loss += loss

        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"轮次 {epoch + 1}/{epochs}, 损失: {running_loss / len(trainloader):.3f}, 耗时: {epoch_time:.2f}秒")

        # 在验证集上评估模型
        current_accuracy = evaluate_subset(model, testloader, subset=100)

        # 检查是否是最佳模型
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_epoch = epoch
            no_improvement = 0  # 重置计数器

            # 保存最佳模型
            checkpoint = {
                'epoch': epoch,
                'model_state': model.save_state(),
                'accuracy': current_accuracy,
                'loss': running_loss / len(trainloader),
                'timestamp': datetime.now().isoformat()
            }

            # 删除之前的最佳模型文件
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)

            # 保存新的最佳模型
            best_model_path = os.path.join(
                checkpoint_dir, f'best_model_epoch_{epoch+1}.npz')
            np.savez(best_model_path, **checkpoint)
            logger.info(f"保存新的最佳模型，准确率: {current_accuracy*100:.2f}%")
        else:
            no_improvement += 1
            logger.info(f"模型性能没有改善，已经 {no_improvement} 轮")

            # 早停检查
            if no_improvement >= patience:
                logger.info(f"触发早停：{patience} 轮没有改善")
                break

    total_time = time.time() - total_start_time
    logger.info(
        f'训练完成，总耗时: {total_time:.2f}秒, 平均每轮耗时: {total_time/epochs:.2f}秒')
    logger.info(f'最佳模型出现在第 {best_epoch+1} 轮，准确率: {best_accuracy*100:.2f}%')

    # 打印各层的计算时间分析
    if hasattr(model, 'print_layer_times'):
        model.print_layer_times()

    # 测试模型性能
    test_model(model, testloader)

    return model, best_model_path


def evaluate_subset(model, testloader, subset=100):
    """
    在测试集的子集上快速评估模型性能

    参数:
        model: 训练好的模型
        testloader: 测试数据加载器
        subset: 要评估的样本数
    返回:
        float: 模型准确率
    """
    logger = logging.getLogger()
    correct = 0
    total = 0

    logger.info(f"在{subset}个测试样本上评估模型...")
    start_time = time.time()

    # 暂时关闭详细输出
    original_verbose = model.verbose
    model.verbose = False

    for i, data in enumerate(testloader, 0):
        if total >= subset:
            break

        inputs, labels = data

        # 将数据转换为 numpy 数组
        inputs = inputs.numpy()
        labels_np = labels.numpy()

        # 预测
        y_pred = model.forward(inputs)

        # 统计准确率
        correct += np.sum(np.argmax(y_pred, axis=1) == labels_np)
        total += labels_np.shape[0]

    # 恢复原始详细输出设置
    model.verbose = original_verbose

    accuracy = correct / total
    logger.info(
        f"快速评估 - 准确率: {accuracy * 100:.2f}%, 耗时: {time.time() - start_time:.2f}秒")

    return accuracy


def test_model(model, testloader):
    """
    测试模型性能

    参数:
        model: 训练好的模型
        testloader: 测试数据加载器
    """
    logger = logging.getLogger()
    correct_predictions = 0
    total_predictions = 0

    logger.info("开始在完整测试集上评估模型...")
    start_time = time.time()

    # 暂时关闭详细输出
    original_verbose = model.verbose
    model.verbose = False

    for i, data in enumerate(testloader, 0):
        if i % 10 == 0:
            logger.info(
                f"测试进度: {i}/{len(testloader)} ({i/len(testloader)*100:.1f}%)")

        inputs, labels = data

        # 将数据转换为 numpy 数组
        inputs = inputs.numpy()
        labels = labels.numpy()

        y_pred = model.forward(inputs)
        correct_predictions += np.sum(np.argmax(y_pred, axis=1) == labels)
        total_predictions += labels.shape[0]

    # 恢复原始详细输出设置
    model.verbose = original_verbose

    accuracy = correct_predictions / total_predictions
    logger.info(
        f"测试集上的准确率: {accuracy * 100:.2f}%, 耗时: {time.time() - start_time:.2f}秒")


def load_best_model(model_path):
    """
    加载最佳模型

    参数:
        model_path: 模型文件路径

    返回:
        加载了最佳权重的模型实例
    """
    logger = logging.getLogger()
    logger.info(f"正在加载模型: {model_path}")

    # 加载检查点
    checkpoint = np.load(model_path, allow_pickle=True)

    # 创建新的模型实例
    model = AlexNet()

    # 加载模型状态
    model.load_state(checkpoint['model_state'].item())

    logger.info(f"成功加载模型，准确率: {checkpoint['accuracy']*100:.2f}%")

    return model


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.info("开始训练AlexNet模型...")

    # 训练模型并获取最佳模型路径
    model, best_model_path = train_model(epochs=100, subset_size=1000, data_dir='./data',
                                         verbose=True, batch_size=256, num_workers=4)

    # 加载并测试最佳模型
    if best_model_path:
        best_model = load_best_model(best_model_path)
        logger.info("在完整测试集上评估最佳模型...")
        _, testloader = load_mnist_torch(
            data_dir='./data', batch_size=256, num_workers=4)
        test_model(best_model, testloader)
