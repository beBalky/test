<<<<<<< HEAD
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import os


def load_mnist_raw(data_dir='./data'):
    """
    直接从原始文件加载MNIST数据集

    参数:
        data_dir: MNIST数据集所在的根目录路径

    返回:
        x_train: 训练图像数据
        y_train: 训练标签（one-hot编码）
        x_test: 测试图像数据
        y_test: 测试标签（one-hot编码）
    """
    # 确保MNIST数据目录存在
    mnist_dir = os.path.join(data_dir, 'MNIST', 'raw')
    os.makedirs(mnist_dir, exist_ok=True)

    # 如果文件不存在，尝试使用PyTorch下载
    train_images_path = os.path.join(mnist_dir, 'train-images-idx3-ubyte')
    train_labels_path = os.path.join(mnist_dir, 'train-labels-idx1-ubyte')
    test_images_path = os.path.join(mnist_dir, 't10k-images-idx3-ubyte')
    test_labels_path = os.path.join(mnist_dir, 't10k-labels-idx1-ubyte')

    files_exist = all(os.path.exists(f) for f in [
                      train_images_path, train_labels_path, test_images_path, test_labels_path])

    if not files_exist:
        print("MNIST原始数据文件不存在，将使用PyTorch下载...")
        # 使用PyTorch下载MNIST数据集
        _ = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True)
        _ = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True)
        print("MNIST数据集下载完成")

    def load_mnist_images(filename):
        try:
            with open(filename, 'rb') as f:
                f.read(16)  # 跳过前16个字节
                images = np.frombuffer(f.read(), dtype=np.uint8)
                images = images.reshape(-1, 28, 28, 1)  # 28x28图像
                return images
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到文件: {filename}，请确保MNIST数据集已正确下载")

    def load_mnist_labels(filename):
        try:
            with open(filename, 'rb') as f:
                f.read(8)  # 跳过前8个字节
                labels = np.frombuffer(f.read(), dtype=np.uint8)
                return labels
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到文件: {filename}，请确保MNIST数据集已正确下载")

    # 加载数据
    try:
        x_train = load_mnist_images(train_images_path)
        y_train = load_mnist_labels(train_labels_path)
        x_test = load_mnist_images(test_images_path)
        y_test = load_mnist_labels(test_labels_path)
    except Exception as e:
        print(f"加载MNIST数据时发生错误: {e}")
        print("将改用PyTorch加载MNIST数据...")
        return load_mnist_pytorch_as_numpy(data_dir)

    # 预处理数据
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 将标签转换为 one-hot 编码
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return x_train, y_train, x_test, y_test


def load_mnist_pytorch_as_numpy(data_dir='./data'):
    """
    使用PyTorch加载MNIST数据，并转换为NumPy数组格式
    
    参数:
        data_dir: 数据目录
        
    返回:
        x_train, y_train, x_test, y_test: 转换为NumPy格式的MNIST数据
    """
    # 数据预处理
    transform = transforms.Compose([transforms.ToTensor()])

    # 加载MNIST数据集
    trainset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform)

    # 转换为NumPy格式
    x_train = np.zeros((len(trainset), 28, 28, 1), dtype=np.float32)
    y_train = np.zeros(len(trainset), dtype=np.int64)

    for i, (img, label) in enumerate(trainset):
        x_train[i, :, :, 0] = img.numpy()[0]  # 从[1,28,28]转为[28,28,1]
        y_train[i] = label

    x_test = np.zeros((len(testset), 28, 28, 1), dtype=np.float32)
    y_test = np.zeros(len(testset), dtype=np.int64)

    for i, (img, label) in enumerate(testset):
        x_test[i, :, :, 0] = img.numpy()[0]
        y_test[i] = label

    # 将标签转换为 one-hot 编码
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return x_train, y_train, x_test, y_test


def load_mnist_torch(subset_size=None, data_dir='./data', batch_size=32, num_workers=2):
    """
    使用PyTorch加载MNIST数据集

    参数:
        subset_size: 如果提供，只加载指定数量的数据
        data_dir: 数据目录
        batch_size: 批次大小，较大的批次可以提高性能
        num_workers: 数据加载的并行工作线程数

    返回:
        trainloader: 训练数据的DataLoader
        testloader: 测试数据的DataLoader
    """
    # 数据预处理
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # 加载 MNIST 训练集和测试集
    trainset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform)

    # 如果指定了子集大小，则只使用部分数据
    if subset_size is not None:
        train_subset = Subset(trainset, list(
            range(min(subset_size, len(trainset)))))
        test_subset = Subset(testset, list(
            range(min(subset_size, len(testset)))))

        trainloader = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(
            test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return trainloader, testloader
=======
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import os


def load_mnist_raw(data_dir='./data'):
    """
    直接从原始文件加载MNIST数据集

    参数:
        data_dir: MNIST数据集所在的根目录路径

    返回:
        x_train: 训练图像数据
        y_train: 训练标签（one-hot编码）
        x_test: 测试图像数据
        y_test: 测试标签（one-hot编码）
    """
    # 确保MNIST数据目录存在
    mnist_dir = os.path.join(data_dir, 'MNIST', 'raw')
    os.makedirs(mnist_dir, exist_ok=True)

    # 如果文件不存在，尝试使用PyTorch下载
    train_images_path = os.path.join(mnist_dir, 'train-images-idx3-ubyte')
    train_labels_path = os.path.join(mnist_dir, 'train-labels-idx1-ubyte')
    test_images_path = os.path.join(mnist_dir, 't10k-images-idx3-ubyte')
    test_labels_path = os.path.join(mnist_dir, 't10k-labels-idx1-ubyte')

    files_exist = all(os.path.exists(f) for f in [
                      train_images_path, train_labels_path, test_images_path, test_labels_path])

    if not files_exist:
        print("MNIST原始数据文件不存在，将使用PyTorch下载...")
        # 使用PyTorch下载MNIST数据集
        _ = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True)
        _ = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True)
        print("MNIST数据集下载完成")

    def load_mnist_images(filename):
        try:
            with open(filename, 'rb') as f:
                f.read(16)  # 跳过前16个字节
                images = np.frombuffer(f.read(), dtype=np.uint8)
                images = images.reshape(-1, 28, 28, 1)  # 28x28图像
                return images
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到文件: {filename}，请确保MNIST数据集已正确下载")

    def load_mnist_labels(filename):
        try:
            with open(filename, 'rb') as f:
                f.read(8)  # 跳过前8个字节
                labels = np.frombuffer(f.read(), dtype=np.uint8)
                return labels
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到文件: {filename}，请确保MNIST数据集已正确下载")

    # 加载数据
    try:
        x_train = load_mnist_images(train_images_path)
        y_train = load_mnist_labels(train_labels_path)
        x_test = load_mnist_images(test_images_path)
        y_test = load_mnist_labels(test_labels_path)
    except Exception as e:
        print(f"加载MNIST数据时发生错误: {e}")
        print("将改用PyTorch加载MNIST数据...")
        return load_mnist_pytorch_as_numpy(data_dir)

    # 预处理数据
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 将标签转换为 one-hot 编码
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return x_train, y_train, x_test, y_test


def load_mnist_pytorch_as_numpy(data_dir='./data'):
    """
    使用PyTorch加载MNIST数据，并转换为NumPy数组格式
    
    参数:
        data_dir: 数据目录
        
    返回:
        x_train, y_train, x_test, y_test: 转换为NumPy格式的MNIST数据
    """
    # 数据预处理
    transform = transforms.Compose([transforms.ToTensor()])

    # 加载MNIST数据集
    trainset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform)

    # 转换为NumPy格式
    x_train = np.zeros((len(trainset), 28, 28, 1), dtype=np.float32)
    y_train = np.zeros(len(trainset), dtype=np.int64)

    for i, (img, label) in enumerate(trainset):
        x_train[i, :, :, 0] = img.numpy()[0]  # 从[1,28,28]转为[28,28,1]
        y_train[i] = label

    x_test = np.zeros((len(testset), 28, 28, 1), dtype=np.float32)
    y_test = np.zeros(len(testset), dtype=np.int64)

    for i, (img, label) in enumerate(testset):
        x_test[i, :, :, 0] = img.numpy()[0]
        y_test[i] = label

    # 将标签转换为 one-hot 编码
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return x_train, y_train, x_test, y_test


def load_mnist_torch(subset_size=None, data_dir='./data', batch_size=32, num_workers=2):
    """
    使用PyTorch加载MNIST数据集

    参数:
        subset_size: 如果提供，只加载指定数量的数据
        data_dir: 数据目录
        batch_size: 批次大小，较大的批次可以提高性能
        num_workers: 数据加载的并行工作线程数

    返回:
        trainloader: 训练数据的DataLoader
        testloader: 测试数据的DataLoader
    """
    # 数据预处理
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # 加载 MNIST 训练集和测试集
    trainset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform)

    # 如果指定了子集大小，则只使用部分数据
    if subset_size is not None:
        train_subset = Subset(trainset, list(
            range(min(subset_size, len(trainset)))))
        test_subset = Subset(testset, list(
            range(min(subset_size, len(testset)))))

        trainloader = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(
            test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return trainloader, testloader
>>>>>>> fbfedc385406b6556c748886c8ab26c2e95c54a6
