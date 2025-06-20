# data_utils.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from MNIST_config import config, device 

def load_mnist_data(config, device):
    """加载MNIST数据集,返回DataLoader和完整数据集张量"""
    # ... (transform_pipeline 和 dataset loading 不变) ...
    data_root = "/home/ouyangzl/BaseLine/MNIST/data"

    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  
        transforms.Lambda(lambda x: x.view(-1))    
    ])
    train_set = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform_pipeline
    )
    test_set = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform_pipeline
    )

    # 创建子集
    train_subset = Subset(train_set, indices=range(config["train_samples"]))
    test_subset = Subset(test_set, indices=range(config["test_samples"]))

    # --- 创建 DataLoader for SGD training ---
    train_loader = DataLoader(
        train_subset,
        batch_size=config["batch_size"],
        shuffle=True # SGD 通常需要 shuffle
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=config["test_samples"], # 测试通常用全批量或较大批量
        shuffle=False
    )

    # --- 加载完整数据 for analysis (Hessian, full gradient norms) ---
    full_train_loader = DataLoader(train_subset, batch_size=len(train_subset), shuffle=False)
    X_train_full, Y_train_labels_full = next(iter(full_train_loader))
    full_test_loader = DataLoader(test_subset, batch_size=len(test_subset), shuffle=False)
    X_test_full, Y_test_labels_full = next(iter(full_test_loader))


    # 转换设备并生成one-hot标签 (for full dataset used in analysis)
    Y_train_onehot_full = _to_one_hot(Y_train_labels_full, config["output_dim"], device)
    Y_test_onehot_full = _to_one_hot(Y_test_labels_full, config["output_dim"], device)

    # Move full data to device
    X_train_full = X_train_full.to(device)
    Y_train_labels_full = Y_train_labels_full.to(device)
    X_test_full = X_test_full.to(device)
    Y_test_labels_full = Y_test_labels_full.to(device)

    # 返回 Loaders 和 Full Tensors
    return (train_loader, test_loader,
            X_train_full, Y_train_labels_full, Y_train_onehot_full,
            X_test_full, Y_test_labels_full, Y_test_onehot_full)


def _to_one_hot(y, num_classes, device):
    """内部函数:生成one-hot编码"""
    y_long = y.squeeze().long()
    # Handle potential out-of-bounds indices if necessary, though unlikely for MNIST
    y_long = torch.clamp(y_long, 0, num_classes - 1)
    return torch.eye(num_classes, device=device)[y_long]