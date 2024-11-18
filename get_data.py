import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler

def get_data_loaders(batch_size = 64,root='.\data',num_train =20000,valid_size = 0.2 ):
    # 数据增强和预处理
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),# 随机裁剪
    transforms.RandomHorizontalFlip(),   #随机变换
    transforms.ToTensor(),             #转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) #归一化
])
    transform_test = transforms.Compose([
    transforms.ToTensor(), #转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

    #加载数据集
    trainset = CIFAR10(root=root, train=True, download=True, transform=transform_train)
    testset = CIFAR10(root=root, train=False, download=True, transform=transform_test)

    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # 定义用于获得培训和验证批次的采样器
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # 准备数据加载器 (组合数据集和采样器 )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,sampler=train_sampler, num_workers=0)
    validloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,sampler=valid_sampler, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,num_workers=0)
    return trainloader,validloader,testloader
