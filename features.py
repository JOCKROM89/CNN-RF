import torch
import numpy as np
from cnn_model import Net
from get_data import get_data_loaders
import torch.nn as nn

# 模型路径及数据加载
model_path = 'E:\python\study python\CNN-RF\model\model_1.pth'#保存的CNN模型
model = Net()
train_on_gpu = torch.cuda.is_available()  # 判断是否有可用的GPU
if train_on_gpu:
    model.cuda()  # 将模型移动到GPU

# 获取训练、验证和测试数据加载器
trainloader, validloader, testloader, testset = get_data_loaders()

class Parameter():
    def __init__(self, model_path, model, train_on_gpu, trainloader, testloader):
        # 初始化类，传入模型路径、模型、是否使用GPU、训练集和测试集数据加载器
        self.model_path = model_path
        self.model = model
        self.train_on_gpu = train_on_gpu
        self.trainloader = trainloader
        self.testloader = testloader

class GetFeatures(Parameter):
    def __init__(self, model_path, model, train_on_gpu, trainloader, testloader):
        # 初始化继承父类Parameter
        super().__init__(model_path, model, train_on_gpu, trainloader, testloader)
        self.model.eval()  # 设置模型为评估模式，禁用dropout等训练时特有的行为
        self.features = []  # 存储特征
        self.X_train_features = []  # 存储训练数据的特征
        self.x_train = []  # 存储训练数据的标签
        self.Y_test_features = []  # 存储测试数据的特征
        self.y_test = []  # 存储测试数据的标签

     def _extract_data_features(self, dataloader, feature_list, label_list):
        # 遍历数据加载器提取特征
        with torch.no_grad():  # 在不计算梯度的情况下执行推理
            for inputs, labels in dataloader:
                if self.train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()  # 将数据移动到GPU

                # 通过模型进行前向传递，得到输出特征
                outputs = self.model(inputs)
                features = outputs.cpu().numpy()  # 从模型输出中获取特征，并转换为numpy数组
                labels = labels.cpu().numpy()    # 获取标签并转换为numpy数组

                # 将特征和标签添加到对应的列表中
                feature_list.append(features)
                label_list.append(labels)
                
    def extract_features(self):
        # 替换全连接层（fc3）为Identity层，仅提取特征
        fc_layer = self.model.fc3
        self.model.fc3 = nn.Identity()
        # 提取训练数据的特征
        self._extract_data_features(self.trainloader, self.X_train_features, self.x_train)
        # 将训练集的特征和标签合并为numpy数组
        X_train_features = np.concatenate(self.X_train_features, axis=0)
        x_train = np.concatenate(self.x_train, axis=0)
        # 提取测试数据的特征
        self._extract_data_features(self.testloader, self.Y_test_features, self.y_test)
        # 将测试集的特征和标签合并为numpy数组
        Y_test_features = np.concatenate(self.Y_test_features, axis=0)
        y_test = np.concatenate(self.y_test, axis=0)

        return X_train_features, x_train, Y_test_features, y_test

   
