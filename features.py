import torch
import numpy as np
from cnn_model import Net
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
model_path ='E:\python\study python\CNN-RF\model\model_1.pth'
model = Net()
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    model.cuda()

valid_size = 0.2
# 获取将用于验证的培训指标
# num_train = len(trainset)
num_train =20000#控制样本数量
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
# 数据预处理和加载
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),# 随机裁剪
    transforms.RandomHorizontalFlip(),#随机变换
    transforms.ToTensor(),#转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_sampler = SubsetRandomSampler(train_idx)
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
trainloader = DataLoader(trainset, batch_size=64,sampler=train_sampler, num_workers=0)
testloader = DataLoader(testset, batch_size=64, num_workers=0)


class Parameter():
    def __init__(self,model_path,model,train_on_gpu,testloader):
        self.model_path = model_path
        self.model = model
        self.train_on_gpu = train_on_gpu
        self.testloader = testloader

class Getfeatures(Parameter):
    def __init__(self, model_path, model, train_on_gpu,testloader):
        super().__init__(model_path, model,train_on_gpu,testloader)
        self.features =[]
        self.X_train_features = []
        self.x_train = []
        self.Y_test_features = []
        self.y_test = []
        self.features = []
    
    def get_features_hook(self,module, input, output):
        self.features.append(output.cpu().detach().numpy())  # 将特征保存在`features`列表中

    def extract_features(self):
        conv3_layer = model.conv3
        hook = conv3_layer.register_forward_hook(self.get_features_hook)
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in trainloader:
                if train_on_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                
                # 前向传播（这时钩子会被触发）
                outputs = model(inputs)
                
                # 保存特征（钩子自动调用）
                label_array = labels.cpu().numpy()
                
                # 将特征和标签保存到列表中
                features_array = self.features[-1]  # 取最后一批数据的输出特征
                self.X_train_features.append(features_array)
                self.x_train.append(label_array)

        X_train_features = np.concatenate(self.X_train_features, axis=0)
        x_train = np.concatenate(self.x_train, axis=0)


        with torch.no_grad():
            for inputs, labels in testloader:
                if train_on_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                
                # 前向传播（这时钩子会被触发）
                outputs = model(inputs)
                
                # 保存特征（钩子自动调用）
                label_array = labels.cpu().numpy()
                
                # 将特征和标签保存到列表中
                features_array = self.features[-1]  # 取最后一批数据的输出特征
                self.Y_test_features.append(features_array)
                self.y_test.append(label_array)

        Y_test_features = np.concatenate(self.Y_test_features, axis=0)
        y_test = np.concatenate(self.y_test, axis=0)

        hook.remove()
        # print("Train features:", X_train_features)
        # print("Train labels:", x_train)
        # print("Test features:", Y_test_features)
        # print("Test labels:", y_test)
        return X_train_features,x_train,Y_test_features,y_test
    


features = Getfeatures(model_path, model, train_on_gpu, testloader)
X_train_features, x_train,Y_test_features,y_test = features.extract_features()
