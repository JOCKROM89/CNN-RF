import torch
import numpy as np
from cnn_model import Net
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

model_path ='E:\python\study python\CNN-RF\model\model_1.pth'
model = Net()
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    model.cuda()
# 钩子列表来保存特征
features = []

# 数据预处理和加载
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
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
        self.features = []
    
    def get_features_hook(self,module, input, output):    #使用了钩子来保存卷积层（conv3）的输出特征
        self.features.append(output.cpu().detach().numpy())  # 将特征保存在`features`列表中

    def extract_features(self):
        conv3_layer = model.conv3
        hook = conv3_layer.register_forward_hook(self.get_features_hook)
        self.model.eval()
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
                self.X_train_features.append(features_array)
                self.x_train.append(label_array)

        X_train_features = np.concatenate(self.X_train_features, axis=0)
        x_train = np.concatenate(self.x_train, axis=0)
        hook.remove()
        # print(X_train_features)
        # print(x_train)
        return X_train_features,x_train

features = Getfeatures(model_path, model, train_on_gpu, testloader)
X_train_features, x_train = features.extract_features()