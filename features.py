import torch
import numpy as np
from cnn_model import Net
from get_data import get_data_loaders


model_path ='E:\python\study python\CNN-RF\model\model_1.pth'
model = Net()
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    model.cuda()

trainloader, validloader, testloader = get_data_loaders()

class Parameter():
    def __init__(self,model_path,model,train_on_gpu,trainloader,testloader):
        self.model_path = model_path
        self.model = model
        self.train_on_gpu = train_on_gpu
        self.trainloader = trainloader
        self.testloader = testloader

class Getfeatures(Parameter):
    def __init__(self, model_path, model, train_on_gpu,trainloader,testloader):
        super().__init__(model_path, model,train_on_gpu,trainloader,testloader)
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
        print("Train features:", X_train_features)
        print("Train labels:", x_train)
        print("Test features:", Y_test_features)
        print("Test labels:", y_test)
        return X_train_features,x_train,Y_test_features,y_test
    


features = Getfeatures(model_path, model, train_on_gpu,trainloader, testloader)
X_train_features, x_train,Y_test_features,y_test = features.extract_features()
