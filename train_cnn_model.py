import torch
import numpy as np
import matplotlib.pyplot as plt
from cnn_model import Net
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim



# 定义超参数
lr = 0.001
batch_size = 64
num_epochs = 20
valid_size = 0.2 # 用作验证的训练集百分比
model_path ='E:\python\study python\CNN-RF\model\model_1.pth' #模型保存路径

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
trainset = CIFAR10(root='.\data', train=True, download=True, transform=transform_train)
testset = CIFAR10(root='.\data', train=False, download=True, transform=transform_test)

# 选择用于验证的训练集样本
num_train =20000#控制样本数量
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

# 定义模型、损失函数和优化器
model = Net() #加载模型
device = torch.device("cuda:0")  # 使用 GPU 0
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# 使用GPU
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    model.cuda()

class Parameter:
    def __init__(self, model, trainloader, validloader, testloader, optimizer, criterion, train_on_gpu, model_path):
        # 初始化训练过程所需的参数
        self.model = model
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_on_gpu = train_on_gpu
        self.model_path = model_path


# 训练管理类（子类）
class trasiningManager(Parameter):
    def __init__(self, model, trainloader, validloader, testloader, optimizer, criterion, train_on_gpu, model_path, epochs=30):
        super().__init__(model, trainloader, validloader, testloader, optimizer, criterion, train_on_gpu, model_path)     
        # 初始化训练管理类特有的参数
        self.epochs = epochs
        self.trainloss = []
        self.validloss = []
        self.accuracy = []
        # 最小验证损失初始化
        self.valid_loss_min = np.Inf

    def train_model(self):
        self.model.train()
        train_loss = 0.0
        for data, target in self.trainloader:
        # move tensors to GPU if CUDA is available
            if self.train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            self.optimizer.zero_grad()#梯度置零
            # 正向传递：通过将输入传递到模型来计算预测输出
            output = self.model(data)
            # 计算批量损失
            loss = self.criterion(output, target)
            # 反向通过：计算损耗相对于模型参数的梯度
            loss.backward()
            # perform a single optimization step (parameter update)
            self.optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
        return train_loss/len(self.trainloader.dataset)
    
    def validate_model(self):
        self.model.eval()  # 设置模型为评估模式
        valid_loss = 0.0

        with torch.no_grad():  # 不计算梯度
            for data, target in self.validloader:
                if self.train_on_gpu:
                    data, target = data.cuda(), target.cuda()

                output = self.model(data)  # 正向传播

                loss = self.criterion(output, target)  # 计算损失

                valid_loss += loss.item() * data.size(0)  # 累加验证损失

        return valid_loss / len(self.validloader.dataset)
    
    def calculate_accuracy(self):
        correct = 0  # 预测正确的图片数
        total = 0  # 总共的图片数
        with torch.no_grad():
            for images, labels in self.testloader:
                #images, labels = data
                if self.train_on_gpu:
                    images, labels = images.cuda(), labels.cuda()
                outputs = self.model(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
        accurary = 100* correct /total
        return accurary

        
    def save_model(self,valid_loss):
        if valid_loss <= self.valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.valid_loss_min,valid_loss))
            torch.save(self.model.state_dict(), self.model_path)
            self.valid_loss_min = valid_loss
           
    # 绘制准确率变化曲线
    def plot_accuracy_curve(self):
        plt.plot(self.accuracy, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend(frameon=False)
        plt.show()
    #绘制学习曲线
    def train_vail_curve(self):
        plt.plot(self.trainloss, label='Training loss')
        plt.plot(self.validloss, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend(frameon=False)
        plt.show()

    def train_and_evaluate(self):
        for epoch in range(1, self.epochs + 1):
            # 训练
            train_loss = self.train_model()
            # 验证
            valid_loss = self.validate_model()
            # 准确率
            acc = self.calculate_accuracy()
            
            # 记录损失和准确率
            self.trainloss.append(train_loss)
            self.validloss.append(valid_loss)
            self.accuracy.append('%d %%'%acc)

            # 输出结果
            print(f"Epoch {epoch}/{self.epochs}")
            print(f"训练集损失: {train_loss:.4f}, 验证集损失: {valid_loss:.4f}, 测试集准确率: {acc:.2f}%")
            # 保存模型
            self.save_model(valid_loss)

        # 绘制训练过程中的准确率变化曲线
        self.plot_accuracy_curve()    
        self.train_vail_curve()




trainer = trasiningManager(model, trainloader, validloader, testloader, optimizer, criterion, train_on_gpu, model_path, epochs=30)
trainer.train_and_evaluate()



