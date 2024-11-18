import torch
import numpy as np
import matplotlib.pyplot as plt
from cnn_model import Net
import torch.nn as nn
import torch.optim as optim
from get_data import get_data_loaders

# 定义超参数
lr = 0.001
num_epochs = 20
# 用作验证的训练集百分比
model_path ='E:\python\study python\CNN-RF\model\model_1.pth' #模型保存路径
# 加载数据
trainloader, validloader, testloader = get_data_loaders()
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
        best_acc = 0.0
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
            if acc > best_acc:
                best_acc = acc
                # 保存模型
                self.save_model(valid_loss)
                print(f"模型保存, 新的最佳准确率: {best_acc:.2f}%")

        # 绘制训练过程中的准确率变化曲线
        self.plot_accuracy_curve()    
        self.train_vail_curve()




trainer = trasiningManager(model, trainloader, validloader, testloader, optimizer, criterion, train_on_gpu, model_path, epochs=3)
trainer.train_and_evaluate()



