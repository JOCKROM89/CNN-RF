# CNN-RF
Use CNN-RF to  Identify images

使用卷积神经网络和随机森林来实现对图像集的辨别分类
## 环境
* Python 3.10及以上
* Pytorch
* CIFAR 10图像集
## 模型流程

CIFAR-10数据集加载——CNN模型——训练模型——图像特征提取——随机森林模型
## 数据集

CIFAR-10 数据集由 10 个类的 60000 张 32x32 彩色图像组成，每个类有 6000 张图像。有 50000 张训练图像和 10000 张测试图像。

数据集分为 5 个训练批次和 1 个测试批次，每个批次有 10000 张图像。测试批次包含每个类中随机选择的 1000 张图像。训练批次包含按随机顺序排列的剩余图像，但某些训练批次可能包含来自一个类的图像多于另一个类的图像。在它们之间，训练批次包含来自每个类的 5000 张图像。
以下是数据集中的类，以及每个类的 10 张随机图像：

![image](https://github.com/JOCKROM89/CNN-RF/blob/master/data/Snipaste_2024-11-18_09-18-54.png)

下载地址：https://www.cs.toronto.edu/~kriz/cifar.html

要想在模型中顺利使用数据集需要对数据集进行处理。在文件get_data.py中对CIFAR-10 数据集中的图片数据进行了数据增强和预处理，同时定义训练时培训和验证批次的采样器。将数据集和采样器组合构成数据加载器，用于CNN-RF模型数据的加载

## CNN模型
具体代码参考cnn_model.py
该卷积神经网络模型是由4个卷积层、2个池化层、2个全连接层构成

## 训练与验证CNN模型
通过运行train_cnn_model.py来对CNN模型训练与验证，在训练过程中如果准确率提升则对训练的模型进行保存。训练完成后会输出准确率变化曲线和模型学习曲线，用来显示在训练过程中模型参数的变化。
该模型的训练在验证集上准确率最佳在89%左右，训练次数为70~80次。过多的训练次数并不能提高模型对图像分类的准确性。

## 图像特征提取
将CNN模型的全连接层（fc3）替换为Identity层，用来提取图像的特征。提取的数据是四维数据，随机森林无法调用因此需要将其调整为二维数据。利用以下代码将训练集的特征和标签合并为numpy数组，测试集同理详见features.py文件。

 > X_train_features = np.concatenate(self.X_train_features, axis=0)
 ## RF模型
 构造随机森林模型，使用的模型具有300棵树，每棵树的层数为19，随机数生成器的种子值为17来确保每次实验都能得到相同的结果。每棵树在分裂时会随机选择 10 个特征。
> rf = RandomForestClassifier(n_estimators=300, random_state=71,max_depth=19,n_jobs=-1,max_features=10)



## 随机森林混淆矩阵

![image](https://github.com/JOCKROM89/CNN-RF/blob/master/data/Snipaste_2024-11-24_20-39-32.png)

## 随机抽取8张图片进行验证

![image](https://github.com/JOCKROM89/CNN-RF/blob/master/data/Snipaste_2024-11-24_20-39-13.png)
