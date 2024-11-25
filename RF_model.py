from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from features import GetFeatures
from get_data import get_data_loaders
from cnn_model import Net
import torch
import matplotlib.pyplot as plt
import numpy as np

trainloader, validloader, testloader,testset = get_data_loaders()
model = Net() #加载模型
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    model.cuda()
state_dict = torch.load('E:\python\study python\CNN-RF\model\model_1.pth')
model.load_state_dict(state_dict)
features = GetFeatures(state_dict, model, train_on_gpu,trainloader, testloader)
X_train_features, x_train,Y_test_features,y_test = features.extract_features()


#建立随机森林模型(300棵树，每棵树的层数为19（可能过拟合），每棵树选用数据集的最大特征数为10)
rf = RandomForestClassifier(n_estimators=300, random_state=71,max_depth=19,n_jobs=-1,max_features=10)
#数据标准化 特征标度
scaler = StandardScaler()
X_train_features = scaler.fit_transform(X_train_features)
y_features_test = scaler.transform(Y_test_features)
#使用交叉验证来评估模型性能
from sklearn.model_selection import cross_val_score
# cv_scores = cross_val_score(rf,X_train_features, x_train, cv=10)
# # 打印交叉验证得分,如果交叉验证得分和测试集准确率差别较大，说明模型可能存在过拟合问题
# print("Cross-validation scores:", cv_scores)
# print("Average score:", np.mean(cv_scores))

#训练模型
rf.fit(X_train_features, x_train)
y_pred =rf.predict(y_features_test)
# 计算分类精度
accuracy=accuracy_score( y_test,y_pred )*100
print('CNN+RF训练准确度为：{:.2f} %'.format(accuracy))

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

#显示图片
# 随机抽取8张测试集图片进行可视化查看分类效果
# fig, axs = plt.subplots(2, 4, figsize=(10, 5))
fig, axs = plt.subplots(2, 4, figsize=(10, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
axs = axs.ravel()
for i in range(8):
    index = np.random.randint(len(testset))
    img = testset[index][0].numpy().transpose((1, 2, 0))
    label = y_test[index]
    feature = y_features_test[index]
    pred = rf.predict([feature])[0]
    axs[i].imshow((img+1)/2)
    axs[i].set_title('True: {}\nPred: {}'.format(classes[label], classes[pred]),
                     color=("green" if label==pred else "red"))
    axs[i].set_xticks([])
    axs[i].set_yticks([])
plt.show()




#混淆矩阵
from sklearn.metrics import confusion_matrix
# 使用模型预测并计算混淆矩阵
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
tn = conf_mat[0, 0]
fp = conf_mat[0, 1]
fn = conf_mat[1, 0]
tp = conf_mat[1, 1]
# 输出混淆矩阵和模型的准确率等指标
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"TN: {tn}\nFP: {fp}\nTP: {tp}\nFN: {fn}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f},Accuracy: {accuracy:.4f}")




#混淆矩阵图
import itertools
# 获取类别标签和混淆矩阵
confusion_mtx = confusion_matrix(y_true=y_test, y_pred=y_pred)
# 绘制混淆矩阵
plt.figure(figsize=(8, 8))
plt.imshow(confusion_mtx, cmap='Blues')
plt.colorbar()

# 添加标签
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# 添加数字，并设置颜色
thresh = confusion_mtx.max() / 2.
for i, j in itertools.product(range(confusion_mtx.shape[0]), range(confusion_mtx.shape[1])):
    plt.text(j, i, format(confusion_mtx[i, j], 'd'),
             horizontalalignment="center", verticalalignment="center",
             color="white" if confusion_mtx[i, j] > thresh else "black")

# 设置图像属性
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

