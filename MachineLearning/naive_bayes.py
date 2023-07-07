# 导入相关库
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 定义一个函数来计算先验概率
def prior_prob(y_train, label):
    # 统计训练集中每个类别的样本数
    total_count = y_train.shape[0]
    class_count = np.sum(y_train == label)
    # 使用拉普拉斯修正
    return (class_count + 1) / (total_count + np.unique(y_train).shape[0])

# 定义一个函数来计算条件概率
def conditional_prob(x_train, y_train, feature_col, feature_val, label):
    # 统计训练集中属于某个类别且某个特征取某个值的样本数
    x_filtered = x_train[y_train == label]
    numerator = np.sum(x_filtered[:, feature_col] == feature_val)
    # 统计训练集中属于某个类别的样本数
    denominator = np.sum(y_train == label)
    # 使用拉普拉斯修正
    return (numerator + 1) / (denominator + np.unique(x_train[:, feature_col]).shape[0])

# 定义一个函数来计算后验概率
def posterior_prob(x_train, y_train, x_test, label):
    # 初始化后验概率为先验概率
    post_prob = prior_prob(y_train, label)
    # 遍历每个特征，计算条件概率，并累乘到后验概率上
    for i in range(x_train.shape[1]):
        post_prob *= conditional_prob(x_train, y_train, i, x_test[i], label)
    # 返回后验概率
    return post_prob

# 定义一个函数来预测单个样本的分类结果
def predict(x_train, y_train, x_test):
    # 初始化一个列表来存储所有可能的后验概率
    post_probs = []
    # 遍历每个类别，计算后验概率，并添加到列表中
    for i in np.unique(y_train):
        post_probs.append(posterior_prob(x_train, y_train, x_test, i))
    # 返回后验概率最大的类别作为预测结果
    return np.argmax(post_probs)

# 加载数据集
# iris = load_iris()
# X = iris.data # 特征矩阵
# y = iris.target # 分类标签

# 定义训练数据
import pandas as pd
data = pd.read_csv('MachineLearning\watermelon3.csv')
dicts = [
    {
        '乌黑': 0,
        '青绿': 1,
        '浅白': 2
    },
    {
        '蜷缩': 0,
        '稍蜷': 1,
        '硬挺': 2
    },
    {
        '沉闷': 0,
        '浊响': 1,
        '清脆': 2
    },
    {
        '清晰': 0,
        '稍糊': 1,
        '模糊': 2

    }, {
        '凹陷': 0,
        '稍凹': 1,
        '平坦': 2
    }, {
        '硬滑': 0,
        '软粘': 1,
    }
]
# data=data.applymap(dicts[0].get,na_action='ignore')
data['色泽'] = data['色泽'].apply(dicts[0].get)
data['根蒂'] = data['根蒂'].apply(dicts[1].get)
data['敲声'] = data['敲声'].apply(dicts[2].get)
data['纹理'] = data['纹理'].apply(dicts[3].get)
data['脐部'] = data['脐部'].apply(dicts[4].get)
data['触感'] = data['触感'].apply(dicts[5].get)


X = data.values[:, 1:9].astype(np.float64)
y = data.values[:, 9]
y[y == '是'] = 1
y[y == '否'] = 0

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用训练集训练分类器（无需额外操作）

# 使用测试集预测分类结果
y_pred = []
for x in X_test:
    y_pred.append(predict(X_train, y_train, x))

accuracy = np.mean(y_pred == y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))
x1=[1.,0.,1.,0.,0.,0.,0.697,0.460]
y1=predict(X_train,y_train,x1)
print("测1是:","好瓜"if y1==1 else"坏瓜")