import numpy as np



# 定义sigmoid函数和它的导数


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 定义二分类交叉熵损失函数和它的导数


def binary_crossentropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def binary_crossentropy_derivative(y_true, y_pred):
    return -(y_true / y_pred - (1 - y_true) / (1 - y_pred))


# 初始化参数
np.random.seed(42)
W1 = np.random.randn(8, 16)  # 输入层到隐藏层的权重矩阵
b1 = np.random.randn(16)  # 隐藏层的偏置向量
W2 = np.random.randn(16, 1)  # 隐藏层到输出层的权重矩阵
b2 = np.random.randn(1)  # 输出层的偏置向量

# 定义学习率
lr = 0.05

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
print(X.shape)
y = data.values[:, 9]

y[y == '是'] = 1
y[y == '否'] = 0
y = y.reshape(17, 1).astype(np.float64)
print(np.array2string(X,separator=', '),np.array2string(y,separator=', '))


# 定义训练轮数
epochs = 10000

# 训练过程
for epoch in range(epochs):
    # 前向传播
    Z1 = X.dot(W1) + b1  # 隐藏层的线性输出
    A1 = sigmoid(Z1)  # 隐藏层的激活输出
    Z2 = A1.dot(W2) + b2  # 输出层的线性输出
    A2 = sigmoid(Z2)  # 输出层的激活输出
    # 计算损失
    loss = binary_crossentropy(y, A2)
    # 反向传播
    dZ2 = binary_crossentropy_derivative(
        y, A2) * sigmoid_derivative(Z2)  # 输出层的梯度
    dW2 = A1.T.dot(dZ2)  # 输出层到隐藏层的权重梯度
    db2 = np.sum(dZ2, axis=0)  # 输出层的偏置梯度
    dZ1 = dZ2.dot(W2.T) * sigmoid_derivative(Z1)  # 隐藏层的梯度
    dW1 = X.T.dot(dZ1)  # 隐藏层到输入层的权重梯度
    db1 = np.sum(dZ1, axis=0)  # 隐藏层的偏置梯度

    # 参数更新
    W1 = W1 - lr * dW1  # 更新输入层到隐藏层的权重
    b1 = b1 - lr * db1  # 更新隐藏层的偏置
    W2 = W2 - lr * dW2  # 更新隐藏层到输出层的权重
    b2 = b2 - lr * db2  # 更新输出层的偏置

    # 打印损失
    if epoch % (epochs//10) == 0:
        print(f'Epoch {epoch + 1}, loss: {loss:.4f}')

# 测试过程
# 给定一个新的输入，计算输出
X_new = X
Z1_new = X_new.dot(W1) + b1
A1_new = sigmoid(Z1_new)
Z2_new = A1_new.dot(W2) + b2
A2_new = sigmoid(Z2_new)
A2_new[A2_new >= 0.5] = 1
A2_new[A2_new < 0.5] = 0
print(f'Output: {A2_new}')
