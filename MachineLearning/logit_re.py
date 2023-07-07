import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

import pandas as pd
data = pd.read_table('D:\Desktop\AI-PYTHON\MachineLearning\watermelon3a.txt',delimiter=',')
X_train=data.values[:,1:3].astype(np.float64)
y_train=data.values[:,3]
y_train[y_train=='是']=1
y_train[y_train=='否']=0
y_train=y_train.astype(np.float64)
print(X_train,y_train)
X=X_train
y=y_train
# 定义数据集
# X = np.array([[2,3], [1,2], [3,4], [4,5], [2,2], [2,0], [1,1], [3,3]])
# y = np.array([1, 1, 1, 1, 0, 0, 0, 0])

# 定义逻辑回归模型
model = LogisticRegression()

# 拟合模型
model.fit(X, y)

# 预测新的数据，并输出预测结果
print(model.predict(X))

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=y)
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
plt.show()