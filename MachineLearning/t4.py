# 导入svm模块
from sklearn import svm, model_selection
import numpy as np

# 读取spiral.data文件里的数据
data = np.loadtxt('MachineLearning\spiral.data')

# 把每一行分成特征和标签
X = data[:, :-1] # 特征是最后一列之前的所有列
y = data[:, -1] # 标签是最后一列

# 用高斯核训练SVM
rbf_model = svm.SVC(kernel='rbf') # kernel='rbf'表示高斯核
rbf_model= svm.SVC(C=8, kernel='rbf', gamma=18)
rbf_model.fit(X, y) # 训练



# 画图
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral) # 画出原始数据点
h = .02  # 网格步长
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 # x轴范围
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 # y轴范围
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) # 网格点坐标矩阵
Z = rbf_model.predict(np.c_[xx.ravel(), yy.ravel()]) # 预测网格点的类别
Z = Z.reshape(xx.shape) # 调整形状和网格点一致
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8) # 画出决策边界
plt.show() # 显示图像
