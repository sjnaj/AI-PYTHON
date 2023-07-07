import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_blobs
import pandas as pd
data = pd.read_table('D:\Desktop\AI-PYTHON\MachineLearning\watermelon3a.txt',delimiter=',')
X=data.values[:,1:3].astype(np.float64)
y=data.values[:,3]
y[y=='是']=1
y[y=='否']=0
y=y.astype(np.float64)


# 建立LDA模型
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.title('LDA')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 计算直线斜率和截距
k = lda.coef_[0][1]/lda.coef_[0][0]
b = lda.intercept_[0]/lda.coef_[0][0]

# 画出决策边界
x_vals = np.array([-20, 20])
line = b + k * x_vals
plt.plot(x_vals, line, '--')

plt.show()
