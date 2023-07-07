import numpy as np
import math
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, num_iterations=100000, learning_rate=0.1):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.W = np.zeros((n, 1))
        self.b = 0
        dw=db=1
        i=0
        while np.linalg.norm(self.learning_rate*np.append(dw,db))>1e-6:
            Z = np.dot(X, self.W) + self.b
            A = self.sigmoid(Z)
            cost = (-1 / m) * np.sum(np.multiply(y, np.log(A)) + np.multiply((1 - y), np.log(1 - A)))
            dz = A - y.reshape(-1, 1)
            dw = (1 / m) * np.dot(X.T, dz)
            db = (1 / m) * np.sum(dz)
            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            if i % 1000 == 0:
                print("Iteration %i, cost: %f" % (i, cost))
            i+=1

    def predict_prob(self, X):
        return self.sigmoid(np.dot(X, self.W) + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)
# X_train = np.array([[1.0, 2.0], [2.0, 1.0], [2.0, 3.0], [3.0, 2.0]])
# y_train = np.array([0, 0, 1, 1])

import pandas as pd
data = pd.read_table('D:\Desktop\AI-PYTHON\MachineLearning\watermelon3a.txt',delimiter=',')
X_train=data.values[:,1:3].astype(np.float64)
y_train=data.values[:,3]
y_train[y_train=='是']=1
y_train[y_train=='否']=0
y_train=y_train.astype(np.float64)
print(X_train,y_train)

lr = LogisticRegression(num_iterations=10000, learning_rate=0.5)

lr.fit(X_train, y_train)

X_test = np.array([0.437,0.211]).reshape(1, -1)
pred_prob = lr.predict(X_train)
print(f"Predict: {pred_prob}")
print(lr.W)

fig, ax = plt.subplots()
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', s=25)
min_x, max_x = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
min_y, max_y = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(min_x, max_x, 0.1), np.arange(min_y, max_y, 0.1))
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.4)
plt.show()