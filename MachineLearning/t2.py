import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score

# 加载数据集
data_iris = load_iris()
X_iris = data_iris.data
y_iris = data_iris.target

data_breast = load_breast_cancer()
X_breast = data_breast.data
y_breast = data_breast.target
# print(X_iris,y_iris,X_breast,y_breast)

# 10折交叉验证
skf = StratifiedKFold(n_splits=10)
lr = LogisticRegression(max_iter=10000)

mean_acc_iris = 0
for train_index, test_index in skf.split(X_iris, y_iris):
    X_train, X_test = X_iris[train_index], X_iris[test_index]
    y_train, y_test = y_iris[train_index], y_iris[test_index]
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mean_acc_iris += accuracy_score(y_test, y_pred)

mean_acc_iris /= 10
print('Iris data set logistic regression 10-fold cross-validation accuracy:', mean_acc_iris) 

mean_acc_breast = 0
for train_index, test_index in skf.split(X_breast, y_breast):
    X_train, X_test = X_breast[train_index], X_breast[test_index]
    y_train, y_test = y_breast[train_index], y_breast[test_index]
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mean_acc_breast += accuracy_score(y_test, y_pred)

mean_acc_breast /= 10
print('Breast Cancer data set logistic regression 10-fold cross-validation accuracy:', mean_acc_breast) 

# 留一法
loo = LeaveOneOut()
lr = LogisticRegression(max_iter=10000)

error_iris = 0
for train_index, test_index in loo.split(X_iris, y_iris):
    X_train, X_test = X_iris[train_index], X_iris[test_index]
    y_train, y_test = y_iris[train_index], y_iris[test_index]
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    if y_pred != y_test:
        error_iris += 1

error_iris /= len(X_iris)
print('Iris data set logistic regression leave-one-out error rate:', error_iris)

error_breast = 0
for train_index, test_index in loo.split(X_breast, y_breast):
    X_train, X_test = X_breast[train_index], X_breast[test_index]
    y_train, y_test = y_breast[train_index], y_breast[test_index]
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    if y_pred != y_test:
        error_breast += 1

error_breast /= len(X_breast)
print('Breast Cancer data set logistic regression leave-one-out error rate:', error_breast)