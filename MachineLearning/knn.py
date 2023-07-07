#导入numpy和operator模块
import numpy as np
import operator
import pandas as pd
#定义欧氏距离函数
def euclidean_distance(x_training,row):
    dis = np.sum((row - x_training)**2,axis=1)
    dis = np.sqrt(dis)
    return dis

#定义预测函数
def predict(dis_sort,y_training,k):
    statistics = {}#定义字典，用于统计k个数据点中各个类别的鸢尾花出现的次数
    for i in range(k):
        rank = dis_sort[i]
        if y_training[rank] in statistics:
            statistics[y_training[rank]] = statistics[y_training[rank]] + 1
        else:
            statistics[y_training[rank]] = 1
    sort_statis = sorted(statistics.items(), key=operator.itemgetter(1), reverse=True)#对statistics字典按照value进行排序
    y_predict = sort_statis[0][0]
    return y_predict

#定义knn函数
def knn(x_training,y_training,x_test,k):
    y_predict = [] #定义一个空列表，用于存储预测结果
    for row in x_test: #对于每一个测试样本
        dis = euclidean_distance(x_training,row) #计算它和训练样本的距离
        dis_sort = np.argsort(dis) #对距离进行排序，并得到索引值
        y_pred = predict(dis_sort,y_training,k) #根据k个最近邻的类别进行预测
        y_predict.append(y_pred) #将预测结果添加到y_predict列表中
    return np.array(y_predict) #将y_predict列表转换为数组并返回
# import os
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

#从数据链接中下载Iris数据集，并用numpy.loadtxt函数读取数据，将其分为x_training, y_training, x_test, y_test四个数组
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data") #读取数据
print(len(data))
data=data.values
np.random.shuffle(data) #打乱数据顺序
x_data = data[:,:4].astype(float) #提取特征列，并转换为浮点型数组
y_data = data[:,4] #提取标签列
split = int(len(data)*0.8) #按照8:2的比例划分训练集和测试集
x_training = x_data[:split] #训练集特征
y_training = y_data[:split] #训练集标签
x_test = x_data[split:] #测试集特征
y_test = y_data[split:] #测试集标签

#调用knn函数，传入x_training, y_training, x_test, k作为参数，得到y_predict数组，并与y_test数组进行比较，计算准确率
k = 10 #设置k的值为5
y_predict = knn(x_training,y_training,x_test,k) #得到预测结果
accuracy = np.sum(y_predict == y_test) / len(y_test) #计算准确率
print("K=",k,"\nThe accuracy of the knn algorithm is:",accuracy) #打印准确率
