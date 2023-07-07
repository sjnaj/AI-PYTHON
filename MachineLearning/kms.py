import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def euclidean_distance(x_training,row):
    dis = np.sum((row - x_training)**2)
    dis = np.sqrt(dis)
    return dis
# 定义 K 均值聚类算法
def k_means(dataset, k, iteration):
    # 初始化簇心向量
    index = random.sample(list(range(len(dataset))), k)
    vectors = []
    for i in index:
        vectors.append(dataset[i])
    # 初始化标签
    labels = []
    for i in range(len(dataset)):
        labels.append(-1)
    # 根据迭代次数重复 K 均值聚类过程
    while iteration > 0:
        # 初始化簇
        C = []
        for i in range(k):
            C.append([])

        for labelIndex, item in enumerate(dataset):
            classIndex = -1
            minDist = 1e6
            for i, point in enumerate(vectors):
                dist = euclidean_distance(item, point)  
                if dist < minDist:
                    classIndex = i
                    minDist = dist
            C[classIndex].append(item)
            labels[labelIndex] = classIndex

        for i, cluster in enumerate(C):
            clusterHeart = []
            dimension = len(dataset[0])
            for j in range(dimension):
                clusterHeart.append(0)
            for item in cluster:
                for j, coordinate in enumerate(item):
                    clusterHeart[j] += coordinate / len(cluster)
            vectors[i] = clusterHeart
        iteration -= 1
    return C, labels

# 获取 Iris 数据集
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data") #读取数据
print(len(data))
data=data.values
np.random.shuffle(data) #打乱数据顺序
x_data = data[:,:4].astype(float) #提取特征列，并转换为浮点型数组
y_data = data[:,4] #提取标签列

# 将数据集进行标准化处理
scaler = StandardScaler()
scaled_data = scaler.fit_transform(x_data)

# 使用 PCA 进行降维
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)



# 运行 K 均值聚类算法

k = len(np.unique(y_data))
iteration = 10
clusters, labels = k_means(pca_data, k, iteration)

# 将结果绘制成散点图
plt.figure(figsize=(8, 6))
for i in range(k):
    cluster_data = np.array(clusters[i])
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i+1}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering Result')
plt.legend()
plt.show()
