
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score,pairwise_distances
from sklearn.preprocessing import StandardScaler

# 读取鸢尾花数据集
column_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
df_iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=0, names=column_names)
X = df_iris.iloc[:, :4].values # 特征矩阵
y = df_iris.iloc[:, 4]# 真实标签
data_class={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
y=y.map(data_class).values


#标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
# 定义聚类目标数目
k = len(np.unique(y))

# 定义K均值算法函数
def kmeans(X, k):
    # 随机初始化k个质心
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    # 初始化聚类标签和上一次迭代的标签
    labels = np.zeros(X.shape[0])
    prev_labels = np.ones(X.shape[0])
    # 迭代直到收敛（标签不再变化）
    while not np.array_equal(labels, prev_labels):
        # 保存上一次的标签
        prev_labels = labels.copy()
        # 计算每个样本到每个质心的距离（欧氏距离）
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        # 分配每个样本到最近的质心
        labels = np.argmin(distances, axis=0)
        # 更新每个质心为其所属样本的均值
        for i in range(k):
            centroids[i] = X[labels == i].mean(axis=0)
    # 返回聚类标签和质心
    return labels, centroids

# 调用K均值算法函数，得到聚类结果
kmeans_labels, kmeans_centroids = kmeans(X, k)

#定义LVQ函数
def lvq(X, y, k, prototypes_per_class=1, alpha=0.1, tol=0.01, max_iter=1000):
    # 初始化原型向量和标签
    unique_labels = np.unique(y)
    num_classes = len(unique_labels)
    prototypes = np.zeros((k, X.shape[1]))
    prototype_labels = np.zeros(k)
    for i, label in enumerate(unique_labels):
        class_indices = np.where(y == label)[0]
        class_samples = X[class_indices]
        class_prototypes = class_samples[np.random.choice(class_samples.shape[0], prototypes_per_class, replace=False)]
        prototypes[i*prototypes_per_class:(i+1)*prototypes_per_class] = class_prototypes
        prototype_labels[i*prototypes_per_class:(i+1)*prototypes_per_class] = label

    # 迭代更新原型向量
    iter = 0
    error = np.inf
    while iter < max_iter and error > tol:
        # 对每个类别进行处理
        for label in unique_labels:
            # 找到该类别的样本和标签
            class_indices = np.where(y == label)[0]
            class_samples = X[class_indices]
            class_labels = y[class_indices]
            # 随机打乱该类别内部的顺序
            np.random.shuffle(class_samples)
            np.random.shuffle(class_labels)

            # 对该类别的每个样本进行处理
            for i in range(class_samples.shape[0]):
                # 计算样本到每个原型向量的距离（欧氏距离）
                distances = np.linalg.norm(class_samples[i] - prototypes, axis=1)
                # 找到最近的原型向量
                nearest_index = np.argpartition(distances, 0)[0]
                nearest_prototype = prototypes[nearest_index]
                nearest_prototype_label = prototype_labels[nearest_index]

                # 如果样本和原型向量属于同一类别，则靠近它；否则远离它
                if class_labels[i] == nearest_prototype_label:
                    prototypes[nearest_index] += alpha * (class_samples[i] - nearest_prototype)
                else:
                    prototypes[nearest_index] -= alpha * (class_samples[i] - nearest_prototype)

        # 逐渐降低学习率
        alpha *= 0.9

        # 计算当前的分类误差
        labels = prototype_labels[np.argpartition(np.linalg.norm(X[:, None] - prototypes, axis=2), 0, axis=1)[:, 0]]
        error = np.mean(labels != y)

        # 更新迭代次数
        iter += 1

    # 计算分类的准确率
    accuracy = 1 - error

    # 返回聚类标签，原型向量和准确率
    return labels, prototypes, accuracy



# 调用LVQ算法函数，得到聚类结果
lvq_labels, lvq_prototypes,acc = lvq(X, y, k)

# 定义EM算法函数
def em(X, k):
    # 随机初始化k个高斯分布的参数（均值，协方差，权重）
    means = X[np.random.choice(X.shape[0], k, replace=False)]
    covs = np.array([np.eye(X.shape[1]) for _ in range(k)])
    weights = np.ones(k) / k
    # 初始化聚类标签和上一次迭代的标签
    labels = np.zeros(X.shape[0])
    prev_labels = np.ones(X.shape[0])
    # 定义高斯分布的概率密度函数
    def gaussian_pdf(x, mean, cov):
        d = x.shape[0]
        return (1 / ((2 * np.pi)**(d / 2) * np.linalg.det(cov)**0.5)) * np.exp(-0.5 * (x - mean).T.dot(np.linalg.inv(cov)).dot(x - mean))
    # 迭代直到收敛（标签不再变化）
    while not np.array_equal(labels, prev_labels):
        # 保存上一次的标签
        prev_labels = labels.copy()
        # E步：计算每个样本属于每个高斯分布的后验概率（responsibility）
        resp = np.zeros((k, X.shape[0]))
        for i in range(k):
            for j in range(X.shape[0]):
                resp[i, j] = weights[i] * gaussian_pdf(X[j], means[i], covs[i])
        resp /= resp.sum(axis=0)
        # M步：更新每个高斯分布的参数（均值，协方差，权重）
        for i in range(k):
            Nk = resp[i].sum()
            means[i] = (resp[i].dot(X)) / Nk
            covs[i] = ((X - means[i]).T * resp[i]).dot(X - means[i]) / Nk
            weights[i] = Nk / X.shape[0]
        # 根据最大后验概率分配每个样本的标签
        labels = np.argmax(resp, axis=0)
    # 返回聚类标签和高斯分布参数
    return labels, (means, covs, weights)

# 调用EM算法函数，得到聚类结果
em_labels, em_params = em(X, k)

# 定义DBSCAN算法函数
def dbscan(X, eps, min_pts):
    # 初始化聚类标签为-1（表示未分类）
    labels = -np.ones(X.shape[0])
    # 初始化聚类簇的编号为0
    cluster_id = 0
    # 定义邻域查询函数，返回给定样本的邻域内的样本索引列表
    def region_query(x):
        neighbors = []
        for i in range(X.shape[0]):
            if np.linalg.norm(x - X[i]) < eps:
                neighbors.append(i)
        return neighbors
    # 定义邻域扩展函数，对给定的核心对象进行广度优先搜索，将其密度可达的样本划分到同一个簇中
    def expand_cluster(x_index, neighbors):
        nonlocal cluster_id
        labels[x_index] = cluster_id
        queue = neighbors.copy()
        while queue:
            q_index = queue.pop(0)
            if labels[q_index] == -1:
                labels[q_index] = cluster_id
                q_neighbors = region_query(X[q_index])
                if len(q_neighbors) >= min_pts:
                    queue.extend(q_neighbors)
            elif labels[q_index] == 0:
                labels[q_index] = cluster_id
    # 对每个未分类的样本进行处理
    for i in range(X.shape[0]):
        if labels[i] == -1:
            neighbors = region_query(X[i])
            if len(neighbors) < min_pts:
                labels[i] = 0 # 标记为噪声点
            else:
                cluster_id += 1 # 创建一个新的簇
                expand_cluster(i, neighbors) # 对该簇进行扩展
    # 返回聚类标签
    return labels


# 调用DBSCAN算法函数，得到聚类结果
dbscan_labels = dbscan(X, 0.8, k)

def agnes(X, k, distance_method='min', metric='euclidean'):
    # 初始化每个样本为一个簇
    clusters = [[i] for i in range(X.shape[0])]
    # 计算簇间的距离（最小距离法）
    distances = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            distances[i, j] = distances[j, i] = np.linalg.norm(X[i] - X[j])
    # 迭代直到达到目标簇的个数
    while len(clusters) > k:
        # 找到距离最小的两个簇
        min_dist = np.inf
        min_pair = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = distances[i, j]
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (i, j)
        # 合并这两个簇，并更新距离矩阵
        i, j = min_pair
        clusters[i].extend(clusters[j])
        clusters.pop(j)
        distances = np.delete(distances, j, axis=0)  # 删除第j行
        distances = np.delete(distances, j, axis=1)  # 删除第j列
        # 重新计算合并后的簇与其他簇之间的距离
        for m in range(len(clusters)):
            if m != i:
                if distance_method == 'min':  # 最小距离法
                    dist = np.min(pairwise_distances(X[clusters[i]], X[clusters[m]], metric=metric))
                elif distance_method == 'max':  # 最大距离法
                    dist = np.max(pairwise_distances(X[clusters[i]], X[clusters[m]], metric=metric))
                elif distance_method == 'avg':  # 平均距离法
                    dist = np.mean(pairwise_distances(X[clusters[i]], X[clusters[m]], metric=metric))
                else:
                    raise ValueError('Invalid distance method')
                distances[i, m] = distances[m, i] = dist
    # 根据簇划分聚类标签
    labels = np.zeros(X.shape[0])
    for i in range(len(clusters)):
        for j in clusters[i]:
            labels[j] = i
    # 返回聚类标签，簇和准确率
    return labels, clusters



# 调用AGNES算法函数，得到聚类结果
agnes_labels, agnes_clusters = agnes(X, k)

# 定义绘制散点图的函数，不同颜色表示不同的聚类标签
def plot_scatter(X, labels, title):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
    plt.title(title)
    plt.xlabel('SepalLength')
    plt.ylabel('SepalWidth')

# 绘制真实标签的散点图
plot_scatter(X, y, 'Ground Truth')

# 绘制K均值算法的散点图，用黑色的x表示质心
plot_scatter(X, kmeans_labels, 'K-Means')
plt.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], marker='x', color='black', s=100)

# 绘制LVQ算法的散点图，用黑色的x表示原型向量
plot_scatter(X, lvq_labels, 'LVQ')
plt.scatter(lvq_prototypes[:, 0], lvq_prototypes[:, 1], marker='x', color='black', s=100)

# 绘制EM算法的散点图，用黑色的x表示高斯分布的均值
plot_scatter(X, em_labels, 'EM')
plt.scatter(em_params[0][:, 0], em_params[0][:, 1], marker='x', color='black', s=100)

# 绘制DBSCAN算法的散点图
plot_scatter(X, dbscan_labels, 'DBSCAN')

# 绘制AGNES算法的散点图
plot_scatter(X, agnes_labels, 'AGNES')


# 显示所有的散点图
plt.show()

# 定义评估聚类性能的函数，使用轮廓系数作为指标
def evaluate(X, labels):
    score = silhouette_score(X, labels)
    return score

# 评估各个算法的聚类性能，并打印结果
kmeans_score = evaluate(X, kmeans_labels)
lvq_score = evaluate(X, lvq_labels)
em_score = evaluate(X, em_labels)
dbscan_score = evaluate(X, dbscan_labels)
agnes_score = evaluate(X, agnes_labels)

print('K-Means score:', kmeans_score)
print('LVQ score:', lvq_score)
print('EM score:', em_score)
print('DBSCAN score:', dbscan_score)
print('AGNES score:', agnes_score)
