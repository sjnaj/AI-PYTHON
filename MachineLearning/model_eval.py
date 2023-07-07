import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,accuracy_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from scipy.stats import ttest_rel

# 指定汽车评估数据集的网址
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'

# 从网上直接读取数据，并指定分隔符为逗号
df = pd.read_csv(url, sep=',')

# 查看数据集的基本信息
print(df.head())
print(df.shape)

# 将最后一列作为类别标签，并将其转换为数值型
y = df.iloc[:,-1]
#改为二分类
y = y.replace({'unacc':0, 'acc':1, 'good':0, 'vgood':0})
# 将前几列作为特征矩阵，并将其转换为数值型
X = df.iloc[:,:-1]
X = pd.get_dummies(X) # 使用one-hot编码处理类别型特征

# 查看X和y的基本信息
print(X.head())
print(X.shape)
print(y.head())
print(y.shape)



print('对使用不同核函数的SVM模型进行检验')

#创建SVM模型，设置两个不同的核函数
svm_rbf=SVC(kernel='rbf',probability=True)
svm_poly = SVC(kernel='poly',probability=True)

#保存名称信息
svm_rbf.__str__="svm_rbf"
svm_poly.__str__="svm_poly"

# 定义一个函数来绘制P-R曲线和ROC曲线
def plot_curves(y_true, y_prob, label):
  # 计算P-R曲线上的点
  
  # y_prob:Target scores, can either be probability estimates of the positive class, or non-thresholded measure of decisions (as returned by decision_function on some classifiers).

  p, r, _ = precision_recall_curve(y_true, y_prob)
  # 计算P-R曲线下面积（AP）
  ap = auc(r, p)
  # 计算ROC曲线上的点
  fpr, tpr, _ = roc_curve(y_true, y_prob)
  # 计算ROC曲线下面积（AUC）
  auc_value = auc(fpr, tpr)
  fig, (ax1, ax2) = plt.subplots(1, 2)
  fig.suptitle(label + ' Curves')
  # 绘制P-R曲线
  ax1.plot(r,p)
  ax1.set_xlabel('Recall')
  ax1.set_ylabel('Precision')
  ax1.set_title('PR Curve (AP = %.4f)' % ap)
  # 绘制ROC曲线
  ax2.plot(fpr,tpr)
  ax2.plot([0.5], [0.5], linestyle='--')
 
  
  ax2.set_xlabel('False Positive Rate')
  ax2.set_ylabel('True Positive Rate')
  ax2.set_title('ROC Curve (AUC = %.4f)' % auc_value)
  #注意关闭图像后程序才能继续运行
  plt.show()
  
# 创建一个分层10折交叉验证分割器对象
skf = StratifiedKFold(n_splits=10)


# 对每个分类器进行循环
for clf in [svm_rbf,svm_poly]:
    # 获取分类器名称
    label=clf.__str__
    print(label + ':')
    # 得到预测的类别标签
    y_pred = cross_val_predict(clf,X,y,cv=skf,n_jobs=-1)
    
    #绘制，传入用'decision_function'计算的预测函数结果
    plot_curves(y,cross_val_predict(clf,X,y,cv=skf,n_jobs=-1,method='decision_function'),label)
    
    # 计算并打印混淆矩阵
    cm = confusion_matrix(y,y_pred)
    print('Confusion Matrix:')
    print(cm)
    
    
    #计算并打印准确率 (A)
    a=accuracy_score(y,y_pred)
    print('Accuracy: %.4f' % a)
    # 计算并打印精确率（P）
    p = precision_score(y,y_pred )
    print('Precision: %.4f' % p)
    # 计算并打印召回率（R）
    r = recall_score(y,y_pred )
    print('Recall: %.4f' % r)
    # 计算并打印F1值（F1）
    f1 = f1_score(y,y_pred )
    print('F1 Score: %.4f' % f1) 
    
    
#得到每一折的准确率

svm_rbf_scores = cross_val_score(svm_rbf, X, y, cv=10,n_jobs=-1, scoring='accuracy')


svm_poly_scores = cross_val_score(svm_poly, X, y, cv=10, n_jobs=-1,scoring='accuracy')

#利用准确率进行t检验，与使用错误率等价
_,p=ttest_rel(1-svm_rbf_scores,1-svm_poly_scores)

alpha = 0.05 # 设置显著度
print('p-value: %.4f'%p)
if p>alpha:
    #平均准确率高者性能高
    print('模型性能有差异，且模型%s性能较优' % (svm_rbf.__str__ if np.mean(svm_rbf_scores)>np.mean(svm_poly_scores) else svm_poly.__str__))
else:
    print("模型无显著差异")
  
    
