import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import csv

# 读取数据集
iris_data = pd.read_csv(r'D:\dataenclorse\second\iris_train.csv')
X = iris_data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].values
y = iris_data["species"].values

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 欧氏距离函数
def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2) ** 2))

# 手动实现KNN函数
def knn_predict(X_train, y_train, X_new, k=1):
    predictions = []
    for x_new in X_new:
        # 计算新数据点到所有训练数据点的距离
        distances = [euclidean_distance(x_new, x_train) for x_train in X_train]
        # 获取按距离排序的索引
        k_indices = np.argsort(distances)[:k]
        # 获取k个最近邻的标签
        k_nearest_labels = [y_train[i] for i in k_indices]
        # 多数投票，最常见的类别标签
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])
    return predictions

# 读取待预测的新数据点
iris_test_data = pd.read_csv(r'D:\dataenclorse\second\iris_test.csv')
X_new = iris_test_data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].values

# 预测新数据点的类别
predictions = knn_predict(X_train, y_train, X_new, k=1)
print("预测的目标类别是：{}".format(predictions))
# 将预测结果保存到新的CSV文件中
file_path = r'D:\dataenclorse\second\test_manual_knn.csv'
with open(file_path, 'w', encoding='utf-8', newline='') as f:
    fieldnames = ['sepal length(cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'class']
    f_csv = csv.DictWriter(f, fieldnames=fieldnames)
    f_csv.writeheader()
    for i in range(len(predictions)):
        f_csv.writerow({
            'sepal length(cm)': X_new[i][0],
            'sepal width (cm)': X_new[i][1],
            'petal length (cm)': X_new[i][2],
            'petal width (cm)': X_new[i][3],
            'class': predictions[i]
        })

print(f"预测结果已保存到 {file_path}")
