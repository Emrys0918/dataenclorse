import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 加载和准备数据
iris_data = pd.read_csv(r'D:\dataenclorse\second\iris_train.csv')
X = iris_data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].values
y = iris_data["species"].values

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 初始化并训练KNN分类器
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# 加载测试数据
test_data = pd.read_csv(r'D:\dataenclorse\second\iris_test.csv')
X_new = test_data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].values

# 进行预测
predictions = knn.predict(X_new)

# 将预测结果写入新的CSV文件
output_file_path = r'D:\dataenclorse\second\test.csv'
test_data['class'] = predictions

# 指定要输出的列顺序
output_columns = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "class"]
test_data.to_csv(output_file_path, index=False, columns=output_columns)

print("预测的目标类别是：{}".format(predictions))
