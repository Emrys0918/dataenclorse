from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import csv

iris_data = load_iris()

# 构造训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split( \
    iris_data['data'], iris_data['target'], random_state=0)

# 构造KNN模型
knn = KNeighborsClassifier(n_neighbors=1)

# 训练模型
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 做出预测
file_path = ("D:\\dataenclorse\\second\\iris_test.csv")
data = pd.read_csv(r'iris_test.csv', usecols=["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"])
grouped_data = [data.iloc[i:i+1].values.tolist() for i in range(0, len(data), 1)]
grouped_data = [sum(sublist, []) for sublist in grouped_data]

X_new = np.array(grouped_data)
#print(grouped_data)
prediction = knn.predict(X_new)
print("预测的目标类别是：{}".format(prediction))
file_path=r'D:\dataenclorse\second\test.csv'
import csv

def getdata(path):
    data_frame = pd.read_csv(r'D:\dataenclorse\second\iris_test.csv')  # skiprows=14
    data_x,data_y = np.array(data_frame['sepal length (cm)']), np.array(data_frame['sepal width (cm)'])
    return data_x,data_y
def getdata2(path):
    data_frame = pd.read_csv(r'D:\dataenclorse\second\iris_test.csv')  # skiprows=14
    data_p,data_q= np.array([data_frame['petal length (cm)'], np.array(data_frame['petal width (cm)'])])
    return  data_p,data_q

data_x,data_y=getdata('iris_test.csv')
data_p,data_q=getdata('iris_test.csv')
with open(file_path,'w',encoding='utf-8',newline='') as f:
        fieldnames=['sepal length(cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)','class']
        f_csv = csv.DictWriter(f, fieldnames=fieldnames)
        f_csv.writeheader()
        for i in range(0,len(prediction)):
            f_csv.writerow({'sepal length(cm)':data_x[i],'sepal width (cm)':data_y[i],'petal length (cm)':data_p[i],'petal width (cm)':data_q[i],'class':prediction[i]})


