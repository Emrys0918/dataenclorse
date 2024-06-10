import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ���غ�׼������
iris_data = pd.read_csv(r'D:\dataenclorse\second\iris_train.csv')
X = iris_data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].values
y = iris_data["species"].values

# �ָ����ݼ�Ϊѵ�����Ͳ��Լ�
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# ��ʼ����ѵ��KNN������
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# ���ز�������
test_data = pd.read_csv(r'D:\dataenclorse\second\iris_test.csv')
X_new = test_data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].values

# ����Ԥ��
predictions = knn.predict(X_new)

# ��Ԥ����д���µ�CSV�ļ�
output_file_path = r'D:\dataenclorse\second\test.csv'
test_data['class'] = predictions

# ָ��Ҫ�������˳��
output_columns = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "class"]
test_data.to_csv(output_file_path, index=False, columns=output_columns)

print("Ԥ���Ŀ������ǣ�{}".format(predictions))
