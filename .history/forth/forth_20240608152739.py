import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
# 1.��ȡ����
titan =pd.read_csv("D:\dataenclorse\forth\train.csv")
x = titan[["pclass", "age", "sex"]]
y = titan["survived"]
# 2.2 ȱʧֵ����
x['age'].fillna(value=titan["age"].mean(), inplace=True)
# 2.3 ���ݼ�����
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22, test_size=0.2)
# 3.��������(�ֵ�������ȡ)
x_train = x_train.to_dict(orient="records")
x_test = x_test.to_dict(orient="records")
transfer = DictVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)
# 4.����ѧϰ(������)
estimator = DecisionTreeClassifier(max_depth=15)
estimator.fit(x_train, y_train)
# 5.ģ������
y_pre = estimator.predict(x_test)
ret = estimator.score(x_test, y_test)
print(ret)
export_graphviz(estimator, out_file="./data/tree.dot", feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'Ů��', '����'])
