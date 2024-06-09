import csv
import numpy as np
import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.svm
import SVC

# 加载数据
train_data = scipy.io.loadmat(r'D:\dataenclorse\third\train_data.mat')
test_data = scipy.io.loadmat(r'D:\dataenclorse\third\test_data.mat')

# 获取训练数据和标签
X_train = train_data['train']
X_train = X_train.reshape(-1, 28 * 28)  # 将图像数据展平为一维数组
y_train = np.repeat(np.arange(1, 201), 15)  # 生成标签

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 初始化SVM模型
svm_model = SVC(kernel='linear', C=1.0, random_state=42)

# 训练SVM模型
svm_model.fit(X_train, y_train)

# 在验证集上进行预测
y_val_pred = svm_model.predict(X_val)

# 在测试集上进行预测
X_test = test_data['test']
X_test = X_test.reshape(-1, 28 * 28)  # 将测试数据展平为一维数组
y_test_pred = svm_model.predict(X_test)

L = list(range(1, 1001))

file_path = r'D:\dataenclorse\third\submission.csv'
with open(file_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['图片', '预测结果']
        f_csv = csv.DictWriter(f, fieldnames=fieldnames)
        f_csv.writeheader()
        for i in range(0, len(y_test_pred)):
            f_csv.writerow({'图片':L[i], '预测结果':y_test_pred[i]})
pass
