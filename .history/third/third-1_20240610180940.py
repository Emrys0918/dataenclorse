import csv
import numpy as np
import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 加载数据
train_data = scipy.io.loadmat(r'D:\dataenclorse\third\train_data.mat')
test_data = scipy.io.loadmat(r'D:\dataenclorse\third\test_data.mat')

# 获取训练数据和标签
X_train = train_data['train'].reshape(-1, 28 * 28)  # 将图像数据展平为一维数组
y_train = np.repeat(np.arange(1, 201), 15)  # 生成标签

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 初始化并训练SVM模型
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# 在验证集和测试集上进行预测
y_val_pred = svm_model.predict(X_val)
X_test = test_data['test'].reshape(-1, 28 * 28)  # 将测试数据展平为一维数组
y_test_pred = svm_model.predict(X_test)

# 写入预测结果到CSV文件
output_file_path = r'D:\dataenclorse\third\submission.csv'
predictions = [{'图片': i+1, '预测结果': pred} for i, pred in enumerate(y_test_pred)]

with open(output_file_path, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['图片', '预测结果']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(predictions)

print("预测结果已保存至:", output_file_path)
