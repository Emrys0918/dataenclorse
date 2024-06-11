import csv
import numpy as np
import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split
from cvxopt import matrix, solvers

# 加载数据
train_data = scipy.io.loadmat(r'D:\dataenclorse\third\train_data.mat')
test_data = scipy.io.loadmat(r'D:\dataenclorse\third\test_data.mat')

# 获取训练数据和标签
X_train = train_data['train']
X_train = X_train.reshape(-1, 28 * 28)  # 将图像数据展平为一维数组
y_train = np.repeat(np.arange(1, 201), 15)  # 生成标签

# 对标签进行二分类处理，例如将标签1和其他标签分开
# 这里仅作为示例，实际上需要对所有标签进行多分类处理
y_train_binary = np.where(y_train == 1, 1, -1)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train_binary, test_size=0.2, random_state=42)

# 定义二次规划问题
def linear_svm_qp(X, y, C):
    m, n = X.shape
    K = np.dot(X, X.T)
    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones((m, 1)))
    G = matrix(np.vstack((-np.eye(m), np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = matrix(y, (1, m), 'd')
    b = matrix(0.0)

    # 解决QP问题
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])

    # 计算权重向量w
    w = ((y * alphas).T @ X).reshape(-1, 1)
    # 计算偏置b
    S = (alphas > 1e-4).flatten()
    b = np.mean(y[S] - np.dot(X[S], w))

    return w, b

# 训练线性SVM模型
C = 1.0
w, b = linear_svm_qp(X_train, y_train, C)

# 预测函数
def predict(X, w, b):
    return np.sign(X @ w + b).flatten()

# 在验证集上进行预测
y_val_pred = predict(X_val, w, b)

# 获取测试数据
X_test = test_data['test']
X_test = X_test.reshape(-1, 28 * 28)  # 将测试数据展平为一维数组

# 在测试集上进行预测
y_test_pred = predict(X_test, w, b)

# 将预测结果保存到CSV文件中
L = list(range(1, 1001))

file_path = r'D:\dataenclorse\third\submission.csv'
with open(file_path, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['图片', '预测结果']
    f_csv = csv.DictWriter(f, fieldnames=fieldnames)
    f_csv.writeheader()
    for i in range(0, len(y_test_pred)):
        f_csv.writerow({'图片': L[i], '预测结果': y_test_pred[i]})

print(f"预测结果已保存到 {file_path}")
