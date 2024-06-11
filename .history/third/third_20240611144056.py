import numpy as np
import scipy.io
import csv

# 加载数据
train_data = scipy.io.loadmat(r'D:\dataenclorse\third\train_data.mat')
test_data = scipy.io.loadmat(r'D:\dataenclorse\third\test_data.mat')

# 获取训练数据和标签
X_train = train_data['train'].reshape(-1, 28 * 28)  # 将图像数据展平为一维数组
y_train = np.repeat(np.arange(1, 201), 15)  # 生成标签

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_size = int(X.shape[0] * test_size)

    X_train = X[indices[:-test_size]]
    y_train = y[indices[:-test_size]]
    X_val = X[indices[-test_size:]]
    y_val = y[indices[-test_size:]]

    return X_train, X_val, y_train, y_val

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
