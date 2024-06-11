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
class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iters = num_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        y_ = np.where(y <= 100, 1, -1)  # 二分类标签转换

        for _ in range(self.num_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
# 初始化并训练SVM模型
svm_model = LinearSVM(learning_rate=0.0001, lambda_param=0.01, num_iters=1000)
svm_model.fit(X_train, y_train)

# 在验证集和测试集上进行预测
y_val_pred = svm_model.predict(X_val)
X_test = test_data['test'].reshape(-1, 28 * 28)  # 将测试数据展平为一维数组
y_test_pred = svm_model.predict(X_test)
