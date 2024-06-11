import pandas as pd
import numpy as np
from collections import defaultdict
from math import sqrt, pi, exp

# 读取数据集
iris_data = pd.read_csv(r'D:\dataenclorse\second\iris_train.csv')
X = iris_data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].values
y = iris_data["species"].values

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 计算每个类别的先验概率
def calculate_prior(y):
    class_counts = defaultdict(int)
    for label in y:
        class_counts[label] += 1
    total_count = len(y)
    priors = {label: count / total_count for label, count in class_counts.items()}
    return priors

# 计算每个类别和每个特征的均值和标准差
def calculate_mean_std(X, y):
    separated = defaultdict(list)
    for i in range(len(y)):
        separated[y[i]].append(X[i])
    summary = {}
    for label, instances in separated.items():
        summary[label] = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*instances)]
    return summary

# 计算高斯分布的概率密度函数
def gaussian_probability(x, mean, std):
    exponent = exp(-((x - mean) ** 2 / (2 * std ** 2)))
    return (1 / (sqrt(2 * pi) * std)) * exponent

# 计算给定特征的似然
def calculate_likelihood(summary, x):
    likelihoods = {}
    for label, stats in summary.items():
        likelihood = 1
        for i in range(len(stats)):
            mean, std = stats[i]
            likelihood *= gaussian_probability(x[i], mean, std)
        likelihoods[label] = likelihood
    return likelihoods

# 使用贝叶斯定理计算后验概率
def calculate_posterior(priors, likelihoods):
    posteriors = {}
    for label in priors:
        posteriors[label] = priors[label] * likelihoods[label]
    total_posterior = sum(posteriors.values())
    for label in posteriors:
        posteriors[label] /= total_posterior
    return posteriors

# 手动实现的朴素贝叶斯分类器预测函数
def naive_bayes_predict(X_train, y_train, X_new):
    priors = calculate_prior(y_train)
    summary = calculate_mean_std(X_train, y_train)
    predictions = []
    for x in X_new:
        likelihoods = calculate_likelihood(summary, x)
        posteriors = calculate_posterior(priors, likelihoods)
        best_label = max(posteriors, key=posteriors.get)
        predictions.append(best_label)
    return predictions

# 读取待预测的新数据点
iris_test_data = pd.read_csv(r'D:\dataenclorse\second\iris_test.csv')
X_new = iris_test_data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].values

# 预测新数据点的类别
predictions = naive_bayes_predict(X_train, y_train, X_new)

# 将预测结果保存到新的CSV文件中
file_path = r'D:\dataenclorse\second\test_manual_bayes.csv'
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
