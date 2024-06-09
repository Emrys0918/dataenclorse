import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
train_data = pd.read_csv('D:\dataenclorse\forth\train.csv')
test_data = pd.read_csv('test.csv')
submission = pd.read_csv('submission.csv')

def preprocess_data(data):
    # 处理缺失值
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)

    # 将性别和登船港口转换为数值
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    # 删除不必要的列
    data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

    return data

# 预处理数据
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# 特征和标签
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# 分割数据集为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树分类器
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 验证模型
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.4f}')

# 对测试集进行预测
test_pred = clf.predict(test_data)

# 将预测结果填入submission.csv中
submission['Survived'] = test_pred

# 保存结果到submission.csv
submission.to_csv('submission.csv', index=False)
