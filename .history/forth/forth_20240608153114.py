import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ��������
train_data = pd.read_csv('D:\dataenclorse\forth\train.csv')
test_data = pd.read_csv('test.csv')
submission = pd.read_csv('submission.csv')

def preprocess_data(data):
    # ����ȱʧֵ
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)

    # ���Ա�͵Ǵ��ۿ�ת��Ϊ��ֵ
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    # ɾ������Ҫ����
    data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

    return data

# Ԥ��������
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# �����ͱ�ǩ
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# �ָ����ݼ�Ϊѵ��������֤��
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ����������������
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# ��֤ģ��
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.4f}')

# �Բ��Լ�����Ԥ��
test_pred = clf.predict(test_data)

# ��Ԥ��������submission.csv��
submission['Survived'] = test_pred

# ��������submission.csv
submission.to_csv('submission.csv', index=False)
