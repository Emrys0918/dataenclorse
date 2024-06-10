import csv
import numpy as np
import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# ��������
train_data = scipy.io.loadmat(r'D:\dataenclorse\third\train_data.mat')
test_data = scipy.io.loadmat(r'D:\dataenclorse\third\test_data.mat')

# ��ȡѵ�����ݺͱ�ǩ
X_train = train_data['train'].reshape(-1, 28 * 28)  # ��ͼ������չƽΪһά����
y_train = np.repeat(np.arange(1, 201), 15)  # ���ɱ�ǩ

# ����ѵ��������֤��
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# ��ʼ����ѵ��SVMģ��
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# ����֤���Ͳ��Լ��Ͻ���Ԥ��
y_val_pred = svm_model.predict(X_val)
X_test = test_data['test'].reshape(-1, 28 * 28)  # ����������չƽΪһά����
y_test_pred = svm_model.predict(X_test)

# д��Ԥ������CSV�ļ�
output_file_path = r'D:\dataenclorse\third\submission.csv'
predictions = [{'ͼƬ': i+1, 'Ԥ����': pred} for i, pred in enumerate(y_test_pred)]

with open(output_file_path, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['ͼƬ', 'Ԥ����']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(predictions)

print("Ԥ�����ѱ�����:", output_file_path)
