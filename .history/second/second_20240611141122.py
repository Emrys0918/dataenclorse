import pandas as pd
import numpy as np
from collections import Counter

# Load dataset
iris_data = pd.read_csv(r'D:\dataenclorse\second\iris_train.csv')
X = iris_data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].values
y = iris_data["species"].values

# Split the dataset (you already have this step, just repeating for context)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Euclidean distance function
def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2) ** 2))

# Manual KNN function
def knn_predict(X_train, y_train, X_new, k=1):
    predictions = []
    for x_new in X_new:
        # Calculate distances from the new point to all training points
        distances = [euclidean_distance(x_new, x_train) for x_train in X_train]
        # Get indices of the sorted distances
        k_indices = np.argsort(distances)[:k]
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [y_train[i] for i in k_indices]
        # Majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])
    return predictions

# Load the new data points for prediction
iris_test_data = pd.read_csv(r'D:\dataenclorse\second\iris_test.csv')
X_new = iris_test_data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].values

# Predict the class labels for the new data points
predictions = knn_predict(X_train, y_train, X_new, k=1)

# Save the predictions to a new CSV file
file_path = r'D:\dataenclorse\second\test_manual_knn.csv'
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

print(f"Predictions saved to {file_path}")
