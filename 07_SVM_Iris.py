# Practical 7: Classification using SVM - Iris Dataset

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

iris = load_iris()

print("Dataset loaded successfully")
print("Number of samples:", iris.data.shape[0])
print("Number of features:", iris.data.shape[1])
print("Classes:", iris.target_names)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Target"] = iris.target
df["Species"] = df["Target"].map(dict(enumerate(iris.target_names)))

print("\nDataset Shape:", df.shape)

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear_svm = SVC(kernel="linear")
linear_svm.fit(X_train_scaled, y_train)
linear_pred = linear_svm.predict(X_test_scaled)

rbf_svm = SVC(kernel="rbf")
rbf_svm.fit(X_train_scaled, y_train)
rbf_pred = rbf_svm.predict(X_test_scaled)

def evaluate_model(name, y_true, y_pred):
    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)
    print("Accuracy :", round(accuracy_score(y_true, y_pred) * 100, 2), "%")
    print("Precision:", round(
        precision_score(y_true, y_pred, average="weighted") * 100, 2
    ), "%")
    print("Recall   :", round(
        recall_score(y_true, y_pred, average="weighted") * 100, 2
    ), "%")
    print("F1 Score :", round(
        f1_score(y_true, y_pred, average="weighted") * 100, 2
    ), "%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

evaluate_model("Linear SVM", y_test, linear_pred)
evaluate_model("RBF SVM", y_test, rbf_pred)
