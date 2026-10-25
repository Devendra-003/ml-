# Practical 10: Ensemble Learning - Random Forest vs Logistic Regression

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset loaded successfully")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

df = pd.DataFrame(X, columns=data.feature_names)
df["Target"] = y

print("\nFirst 5 rows:")
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train_scaled, y_train)

lr_prediction = lr.predict(X_test_scaled)

lr_accuracy = accuracy_score(y_test, lr_prediction)

print(
    "\nLogistic Regression Accuracy:",
    round(lr_accuracy * 100, 2),
    "%"
)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_prediction = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_prediction)

print(
    "Random Forest Accuracy:",
    round(rf_accuracy * 100, 2),
    "%"
)

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [lr_accuracy * 100, rf_accuracy * 100]
})

print("\nPerformance Comparison:")
print(results)

plt.figure(figsize=(7, 5))
plt.bar(results["Model"], results["Accuracy"])
plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("Comparison of Machine Learning Models")
plt.ylim(0, 100)
plt.show()

print("\nLogistic Regression:")
print(classification_report(y_test, lr_prediction))

print("\nRandom Forest:")
print(classification_report(y_test, rf_prediction))

print("\nFinal Result")
print("----------------------")
print(
    "Logistic Regression Accuracy:",
    round(lr_accuracy * 100, 2),
    "%"
)
print(
    "Random Forest Accuracy:",
    round(rf_accuracy * 100, 2),
    "%"
)
