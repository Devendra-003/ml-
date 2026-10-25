# Practical 4: Polynomial Regression - Insurance Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("insurance.csv")

print("Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

X = df[["bmi"]]
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

degrees = [1, 2, 3, 4, 5, 6]
results = []

plt.figure(figsize=(18, 10))
x_grid = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

for degree in degrees:
    poly = PolynomialFeatures(degree)
    X_poly_train = poly.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    y_pred = model.predict(poly.transform(X_test))

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append([degree, mse, rmse, r2])

    plt.subplot(2, 3, degree)
    plt.scatter(X, y, s=10)
    plt.plot(
        x_grid,
        model.predict(poly.transform(x_grid)),
        linewidth=2
    )
    plt.title(f"Degree {degree}")
    plt.xlabel("BMI")
    plt.ylabel("Charges")
    plt.grid(True)

plt.tight_layout()
plt.show()

results_df = pd.DataFrame(
    results,
    columns=["Degree", "MSE", "RMSE", "R2 Score"]
)

print("\nPolynomial Regression Results:")
print(results_df)

plt.figure(figsize=(8, 5))
plt.plot(results_df["Degree"], results_df["R2 Score"], marker="o")
plt.xlabel("Polynomial Degree")
plt.ylabel("R2 Score")
plt.title("Degree vs R2 Score")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(results_df["Degree"], results_df["RMSE"], marker="o")
plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")
plt.title("Degree vs RMSE")
plt.grid(True)
plt.show()
