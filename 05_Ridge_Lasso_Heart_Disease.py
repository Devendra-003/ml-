# Practical 5: Ridge and Lasso Regression - Heart Disease Dataset

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("heart.csv")

print("Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

numeric_df = df.select_dtypes(include=np.number).copy()

target_col = "target" if "target" in numeric_df.columns else "Target"
if target_col not in numeric_df.columns:
    raise ValueError("Target column not found. Rename the target column to 'target'.")

X = numeric_df.drop(target_col, axis=1)
y = numeric_df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ridge_pred = ridge.predict(X_test_scaled)

lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X_train_scaled, y_train)
lasso_pred = lasso.predict(X_test_scaled)

ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

print("\nRidge Regression")
print("MSE :", round(ridge_mse, 4))
print("R2  :", round(ridge_r2, 4))

print("\nLasso Regression")
print("MSE :", round(lasso_mse, 4))
print("R2  :", round(lasso_r2, 4))

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Ridge Coefficient": ridge.coef_,
    "Lasso Coefficient": lasso.coef_
})

print("\nCoefficient Comparison:")
print(coefficients)
