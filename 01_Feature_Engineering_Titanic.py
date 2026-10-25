# Practical 1: Feature Engineering - Titanic Dataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Titanic.csv")

print("Dataset Shape:", df.shape)
print(df.head())
print("\nMissing Values:\n", df.isnull().sum())

if "Age" in df.columns:
    df["Age"] = df["Age"].fillna(df["Age"].median())

if "Fare" in df.columns:
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

for col in ["Embarked", "Sex"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

if "SibSp" in df.columns and "Parch" in df.columns:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

if "Age" in df.columns:
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, np.inf],
        labels=["Child", "Teen", "Adult", "Middle_Age", "Senior"]
    )

if "Sex" in df.columns:
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

if "Embarked" in df.columns:
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

print("\nAfter Feature Engineering:")
print(df.head())
print("\nRemaining Missing Values:\n", df.isnull().sum())
