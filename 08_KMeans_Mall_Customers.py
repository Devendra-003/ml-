# Practical 8: K-Means Clustering - Mall Customers Dataset

import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv("Mall_Customers.csv")

data.columns = data.columns.str.strip()

print("First 5 rows of the dataset:")
print(data.head())

print("\nColumn names:")
print(data.columns.tolist())

required_columns = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = data[required_columns].dropna().values

print("\nSelected Features:")
print(X[:5])

wcss = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X)
    wcss.append(model.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method - Mall Customers")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

print("\nSilhouette Scores")
silhouette_scores = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print("K =", k, "Score =", round(score, 3))

plt.figure(figsize=(7, 5))
plt.plot(range(2, 11), silhouette_scores, marker="o")
plt.title("Silhouette Score - Mall Customers")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

data["Cluster"] = clusters

print("\nClustered Dataset:")
print(data.head(10))

print("\nCluster Counts:")
print(data["Cluster"].value_counts().sort_index())

print("\nCluster Centers:")
print(kmeans.cluster_centers_)
