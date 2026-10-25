# Practical 9: Density-Based Clustering using DBSCAN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"

columns = [
    "Area",
    "Perimeter",
    "Compactness",
    "Kernel_Length",
    "Kernel_Width",
    "Asymmetry",
    "Groove_Length",
    "Class"
]

df = pd.read_csv(
    url,
    sep=r"\s+",
    header=None,
    names=columns
)

print("Dataset loaded successfully!")
print("Shape of dataset:", df.shape)
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

X = df.drop("Class", axis=1)

print("\nFeatures used for DBSCAN:")
print(X.columns.tolist())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData standardized successfully.")

dbscan = DBSCAN(
    eps=0.8,
    min_samples=5
)

labels = dbscan.fit_predict(X_scaled)
df["Cluster"] = labels

print("\nDBSCAN clustering completed.")
print("Unique labels:", np.unique(labels))

cluster_counts = df["Cluster"].value_counts().sort_index()

print("\nNumber of points in each cluster:")
print(cluster_counts)

noise_count = int(np.sum(labels == -1))
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print("\nNumber of clusters:", n_clusters)
print("Number of noise points:", noise_count)

noise_points = df[df["Cluster"] == -1]
print("\nNoise Points:")
print(noise_points)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)
print("Total:", pca.explained_variance_ratio_.sum())

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("DBSCAN Clustering - Seeds Dataset")
plt.grid(True)
plt.show()

mask = labels != -1

if len(set(labels[mask])) >= 2:
    score = silhouette_score(X_scaled[mask], labels[mask])
    print("\nSilhouette Score (excluding noise):", round(score, 4))
else:
    print("\nSilhouette score cannot be calculated.")

print("\nClass vs DBSCAN Cluster:")
print(pd.crosstab(df["Class"], df["Cluster"]))
