import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
import os

# -------------------------------------------------------------------------
# Load Dataset
# -------------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv("amazon.csv")
print(f"Loaded {len(df)} records with columns: {df.columns.tolist()}")

# -------------------------------------------------------------------------
# Data Preparation
# -------------------------------------------------------------------------
print("\nPreparing data...")

# Clean data
df['Brand Name'] = df['Brand Name'].fillna(df['Product Name'])
df['Category'] = df['Category'].fillna("Misc")
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(1)

# Extract primary category
df['Primary_Category'] = df['Category'].str.split('|').str[0].str.strip()

# Create brand-category matrix
print("\nCreating brand-category matrix...")
category_matrix = df.pivot_table(
    index='Brand Name',
    columns='Primary_Category',
    values='Quantity',
    aggfunc='sum',
    fill_value=0
)

# Remove empty rows
category_matrix = category_matrix.loc[category_matrix.sum(axis=1) > 0]

if len(category_matrix) == 0:
    raise ValueError("No valid data remaining after preprocessing - check your input data")

print(f"\nFinal matrix shape: {category_matrix.shape}")
print(category_matrix.head())

# -------------------------------------------------------------------------
# Clustering
# -------------------------------------------------------------------------
print("\nPerforming clustering...")
X = category_matrix.values.astype(float)

# Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

# Elbow method (for visualization)
print("\nDetermining optimal clusters...")
sse = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_data)
    sse.append(km.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), sse, marker='o')
plt.title("Elbow Method for Optimal Cluster Number")
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squared Errors")
plt.grid()
plt.savefig("elbow_plot.png")
print("Saved elbow plot as 'elbow_plot.png'")

# Choose optimal clusters (adjust after checking elbow_plot.png)
optimal_clusters = 5
print(f"\nClustering with {optimal_clusters} clusters...")
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

category_matrix["Cluster"] = clusters

# -------------------------------------------------------------------------
# Cluster Analysis
# -------------------------------------------------------------------------
print("\nCluster Distribution:")
print(category_matrix["Cluster"].value_counts().sort_index())

print("\nTop Categories per Cluster:")
for cluster in range(optimal_clusters):
    cluster_data = category_matrix[category_matrix["Cluster"] == cluster].drop("Cluster", axis=1)
    top_categories = cluster_data.sum().sort_values(ascending=False).head(5)
    print(f"\nCluster {cluster} (Size: {len(cluster_data)} brands):")
    print(top_categories)

# -------------------------------------------------------------------------
# Save Model Artifacts
# -------------------------------------------------------------------------
print("\nSaving model artifacts...")
os.makedirs("model", exist_ok=True)

# Save trained model & scaler
with open("model/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save category names (without Cluster column)
categories = category_matrix.columns.tolist()[:-1]
with open("model/categories.pkl", "wb") as f:
    pickle.dump(categories, f)

# Save clustered data
category_matrix.to_csv("model/clustered_brands.csv")

print("\n✅ Training complete! Files saved in /model directory:")
print("- kmeans_model.pkl (trained model)")
print("- scaler.pkl (fitted scaler)")
print("- categories.pkl (category list)")
print("- clustered_brands.csv (cluster assignments)")
