import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Load your dataset
print("Loading dataset...")
df = pd.read_csv('amazon.csv')
print(f"Loaded {len(df)} records with columns: {df.columns.tolist()}")

# Data Preparation ------------------------------------------------------------

print("\nPreparing data...")

# 1. Clean and prepare the data
# Ensure we have required columns
required_cols = ['Brand Name', 'Category', 'Quantity']
if not all(col in df.columns for col in required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    raise ValueError(f"Missing required columns: {missing}")

print("\nChecking missing values before cleaning:")
print(df[['Brand Name', 'Category', 'Quantity']].isnull().sum())
print(df[['Brand Name', 'Category', 'Quantity']].head(20))

# Clean data
df = df.dropna(subset=['Brand Name', 'Category', 'Quantity'])
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(1)
df['Primary_Category'] = df['Category'].str.split('|').str[0].str.strip()


# 2. Create brand-category matrix
print("\nCreating brand-category matrix...")
category_matrix = df.pivot_table(
    index='Brand Name',
    columns='Primary_Category',
    values='Quantity',
    aggfunc='sum',
    fill_value=0
)

# Remove brands with no purchases
category_matrix = category_matrix.loc[category_matrix.sum(axis=1) > 0]

print(f"\nFinal matrix shape: {category_matrix.shape}")
print("Sample of the matrix:")
print(category_matrix.head())

if len(category_matrix) == 0:
    raise ValueError("No valid data remaining after preprocessing - check your input data")

# Clustering ----------------------------------------------------------------

print("\nPerforming clustering...")

# Convert to numpy array and scale
X = category_matrix.values.astype(float)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

# Elbow Method
print("\nDetermining optimal clusters...")
sse = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), sse, marker='o')
plt.title('Elbow Method for Optimal Cluster Number')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Squared Errors')
plt.grid()
plt.savefig('elbow_plot.png')
print("Saved elbow plot as 'elbow_plot.png'")

# Apply clustering
optimal_clusters = 5  # You can change this based on the elbow plot
print(f"\nClustering with {optimal_clusters} clusters...")
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)
category_matrix['Cluster'] = clusters

# Analysis -------------------------------------------------------------------

print("\nCluster Analysis:")
print("\nCluster Distribution:")
print(category_matrix['Cluster'].value_counts().sort_index())

print("\nTop Categories per Cluster:")
for cluster in range(optimal_clusters):
    cluster_data = category_matrix[category_matrix['Cluster'] == cluster].drop('Cluster', axis=1)
    top_categories = cluster_data.sum().sort_values(ascending=False).head(5)
    print(f"\nCluster {cluster} (Size: {len(cluster_data)} brands):")
    print(top_categories)

# Save model artifacts
print("\nSaving model artifacts...")
os.makedirs('model', exist_ok=True)

# Save the models
pickle.dump(kmeans, open('model/amazon_cluster.pkl', 'wb'))
pickle.dump(scaler, open('model/amazon_scaler.pkl', 'wb'))

# Save the category names (excluding the 'Cluster' column)
with open('model/amazon_categories.pkl', 'wb') as f:
    pickle.dump(category_matrix.columns.tolist()[:-1], f)

# Save the clustered data for reference
category_matrix.to_csv('model/clustered_brands.csv')

print("\nSuccessfully completed!")
print("Saved files in /model directory:")
print("- amazon_cluster.pkl (trained model)")
print("- amazon_scaler.pkl (normalization scaler)")
print("- amazon_categories.pkl (category list)")
print("- clustered_brands.csv (cluster assignments)")