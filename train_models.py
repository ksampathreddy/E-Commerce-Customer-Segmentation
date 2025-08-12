import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (adjust path as needed)
df = pd.read_csv('amazon_products.csv')# Basic cleaning
df = df.dropna(subset=['customer_id', 'product_category'])
df = df[df['customer_id'] != 'unknown']

# Create customer-product matrix
customer_product = pd.crosstab(
    index=df['customer_id'],
    columns=df['product_category'],
    values=df['price'],
    aggfunc='count',
    fill_value=0
)

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_product)# Reduce dimensions for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Find optimal clusters (Elbow Method)
sse = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

plt.plot(range(2, 11), sse)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Final clustering (let's assume 5 clusters)
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add clusters to reduced data
reduced_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
reduced_df['Cluster'] = clusters
reduced_df['CustomerID'] = customer_product.index# Visualize clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PC1', y='PC2',
    hue='Cluster',
    palette=sns.color_palette('hls', n_clusters),
    data=reduced_df,
    legend='full'
)
plt.title('Customer Segments Based on Amazon Purchases')
plt.show()

# Analyze top products per cluster
cluster_products = pd.DataFrame(scaled_data, columns=customer_product.columns)
cluster_products['Cluster'] = clusters

top_products_per_cluster = {}
for i in range(n_clusters):
    top_products = cluster_products[cluster_products['Cluster'] == i].mean().sort_values(ascending=False)[:5]
    top_products_per_cluster[f'Cluster {i}'] = top_products.index.tolist()

pd.DataFrame(top_products_per_cluster)