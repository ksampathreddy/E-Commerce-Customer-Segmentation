import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import os
import json
from datetime import datetime

class AmazonCustomerSegmentation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.category_names = []
        
    def load_and_preprocess(self, file_path):
        """Load and preprocess the Amazon dataset"""
        print("Loading dataset...")
        df = pd.read_csv("amazon.csv")
        print(f"Loaded {len(df)} records")
        
        # Data cleaning
        df = df.dropna(subset=['Brand Name', 'Category', 'Quantity', 'Selling Price'])
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(1)
        df['Selling Price'] = pd.to_numeric(df['Selling Price'], errors='coerce').fillna(0)
        df['Total_Spend'] = df['Quantity'] * df['Selling Price']
        
        # Extract primary category
        df['Primary_Category'] = df['Category'].str.split('|').str[0].str.strip().fillna('Unknown')
        
        return df
    
    def create_features(self, df):
        """Create customer-brand features matrix"""
        print("Creating feature matrix...")
        
        # RFM + Category features
        features = df.groupby('Brand Name').agg({
            'Total_Spend': ['sum', 'mean'],  # Monetary value
            'Quantity': ['sum', 'count'],    # Frequency
            'Primary_Category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
        }).reset_index()
        
        features.columns = ['Brand', 'Total_Spend', 'Avg_Spend', 'Total_Quantity', 'Purchase_Count', 'Top_Category']
        
        # Add category-wide features
        category_matrix = df.pivot_table(
            index='Brand Name',
            columns='Primary_Category',
            values='Total_Spend',
            aggfunc='sum',
            fill_value=0
        )
        
        # Merge with RFM features
        final_features = features.merge(category_matrix, left_on='Brand', right_index=True, how='left')
        final_features = final_features.fillna(0)
        
        return final_features
    
    def train_clusters(self, features, n_clusters=5):
        """Train clustering model"""
        print("Training clustering model...")
        
        # Select numerical features for clustering
        numerical_cols = [col for col in features.columns if col not in ['Brand', 'Top_Category']]
        X = features[numerical_cols].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine optimal clusters
        silhouette_scores = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
        
        # Plot elbow and silhouette
        self._plot_metrics(range(2, 11), silhouette_scores)
        
        # Train final model
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        features['Cluster'] = self.model.fit_predict(X_scaled)
        
        # Reduce dimensions for visualization
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(X_scaled)
        features['PC1'] = reduced_data[:, 0]
        features['PC2'] = reduced_data[:, 1]
        
        return features
    
    def _plot_metrics(self, k_values, silhouette_scores):
        """Plot evaluation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow method (using inertia)
        inertias = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaler.transform(self.features[[col for col in self.features.columns 
                                                         if col not in ['Brand', 'Top_Category']]].values))
            inertias.append(kmeans.inertia_)
        
        ax1.plot(k_values, inertias, 'bo-')
        ax1.set_xlabel('Number of clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True)
        
        # Silhouette scores
        ax2.plot(k_values, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)
        
        plt.savefig('model/cluster_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_clusters(self, clustered_data):
        """Analyze and describe clusters"""
        cluster_profiles = []
        
        for cluster in sorted(clustered_data['Cluster'].unique()):
            cluster_data = clustered_data[clustered_data['Cluster'] == cluster]
            
            profile = {
                'Cluster': cluster,
                'Size': len(cluster_data),
                'Avg_Spend': cluster_data['Total_Spend'].mean(),
                'Avg_Quantity': cluster_data['Total_Quantity'].mean(),
                'Top_Categories': cluster_data['Top_Category'].value_counts().head(3).to_dict(),
                'Description': self._generate_cluster_description(cluster_data)
            }
            cluster_profiles.append(profile)
        
        return pd.DataFrame(cluster_profiles)
    
    def _generate_cluster_description(self, cluster_data):
        """Generate human-readable cluster description"""
        avg_spend = cluster_data['Total_Spend'].mean()
        avg_quantity = cluster_data['Total_Quantity'].mean()
        top_category = cluster_data['Top_Category'].mode()[0]
        
        if avg_spend > 10000:
            spend_desc = "high-value"
        elif avg_spend > 5000:
            spend_desc = "medium-value"
        else:
            spend_desc = "low-value"
            
        return f"{spend_desc} customers primarily purchasing {top_category} products"
    
    def save_models(self, features):
        """Save all model artifacts"""
        os.makedirs('model', exist_ok=True)
        
        pickle.dump(self.model, open('model/cluster_model.pkl', 'wb'))
        pickle.dump(self.scaler, open('model/scaler.pkl', 'wb'))
        pickle.dump(features.columns.tolist(), open('model/feature_columns.pkl', 'wb'))
        
        # Save cluster profiles
        profiles = self.analyze_clusters(features)
        profiles.to_csv('model/cluster_profiles.csv', index=False)
        
        # Save visualization data
        self._create_visualizations(features)
    
    def _create_visualizations(self, features):
        """Create interactive visualizations"""
        # 2D Cluster plot
        fig = px.scatter(features, x='PC1', y='PC2', color='Cluster',
                        hover_data=['Brand', 'Total_Spend', 'Top_Category'],
                        title='Customer Segments Visualization')
        fig.write_html('model/cluster_visualization.html')
        
        # Cluster statistics
        cluster_stats = features.groupby('Cluster').agg({
            'Total_Spend': 'mean',
            'Total_Quantity': 'mean',
            'Purchase_Count': 'mean'
        }).reset_index()
        
        fig = px.bar(cluster_stats, x='Cluster', y='Total_Spend', 
                    title='Average Spending by Cluster')
        fig.write_html('model/spending_by_cluster.html')

def main():
    """Main execution function"""
    print("Starting Amazon Customer Segmentation Training...")
    
    # Initialize and train
    segmenter = AmazonCustomerSegmentation()
    
    # Load data
    df = segmenter.load_and_preprocess('data/your_amazon_data.csv')
    
    # Create features
    features = segmenter.create_features(df)
    segmenter.features = features  # Store for plotting
    
    # Train clusters
    clustered_data = segmenter.train_clusters(features, n_clusters=5)
    
    # Save results
    segmenter.save_models(clustered_data)
    
    print("\nTraining completed successfully!")
    print("Files saved in /model directory:")
    print("- cluster_model.pkl, scaler.pkl, feature_columns.pkl")
    print("- cluster_profiles.csv (cluster descriptions)")
    print("- cluster_metrics.png (evaluation plots)")
    print("- cluster_visualization.html (interactive plot)")

if __name__ == "__main__":
    main()