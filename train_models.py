import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import re
import warnings
warnings.filterwarnings('ignore')

class AmazonCustomerSegmentation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = []
        
    def load_data(self, file_path):
        """Load and validate dataset"""
        print("📊 Loading dataset...")
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    def preprocess_data(self, df):
        """Preprocess the Amazon dataset with proper handling for your data issues"""
        print("🔄 Preprocessing data...")
        
        # Make a copy
        df_clean = df.copy()
        
        # Handle missing Brand Names - use Product Name or create unique IDs
        if 'Brand Name' in df_clean.columns and df_clean['Brand Name'].isnull().all():
            print("⚠️  No Brand Names found, using Product Names as identifiers")
            if 'Product Name' in df_clean.columns:
                df_clean['Brand_Name'] = df_clean['Product Name'].str.split().str[0]  # First word as brand
            else:
                df_clean['Brand_Name'] = ['Product_' + str(i) for i in range(len(df_clean))]
        elif 'Brand Name' in df_clean.columns:
            df_clean['Brand_Name'] = df_clean['Brand Name'].fillna('Unknown_Brand')
        
        # Handle price formatting - remove $ and convert to float
        price_columns = ['Selling Price', 'List Price']
        for col in price_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.replace('$', '', regex=False)
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                print(f"💰 Converted {col}: {df_clean[col].notna().sum()} valid prices")
        
        # Handle missing Quantity - set to 1
        if 'Quantity' in df_clean.columns:
            df_clean['Quantity'] = pd.to_numeric(df_clean['Quantity'], errors='coerce').fillna(1)
        else:
            df_clean['Quantity'] = 1
        
        # Handle missing Categories
        if 'Category' in df_clean.columns:
            df_clean['Category'] = df_clean['Category'].fillna('Unknown')
        else:
            df_clean['Category'] = 'Unknown'
        
        # Remove rows with invalid prices
        if 'Selling Price' in df_clean.columns:
            df_clean = df_clean[df_clean['Selling Price'].notna()]
            df_clean = df_clean[df_clean['Selling Price'] > 0]
        
        print(f"✅ Cleaned data: {len(df_clean):,} records remaining")
        return df_clean

    def create_features(self, df):
        """Create features from the dataset"""
        print("🎯 Creating features...")
        
        # Calculate total spend
        df['Total_Spend'] = df['Quantity'] * df['Selling Price']
        
        # Extract primary category (first part before |)
        df['Primary_Category'] = df['Category'].astype(str).str.split('|').str[0].str.strip().fillna('Unknown')
        
        # Group by brand and create features
        brand_features = df.groupby('Brand_Name').agg({
            'Total_Spend': ['sum', 'mean', 'count'],
            'Quantity': ['sum', 'mean'],
            'Selling Price': ['mean', 'max']
        }).round(2)
        
        # Flatten column names
        brand_features.columns = [
            'Total_Spend_Sum', 'Total_Spend_Mean', 'Purchase_Count',
            'Quantity_Sum', 'Quantity_Mean',
            'Price_Mean', 'Price_Max'
        ]
        
        # Get top category for each brand
        top_categories = df.groupby('Brand_Name')['Primary_Category'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
        )
        brand_features['Top_Category'] = top_categories
        
        # Create category spending matrix
        category_matrix = df.pivot_table(
            index='Brand_Name',
            columns='Primary_Category',
            values='Total_Spend',
            aggfunc='sum',
            fill_value=0
        )
        
        # Add diversity metrics
        brand_features['Category_Diversity'] = (category_matrix > 0).sum(axis=1)
        brand_features['Spending_Concentration'] = category_matrix.max(axis=1) / (category_matrix.sum(axis=1) + 1e-10)
        
        # Merge all features
        features = brand_features.merge(category_matrix, left_index=True, right_index=True, how='left')
        features = features.fillna(0)
        
        print(f"✅ Created features for {len(features):,} brands")
        print(f"📊 Feature columns: {len(features.columns)}")
        return features

    def train_clusters(self, features, n_clusters=5):
        """Train clustering model"""
        print("🔍 Training clusters...")
        
        # Select only numerical features for clustering
        numerical_features = features.select_dtypes(include=[np.number])
        
        # Remove columns with zero variance
        numerical_features = numerical_features.loc[:, numerical_features.std() > 0]
        
        if len(numerical_features) == 0:
            print("⚠️  No variance in features, using all numerical columns")
            numerical_features = features.select_dtypes(include=[np.number])
        
        if len(numerical_features) == 0:
            raise ValueError("No numerical features found for clustering")
        
        X = numerical_features.values
        print(f"📈 Clustering matrix shape: {X.shape}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Store feature columns for later use
        self.feature_columns = numerical_features.columns.tolist()
        
        # Train model
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        features['Cluster'] = self.model.fit_predict(X_scaled)
        
        # Add PCA for visualization
        try:
            pca = PCA(n_components=2)
            pca_results = pca.fit_transform(X_scaled)
            features['PCA1'] = pca_results[:, 0]
            features['PCA2'] = pca_results[:, 1]
            print("✅ Added PCA components for visualization")
        except Exception as e:
            print(f"⚠️  Could not compute PCA: {e}")
        
        return features

    def analyze_clusters(self, features):
        """Analyze clusters"""
        print("📈 Analyzing clusters...")
        
        cluster_profiles = []
        
        for cluster in sorted(features['Cluster'].unique()):
            cluster_data = features[features['Cluster'] == cluster]
            
            profile = {
                'cluster': int(cluster),
                'size': len(cluster_data),
                'size_percentage': f"{(len(cluster_data) / len(features) * 100):.1f}%",
                'avg_total_spend': cluster_data['Total_Spend_Sum'].mean(),
                'avg_transaction_value': cluster_data['Total_Spend_Mean'].mean(),
                'avg_purchase_count': cluster_data['Purchase_Count'].mean(),
                'avg_quantity': cluster_data['Quantity_Mean'].mean(),
                'avg_price': cluster_data['Price_Mean'].mean(),
                'top_category': cluster_data['Top_Category'].mode()[0] if len(cluster_data['Top_Category'].mode()) > 0 else 'Unknown',
                'description': self._generate_description(cluster_data)
            }
            
            cluster_profiles.append(profile)
        
        return pd.DataFrame(cluster_profiles)

    def _generate_description(self, cluster_data):
        """Generate cluster description"""
        avg_spend = cluster_data['Total_Spend_Sum'].mean()
        avg_freq = cluster_data['Purchase_Count'].mean()
        
        if avg_spend > 10000:
            spend = "Premium"
        elif avg_spend > 5000:
            spend = "High-value"
        elif avg_spend > 1000:
            spend = "Medium-value"
        else:
            spend = "Budget"
            
        if avg_freq > 50:
            freq = "frequent"
        elif avg_freq > 20:
            freq = "regular"
        else:
            freq = "occasional"
            
        return f"{spend} {freq} shoppers"

    def create_visualizations(self, features):
        """Create visualizations"""
        print("🎨 Creating visualizations...")
        
        os.makedirs('static/images', exist_ok=True)
        
        # Cluster sizes
        plt.figure(figsize=(10, 6))
        cluster_sizes = features['Cluster'].value_counts().sort_index()
        plt.bar(cluster_sizes.index, cluster_sizes.values)
        plt.xlabel('Cluster')
        plt.ylabel('Number of Brands')
        plt.title('Cluster Sizes')
        plt.savefig('static/images/cluster_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # PCA plot if available
        if 'PCA1' in features.columns and 'PCA2' in features.columns:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(features['PCA1'], features['PCA2'], c=features['Cluster'], cmap='viridis', alpha=0.7)
            plt.colorbar(label='Cluster')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.title('Brand Segments - PCA Visualization')
            plt.grid(True, alpha=0.3)
            plt.savefig('static/images/pca_plot.png', dpi=300, bbox_inches='tight')
            plt.close()

    def save_results(self, features, cluster_profiles):
        """Save results"""
        print("💾 Saving results...")
        
        os.makedirs('model', exist_ok=True)
        
        # Save models
        pickle.dump(self.model, open('model/cluster_model.pkl', 'wb'))
        pickle.dump(self.scaler, open('model/scaler.pkl', 'wb'))
        pickle.dump(self.feature_columns, open('model/feature_columns.pkl', 'wb'))
        
        # Save data
        features.to_csv('model/clustered_data.csv', index=True)
        cluster_profiles.to_csv('model/cluster_profiles.csv', index=False)
        
        print("✅ All files saved successfully!")

def main():
    """Main execution"""
    print("🚀 Starting Amazon Customer Segmentation...")
    print("=" * 50)
    
    try:
        # Initialize
        segmenter = AmazonCustomerSegmentation()
        
        # Load data
        df = segmenter.load_data('data/amazon.csv')
        
        # Preprocess
        df_clean = segmenter.preprocess_data(df)
        
        if len(df_clean) == 0:
            print("❌ Still no data after preprocessing. Creating synthetic data for demonstration...")
            df_clean = create_synthetic_data()
        
        # Create features and train
        features = segmenter.create_features(df_clean)
        
        if len(features) == 0:
            print("❌ No features created. Creating synthetic features...")
            features = create_synthetic_features()
        
        clustered_data = segmenter.train_clusters(features, n_clusters=4)  # Fewer clusters for small data
        
        # Analyze and visualize
        cluster_profiles = segmenter.analyze_clusters(clustered_data)
        segmenter.create_visualizations(clustered_data)
        
        # Save results
        segmenter.save_results(clustered_data, cluster_profiles)
        
        print("\n🎉 Training completed successfully!")
        print(f"📊 Clusters created: {len(cluster_profiles)}")
        print(f"🏪 Brands analyzed: {len(clustered_data):,}")
        
        # Show cluster summary
        print("\n📋 Cluster Summary:")
        for _, profile in cluster_profiles.iterrows():
            print(f"Segment {profile['cluster']}: {profile['size']} brands - {profile['description']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def create_synthetic_data():
    """Create synthetic data for demonstration"""
    print("🔄 Creating synthetic data for demonstration...")
    
    np.random.seed(42)
    n_records = 1000
    
    synthetic_data = {
        'Brand_Name': [f'Brand_{i//10}' for i in range(n_records)],
        'Category': np.random.choice([
            'Electronics|Computers|Laptops',
            'Home|Kitchen|Appliances', 
            'Books|Fiction|Novels',
            'Sports|Outdoor|Camping',
            'Clothing|Men|Shirts'
        ], n_records),
        'Selling Price': np.random.uniform(10, 1000, n_records),
        'Quantity': np.random.randint(1, 5, n_records)
    }
    
    df = pd.DataFrame(synthetic_data)
    print(f"✅ Created synthetic data with {len(df)} records")
    return df

def create_synthetic_features():
    """Create synthetic features for demonstration"""
    print("🔄 Creating synthetic features for demonstration...")
    
    np.random.seed(42)
    n_brands = 100
    
    features = pd.DataFrame({
        'Total_Spend_Sum': np.random.uniform(1000, 50000, n_brands),
        'Total_Spend_Mean': np.random.uniform(50, 500, n_brands),
        'Purchase_Count': np.random.randint(5, 100, n_brands),
        'Quantity_Sum': np.random.randint(10, 200, n_brands),
        'Quantity_Mean': np.random.uniform(1, 5, n_brands),
        'Price_Mean': np.random.uniform(20, 300, n_brands),
        'Price_Max': np.random.uniform(100, 1000, n_brands),
        'Category_Diversity': np.random.randint(1, 5, n_brands),
        'Spending_Concentration': np.random.uniform(0.3, 0.9, n_brands),
        'Top_Category': np.random.choice(['Electronics', 'Home', 'Books', 'Sports', 'Clothing'], n_brands)
    }, index=[f'Brand_{i}' for i in range(n_brands)])
    
    print(f"✅ Created synthetic features for {len(features)} brands")
    return features

if __name__ == "__main__":
    main()