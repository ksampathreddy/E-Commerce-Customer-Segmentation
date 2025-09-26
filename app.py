from flask import Flask, render_template, request, jsonify, send_file
import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# Global variables for loaded models
models = {}
cluster_profiles = None
models_loaded = False

def load_models():
    """Load trained models and data"""
    global models, cluster_profiles, models_loaded
    
    try:
        print("🔍 Attempting to load models...")
        
        # Check if model files exist
        required_files = [
            'model/cluster_model.pkl',
            'model/scaler.pkl', 
            'model/feature_columns.pkl',
            'model/cluster_profiles.csv'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            return False
        
        # Load models and data
        models['cluster'] = pickle.load(open('model/cluster_model.pkl', 'rb'))
        models['scaler'] = pickle.load(open('model/scaler.pkl', 'rb'))
        models['features'] = pickle.load(open('model/feature_columns.pkl', 'rb'))
        cluster_profiles = pd.read_csv('model/cluster_profiles.csv')
        
        models_loaded = True
        print("✅ Models loaded successfully!")
        print(f"📊 Available features: {models['features']}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Load models at startup
print("🚀 Starting Flask application...")
models_loaded = load_models()

@app.route('/')
def home():
    """Main page"""
    return render_template('index.html', models_loaded=models_loaded)

# FIX: Added GET method to display the form
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Predict cluster for input data - handles both GET (form) and POST (submission)"""
    if not models_loaded:
        return render_template('error.html', error="Models not loaded. Please run 'python train_models.py' first.")
    
    # Handle GET request - show the prediction form
    if request.method == 'GET':
        return render_template('predict.html', models_loaded=models_loaded)
    
    # Handle POST request - process the form data
    try:
        # Get form data with default values
        input_data = {
            'Total_Spend_Sum': float(request.form.get('Total_Spend_Sum', 0)),
            'Total_Spend_Mean': float(request.form.get('Total_Spend_Mean', 0)),
            'Purchase_Count': float(request.form.get('Purchase_Count', 0)),
            'Quantity_Sum': float(request.form.get('Quantity_Sum', 0)),
            'Price_Mean': float(request.form.get('Price_Mean', 0)),
            'Category_Diversity': float(request.form.get('Category_Diversity', 1))
        }
        
        print(f"📨 Received input data: {input_data}")
        
        # Create feature vector
        feature_vector = np.zeros(len(models['features']))
        for i, feature in enumerate(models['features']):
            feature_vector[i] = input_data.get(feature, 0)
        
        print(f"🔢 Feature vector: {feature_vector}")
        
        # Scale and predict
        scaled_data = models['scaler'].transform([feature_vector])
        cluster = int(models['cluster'].predict(scaled_data)[0])
        
        print(f"🎯 Predicted cluster: {cluster}")
        
        # Get cluster profile with SAFE field access
        profile_row = cluster_profiles[cluster_profiles['cluster'] == cluster].iloc[0]
        profile = profile_row.to_dict()
        
        # Use .get() with default values to prevent KeyError
        safe_profile = {
            'cluster': profile.get('cluster', cluster),
            'size': profile.get('size', 0),
            'size_percentage': profile.get('size_percentage', '0%'),
            'avg_total_spend': profile.get('avg_total_spend', 0),
            'avg_transaction_value': profile.get('avg_transaction_value', 0),
            'avg_purchase_count': profile.get('avg_purchase_count', 0),
            'avg_quantity': profile.get('avg_quantity', 0),
            'avg_price': profile.get('avg_price', 0),
            'avg_category_diversity': profile.get('avg_category_diversity', profile.get('Category_Diversity', 1)),
            'top_category': profile.get('top_category', 'Unknown'),
            'description': profile.get('description', 'No description available')
        }
        
        print(f"📊 Profile data: {safe_profile}")
        
        return render_template('results.html', 
                             cluster=cluster,
                             profile=safe_profile,
                             input_data=input_data,
                             models_loaded=models_loaded)
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template('error.html', error=f"Prediction error: {str(e)}")

@app.route('/dashboard')
def dashboard():
    """Interactive dashboard"""
    if not models_loaded:
        return render_template('error.html', error="Models not loaded. Please run 'python train_models.py' first.")
    
    try:
        # Prepare safe cluster profiles
        safe_profiles = []
        for profile in cluster_profiles.to_dict('records'):
            safe_profile = {
                'cluster': profile.get('cluster', 0),
                'size': profile.get('size', 0),
                'size_percentage': profile.get('size_percentage', '0%'),
                'avg_total_spend': profile.get('avg_total_spend', 0),
                'avg_purchase_count': profile.get('avg_purchase_count', 0),
                'avg_category_diversity': profile.get('avg_category_diversity', 1),
                'description': profile.get('description', 'No description available'),
                'top_category': profile.get('top_category', 'Unknown')
            }
            safe_profiles.append(safe_profile)
        
        return render_template('dashboard.html', 
                             cluster_profiles=safe_profiles,
                             models_loaded=models_loaded)
    
    except Exception as e:
        return render_template('error.html', error=f"Dashboard error: {str(e)}")

@app.route('/clusters')
def clusters_overview():
    """Cluster overview page"""
    if not models_loaded:
        return render_template('error.html', error="Models not loaded. Please run 'python train_models.py' first.")
    
    try:
        safe_profiles = []
        for profile in cluster_profiles.to_dict('records'):
            safe_profile = {
                'cluster': profile.get('cluster', 0),
                'size': profile.get('size', 0),
                'size_percentage': profile.get('size_percentage', '0%'),
                'avg_total_spend': profile.get('avg_total_spend', 0),
                'avg_purchase_count': profile.get('avg_purchase_count', 0),
                'avg_category_diversity': profile.get('avg_category_diversity', 1),
                'top_category': profile.get('top_category', 'Unknown'),
                'description': profile.get('description', 'No description available')
            }
            safe_profiles.append(safe_profile)
        
        return render_template('clusters.html', 
                             clusters=safe_profiles,
                             models_loaded=models_loaded)
    
    except Exception as e:
        return render_template('error.html', error=f"Clusters page error: {str(e)}")

@app.route('/download/cluster_data')
def download_cluster_data():
    """Download clustered data as CSV"""
    if not models_loaded:
        return render_template('error.html', error="Models not loaded.")
    
    try:
        return send_file('model/clustered_data.csv',
                        as_attachment=True,
                        download_name='customer_segments.csv')
    except Exception as e:
        return render_template('error.html', error=f"Download error: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API for predictions"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Create feature vector from JSON data
        feature_vector = np.zeros(len(models['features']))
        for i, feature in enumerate(models['features']):
            feature_vector[i] = data.get(feature, 0)
        
        scaled_data = models['scaler'].transform([feature_vector])
        cluster = int(models['cluster'].predict(scaled_data)[0])
        profile_row = cluster_profiles[cluster_profiles['cluster'] == cluster].iloc[0]
        profile = profile_row.to_dict()
        
        return jsonify({
            'cluster': cluster,
            'profile': profile,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("\n🌐 Flask Server Starting...")
    print("📊 Available Routes:")
    print("   GET  /                      - Home page")
    print("   GET  /predict               - Prediction form")
    print("   POST /predict               - Process prediction")
    print("   GET  /dashboard             - Analytics dashboard")
    print("   GET  /clusters              - Cluster overview")
    print("   GET  /download/cluster_data - Download data")
    print("   POST /api/predict           - JSON API")
    print("\n⚡ Starting server on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)