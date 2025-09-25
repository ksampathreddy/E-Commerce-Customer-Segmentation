from flask import Flask, render_template, request, jsonify, send_file
import pickle
import pandas as pd
import numpy as np
import json
import os
from config import Config
import plotly
import plotly.express as px
from datetime import datetime

app = Flask(__name__)
app.config.from_object(Config)

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
            'model/cluster_profiles.csv',
            'model/clustered_data.csv'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            print("💡 Please run 'python train_models.py' first to generate model files")
            return False
        
        # Load models
        models['cluster'] = pickle.load(open('model/cluster_model.pkl', 'rb'))
        models['scaler'] = pickle.load(open('model/scaler.pkl', 'rb'))
        models['features'] = pickle.load(open('model/feature_columns.pkl', 'rb'))
        
        # Load data
        cluster_profiles = pd.read_csv('model/cluster_profiles.csv')
        clustered_data = pd.read_csv('model/clustered_data.csv', index_col=0)
        
        # Load cluster descriptions (optional)
        try:
            with open('model/cluster_descriptions.json', 'r') as f:
                models['descriptions'] = json.load(f)
        except FileNotFoundError:
            models['descriptions'] = {}
            print("⚠️  Cluster descriptions file not found, continuing without it")
        
        models_loaded = True
        print("✅ Models loaded successfully!")
        print(f"   - Clusters: {len(cluster_profiles)}")
        print(f"   - Features: {len(models['features'])}")
        print(f"   - Data points: {len(clustered_data)}")
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

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Predict cluster for input data"""
    if not models_loaded:
        return render_template('error.html', 
                             error="Models not loaded. Please run 'python train_models.py' first to train the model.")
    
    if request.method == 'GET':
        # Show the prediction form
        return render_template('predict.html', models_loaded=models_loaded)
    
    try:
        # Get form data
        input_data = {}
        for feature in models['features']:
            if feature in request.form:
                input_data[feature] = float(request.form.get(feature, 0))
        
        # Create feature vector
        feature_vector = np.zeros(len(models['features']))
        for i, feature in enumerate(models['features']):
            feature_vector[i] = input_data.get(feature, 0)
        
        # Scale and predict
        scaled_data = models['scaler'].transform([feature_vector])
        cluster = int(models['cluster'].predict(scaled_data)[0])
        
        # Get cluster profile
        profile = cluster_profiles[cluster_profiles['cluster'] == cluster].iloc[0].to_dict()
        
        return render_template('results.html', 
                             cluster=cluster,
                             profile=profile,
                             input_data=input_data,
                             cluster_colors=Config.CLUSTER_COLORS,
                             models_loaded=models_loaded)
    
    except Exception as e:
        return render_template('error.html', error=f"Prediction error: {str(e)}")

@app.route('/dashboard')
def dashboard():
    """Interactive dashboard"""
    if not models_loaded:
        return render_template('error.html', 
                             error="Models not loaded. Please run 'python train_models.py' first.")
    
    try:
        # Check if visualization files exist
        viz_files = {
            'pca_plot': 'static/images/pca_plot.png',
            'cluster_sizes': 'static/images/cluster_sizes.png'
        }
        
        # Check which visualization files exist
        available_viz = {}
        for name, path in viz_files.items():
            available_viz[name] = os.path.exists(path)
        
        return render_template('dashboard.html',
                             cluster_profiles=cluster_profiles.to_dict('records'),
                             cluster_colors=Config.CLUSTER_COLORS,
                             available_viz=available_viz,
                             models_loaded=models_loaded)
    
    except Exception as e:
        return render_template('error.html', error=f"Dashboard error: {str(e)}")

@app.route('/clusters')
def clusters_overview():
    """Cluster overview page"""
    if not models_loaded:
        return render_template('error.html', 
                             error="Models not loaded. Please run 'python train_models.py' first.")
    
    try:
        return render_template('clusters.html',
                             clusters=cluster_profiles.to_dict('records'),
                             cluster_colors=Config.CLUSTER_COLORS,
                             models_loaded=models_loaded)
    except Exception as e:
        return render_template('error.html', error=f"Clusters page error: {str(e)}")

@app.route('/api/clusters')
def api_clusters():
    """API endpoint for cluster data"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 503
    
    try:
        clustered_data = pd.read_csv('model/clustered_data.csv')
        return jsonify(clustered_data.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 404

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
        profile = cluster_profiles[cluster_profiles['cluster'] == cluster].iloc[0].to_dict()
        
        return jsonify({
            'cluster': cluster,
            'profile': profile,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/download/cluster_data')
def download_cluster_data():
    """Download clustered data as CSV"""
    if not models_loaded:
        return render_template('error.html', error="Models not loaded")
    
    try:
        return send_file('model/clustered_data.csv',
                        as_attachment=True,
                        download_name='customer_segments.csv',
                        mimetype='text/csv')
    except Exception as e:
        return render_template('error.html', error=f"Download error: {str(e)}")

@app.route('/reload_models')
def reload_models():
    """Force reload models (for debugging)"""
    global models_loaded
    models_loaded = load_models()
    if models_loaded:
        return jsonify({'status': 'success', 'message': 'Models reloaded successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to reload models'})

if __name__ == '__main__':
    print("\n🌐 Starting Flask server...")
    print("📊 Available routes:")
    print("   - http://localhost:5000/ (Home)")
    print("   - http://localhost:5000/predict (Customer Prediction)")
    print("   - http://localhost:5000/dashboard (Analytics Dashboard)")
    print("   - http://localhost:5000/clusters (Cluster Overview)")
    print("   - http://localhost:5000/reload_models (Reload Models)")
    print("\n⚡ Server is running! Press Ctrl+C to stop.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)