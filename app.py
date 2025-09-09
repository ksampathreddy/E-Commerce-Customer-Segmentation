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

def load_models():
    """Load trained models and data"""
    global models, cluster_profiles
    
    try:
        models['cluster'] = pickle.load(open('model/cluster_model.pkl', 'rb'))
        models['scaler'] = pickle.load(open('model/scaler.pkl', 'rb'))
        models['features'] = pickle.load(open('model/feature_columns.pkl', 'rb'))
        
        cluster_profiles = pd.read_csv('model/cluster_profiles.csv')
        clustered_data = pd.read_csv('model/clustered_data.csv', index_col=0)
        
        # Load cluster descriptions
        try:
            with open('model/cluster_descriptions.json', 'r') as f:
                models['descriptions'] = json.load(f)
        except:
            models['descriptions'] = {}
        
        print("✅ Models loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

# Load models at startup
if not load_models():
    print("Warning: Could not load models. Training required.")

@app.route('/')
def home():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict cluster for input data"""
    try:
        if not models:
            return render_template('error.html', error="Models not loaded. Please train first.")
        
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
                             cluster_colors=Config.CLUSTER_COLORS)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/dashboard')
def dashboard():
    """Interactive dashboard"""
    try:
        # Load visualization components
        with open('templates/components/pca_plot.html', 'r') as f:
            pca_plot = f.read()
        
        with open('templates/components/spending_plot.html', 'r') as f:
            spending_plot = f.read()
        
        with open('templates/components/cluster_metrics.html', 'r') as f:
            metrics_plot = f.read()
        
        return render_template('dashboard.html',
                             pca_plot=pca_plot,
                             spending_plot=spending_plot,
                             metrics_plot=metrics_plot,
                             cluster_profiles=cluster_profiles.to_dict('records'),
                             cluster_colors=Config.CLUSTER_COLORS)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/clusters')
def clusters_overview():
    """Cluster overview page"""
    if cluster_profiles is not None:
        return render_template('clusters.html',
                             clusters=cluster_profiles.to_dict('records'),
                             cluster_colors=Config.CLUSTER_COLORS)
    else:
        return render_template('error.html', error="No cluster data available")

@app.route('/api/clusters')
def api_clusters():
    """API endpoint for cluster data"""
    try:
        clustered_data = pd.read_csv('model/clustered_data.csv')
        return jsonify(clustered_data.to_dict('records'))
    except:
        return jsonify({'error': 'No data available'}), 404

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API for predictions"""
    try:
        data = request.get_json()
        
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
    try:
        return send_file('model/clustered_data.csv',
                        as_attachment=True,
                        download_name='customer_segments.csv',
                        mimetype='text/csv')
    except:
        return render_template('error.html', error="Data not available for download")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)