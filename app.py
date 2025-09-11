from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import json
import os

app = Flask(__name__)

# Load models
def load_models():
    models = {}
    try:
        models['cluster'] = pickle.load(open('model/cluster_model.pkl', 'rb'))
        models['scaler'] = pickle.load(open('model/scaler.pkl', 'rb'))
        models['features'] = pickle.load(open('model/feature_columns.pkl', 'rb'))
        models['profiles'] = pd.read_csv('model/cluster_profiles.csv')
        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

models = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        input_data = {
            'Total_Spend': float(request.form.get('total_spend', 0)),
            'Avg_Spend': float(request.form.get('avg_spend', 0)),
            'Total_Quantity': float(request.form.get('total_quantity', 0)),
            'Purchase_Count': float(request.form.get('purchase_count', 0)),
        }
        
        # Add category spending (simplified for demo)
        for category in ['Electronics', 'Books', 'Home', 'Clothing']:
            input_data[category] = float(request.form.get(category.lower(), 0))
        
        # Create feature vector
        feature_vector = np.zeros(len(models['features']))
        for i, feature in enumerate(models['features']):
            if feature in input_data:
                feature_vector[i] = input_data[feature]
        
        # Scale and predict
        scaled_data = models['scaler'].transform([feature_vector])
        cluster = models['cluster'].predict(scaled_data)[0]
        
        # Get cluster profile
        profile = models['profiles'][models['profiles']['Cluster'] == cluster].iloc[0]
        
        return render_template('results.html', 
                             cluster=cluster,
                             profile=profile.to_dict(),
                             input_data=input_data)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/dashboard')
def dashboard():
    """Interactive dashboard"""
    try:
        # Load visualization data
        with open('model/cluster_visualization.html', 'r') as f:
            cluster_plot = f.read()
        
        with open('model/spending_by_cluster.html', 'r') as f:
            spending_plot = f.read()
        
        return render_template('dashboard.html',
                             cluster_plot=cluster_plot,
                             spending_plot=spending_plot)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint"""
    try:
        data = request.get_json()
        # Similar prediction logic as above
        return jsonify({'cluster': int(cluster), 'profile': profile.to_dict()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)