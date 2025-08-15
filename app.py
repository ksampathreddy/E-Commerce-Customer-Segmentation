from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
product_categories = pickle.load(open('categories.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', categories=product_categories)

@app.route('/predict', methods=['POST'])
def predict():
    # Create input vector
    input_data = np.zeros(len(product_categories))
    for category in product_categories:
        input_data[product_categories.index(category)] = float(request.form.get(category, 0))
    
    # Scale and predict
    scaled_data = scaler.transform([input_data])
    cluster = model.predict(scaled_data)[0]
    
    return render_template('index.html', 
                         prediction_text=f'Customer belongs to Cluster {cluster}',
                         categories=product_categories)

if __name__ == "__main__":
    app.run(debug=True)