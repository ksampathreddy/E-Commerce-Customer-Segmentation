from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# -------------------------------------------------------------------------
# Load Model Artifacts
# -------------------------------------------------------------------------
MODEL_DIR = "model"

with open(os.path.join(MODEL_DIR, "kmeans_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "categories.pkl"), "rb") as f:
    product_categories = pickle.load(f)

# Create index map for faster lookup
category_index = {cat: i for i, cat in enumerate(product_categories)}

# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html", categories=product_categories)

@app.route("/predict", methods=["POST"])
def predict():
    # Initialize input vector with zeros
    input_data = np.zeros(len(product_categories))

    # Fill input vector from form data
    for category in product_categories:
        value = request.form.get(category, 0)
        try:
            input_data[category_index[category]] = float(value)
        except ValueError:
            input_data[category_index[category]] = 0.0

    # Scale and predict
    scaled_data = scaler.transform([input_data])
    cluster = model.predict(scaled_data)[0]

    return render_template(
        "index.html",
        prediction_text=f"Customer belongs to Cluster {cluster}",
        categories=product_categories
    )

# -------------------------------------------------------------------------
# Run App
# -------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
