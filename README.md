## This project is an advanced customer segmentation system for e-commerce businesses, specifically designed for Amazon product data. It uses machine learning clustering algorithms to group customers based on their purchasing behavior, enabling targeted marketing strategies and personalized customer experiences.

# Features
Advanced Customer Segmentation: Group customers based on RFM (Recency, Frequency, Monetary) analysis and purchasing patterns

Interactive Web Interface: User-friendly Flask web application for data analysis and visualization

Machine Learning Models: K-Means clustering algorithm for identifying customer segments

Comprehensive Analytics: Detailed cluster analysis with visualizations and insights

REST API: JSON API endpoints for integration with other systems

Data Export: Download clustered data for further analysis

# Technologies Used
Backend: Python, Flask, Scikit-learn, Pandas, NumPy

Frontend: HTML5, CSS3, JavaScript, Jinja2 templates

Visualization: Matplotlib, Seaborn, Plotly

Machine Learning: K-Means clustering, PCA for dimensionality reduction

Data Processing: Pandas for data manipulation and feature engineering

# Installation
Clone the repository

``bash
git clone <repository-url>
cd E-Commerce-Customer-Segmentation
``
Create a virtual environment

``bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
``
Install dependencies

``bash
pip install -r requirements.txt
``
Prepare your data

Place your Amazon dataset in the data/ directory

Run thw web app
``bash
python app.py
``
