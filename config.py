import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'amazon-customer-segmentation-secret-2024'
    MODEL_PATH = 'model/'
    DATA_PATH = 'data/'
    UPLOAD_FOLDER = 'uploads/'
    ALLOWED_EXTENSIONS = {'csv', 'json'}
    
    # Cluster colors for consistent visualization
    CLUSTER_COLORS = {
        0: '#FF6B6B',
        1: '#4ECDC4',
        2: '#45B7D1',
        3: '#F9A602',
        4: '#9B59B6',
        5: '#E74C3C',
        6: '#2ECC71',
        7: '#F39C12',
        8: '#3498DB',
        9: '#1ABC9C'
    }
    
    # Feature descriptions for UI
    FEATURE_DESCRIPTIONS = {
        'Total_Spend_Sum': 'Total amount spent by customer',
        'Total_Spend_Mean': 'Average transaction value',
        'Total_Spend_Std': 'Spending variability',
        'Total_Spend_Max': 'Maximum transaction amount',
        'Quantity_Sum': 'Total items purchased',
        'Quantity_Mean': 'Average items per transaction',
        'Purchase_Count': 'Number of purchases',
        'Price_Mean': 'Average item price',
        'Price_Max': 'Maximum item price',
        'Category_Diversity': 'Number of different categories purchased',
        'Spending_Concentration': 'Focus on top category (0-1)'
    }