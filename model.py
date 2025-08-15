# Save necessary files
import pickle
pickle.dump(kmeans, open('kmeans_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(list(customer_product.columns), open('categories.pkl', 'wb'))
