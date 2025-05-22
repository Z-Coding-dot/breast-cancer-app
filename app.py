from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert input data to numpy array
        features = np.array([[
            data['mean_radius'], data['mean_texture'], data['mean_perimeter'], data['mean_area'],
            data['mean_smoothness'], data['mean_compactness'], data['mean_concavity'],
            data['mean_concave_points'], data['mean_symmetry'], data['mean_fractal_dimension'],
            data['radius_error'], data['texture_error'], data['perimeter_error'], data['area_error'],
            data['smoothness_error'], data['compactness_error'], data['concavity_error'],
            data['concave_points_error'], data['symmetry_error'], data['fractal_dimension_error'],
            data['worst_radius'], data['worst_texture'], data['worst_perimeter'], data['worst_area'],
            data['worst_smoothness'], data['worst_compactness'], data['worst_concavity'],
            data['worst_concave_points'], data['worst_symmetry'], data['worst_fractal_dimension']
        ]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)
        
        return jsonify({
            "prediction": int(prediction[0]),
            "probability": float(probability[0][1]),
            "interpretation": "Malignant" if prediction[0] == 0 else "Benign"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    app.run(debug=True, host='127.0.0.1', port=8000)
