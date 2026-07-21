import os
import pickle
import json
import numpy as np
from flask import Flask, request, jsonify, render_template_string

# 1. INTERCEPT AND PATCH THE JSON ENGINE (Bypasses Keras internal validation bugs)
original_json_loads = json.loads

def patched_json_loads(s, *args, **kwargs):
    obj = original_json_loads(s, *args, **kwargs)
    if isinstance(obj, dict) and obj.get("class_name") == "Sequential":
        if "config" in obj and "layers" in obj["config"]:
            for layer in obj["config"]["layers"]:
                if "config" in layer and "quantization_config" in layer["config"]:
                    del layer["config"]["quantization_config"]
    return obj

json.loads = patched_json_loads

import keras

# Initialize Flask app
app = Flask(__name__)

# Base configuration
MODEL_PATH = "model.pkl"
model = None

def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
        print("Loading Bank Churn ANN model...")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("ANN Model loaded successfully!")

# Defining the standard 10 features for Bank Churn prediction
FEATURES_CONFIG = [
    {"id": "credit_score", "name": "Credit Score", "placeholder": "e.g., 650"},
    {"id": "age", "name": "Age", "placeholder": "e.g., 42"},
    {"id": "tenure", "name": "Tenure (Years with Bank)", "placeholder": "e.g., 5"},
    {"id": "balance", "name": "Account Balance ($)", "placeholder": "e.g., 50000"},
    {"id": "num_products", "name": "Number of Bank Products", "placeholder": "e.g., 2"},
    {"id": "has_credit_card", "name": "Has Credit Card? (1 = Yes, 0 = No)", "placeholder": "1 or 0"},
    {"id": "is_active_member", "name": "Is Active Member? (1 = Yes, 0 = No)", "placeholder": "1 or 0"},
    {"id": "estimated_salary", "name": "Estimated Salary ($)", "placeholder": "e.g., 85000"},
    {"id": "geography", "name": "Geography Code (e.g., Germany=1, France=0)", "placeholder": "1 or 0"},
    {"id": "gender", "name": "Gender Code (Male=1, Female=0)", "placeholder": "1 or 0"}
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bank Customer Churn Predictor</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .container { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); width: 100%; max-width: 600px; }
        h2 { text-align: center; color: #2c3e50; margin-bottom: 5px; }
        p.subtitle { text-align: center; color: #7f8c8d; margin-bottom: 25px; font-size: 14px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .input-group { display: flex; flex-direction: column; }
        label { font-size: 13px; font-weight: 600; color: #34495e; margin-bottom: 5px; }
        input { padding: 10px 12px; border: 1px solid #ccd1d9; border-radius: 6px; font-size: 14px; outline: none; transition: border 0.2s; }
        input:focus { border-color: #3498db; }
        button { grid-column: span 2; background-color: #2ecc71; color: white; border: none; padding: 14px; border-radius: 6px; font-size: 16px; font-weight: bold; cursor: pointer; margin-top: 15px; transition: background 0.2s; }
        button:hover { background-color: #27ae60; }
        #result { margin-top: 25px; padding: 15px; border-radius: 6px; text-align: center; font-size: 18px; font-weight: bold; display: none; }
        .churn { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .stay { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    </style>
</head>
<body>

<div class="container">
    <h2>Bank Customer Churn Inference</h2>
    <p class="subtitle">Enter customer details to check retention risk</p>
    <form id="predictionForm">
        <div class="grid">
            {% for feature in features_config %}
            <div class="input-group">
                <label for="{{ feature.id }}">{{ feature.name }}</label>
                <input type="number" id="{{ feature.id }}" step="any" required placeholder="{{ feature.placeholder }}">
            </div>
            {% endfor %}
            <button type="submit">Predict Status</button>
        </div>
    </form>
    <div id="result"></div>
</div>

<script>
    document.getElementById('predictionForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const resultDiv = document.getElementById('result');
        resultDiv.style.display = 'none';
        
        const payload = {};
        {% for feature in features_config %}
        payload["{{ feature.id }}"] = parseFloat(document.getElementById("{{ feature.id }}").value);
        {% endfor %}

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const data = await response.json();
            
            if (response.ok) {
                if (data.status === "Customer Will Leave (High Risk)") {
                    resultDiv.className = 'churn';
                } else {
                    resultDiv.className = 'stay';
                }
                resultDiv.innerHTML = `Result: ${data.status}<br><small style="font-weight:normal; font-size:13px;">Risk Probability Score: ${(data.probability * 100).toFixed(2)}%</small>`;
            } else {
                resultDiv.className = 'error';
                resultDiv.innerText = `Error: ${data.error}`;
            }
        } catch (err) {
            resultDiv.className = 'error';
            resultDiv.innerText = 'Error connecting to the system engine.';
        }
        resultDiv.style.display = 'block';
    });
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, features_config=FEATURES_CONFIG)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Missing input payload'}), 400
        
        feature_order = [
            'credit_score', 'age', 'tenure', 'balance', 'num_products', 
            'has_credit_card', 'is_active_member', 'estimated_salary', 'geography', 'gender'
        ]
        
        features = [data.get(key, 0.0) for key in feature_order]
        input_data = np.array([features], dtype=np.float32)
        
        prediction = model.predict(input_data)
        raw_val = float(prediction[0][0])
        
        if raw_val >= 0.5:
            categorical_output = "Customer Will Leave (High Risk)"
        else:
            categorical_output = "Customer Will Stay (Low Risk)"
            
        return jsonify({
            'status': categorical_output,
            'probability': raw_val
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=8080, debug=False)
