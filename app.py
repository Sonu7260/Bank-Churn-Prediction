import os
import io
import pickle
import numpy as np
from flask import Flask, render_template_string, request
import keras

app = Flask(__name__)

MODEL_PATH = "Sequential_pkl.pkl"
model = None

try:
    # Open and unpickle the Keras Sequential object properly
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Sequential_pkl.pkl loaded successfully via Pickle!")
except Exception as e:
    print(f"Failed via pickle.load: {e}")
    # Fallback method if unpickling raw BytesIO requires custom keras unpickling
    try:
        model = keras.models.load_model(MODEL_PATH)
    except Exception as err:
        print(f"Failed via load_model: {err}")
        model = None

# Embedded HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bank Churn Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen flex flex-col justify-between font-sans">
    <header class="border-b border-slate-800 bg-slate-950/50 py-4 px-6">
        <div class="max-w-6xl mx-auto flex justify-between items-center">
            <h1 class="text-xl font-bold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
                ChurnGuard AI
            </h1>
            <span class="text-xs px-3 py-1 bg-slate-800 border border-slate-700 text-slate-400 rounded-full font-mono">
                Sequential_pkl.pkl
            </span>
        </div>
    </header>

    <main class="max-w-5xl mx-auto my-8 px-4 w-full">
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div class="lg:col-span-2 bg-slate-800/60 border border-slate-700/60 rounded-2xl p-6 shadow-xl">
                <h2 class="text-lg font-semibold text-slate-200 mb-1">Customer Attributes</h2>
                <p class="text-xs text-slate-400 mb-6">Enter details to run neural network churn prediction.</p>

                <form action="/predict" method="POST" class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Credit Score</label>
                        <input type="number" name="credit_score" min="300" max="850" value="{{ form_data.credit_score if form_data else 650 }}" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Age</label>
                        <input type="number" name="age" min="18" max="100" value="{{ form_data.age if form_data else 38 }}" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Tenure (Years)</label>
                        <input type="number" name="tenure" min="0" max="10" value="{{ form_data.tenure if form_data else 5 }}" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Account Balance ($)</label>
                        <input type="number" step="0.01" name="balance" value="{{ form_data.balance if form_data else 75000.00 }}" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Number of Products</label>
                        <select name="num_products" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100">
                            <option value="1">1 Product</option>
                            <option value="2" selected>2 Products</option>
                            <option value="3">3 Products</option>
                            <option value="4">4 Products</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Geography</label>
                        <select name="geography" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100">
                            <option value="France" selected>France</option>
                            <option value="Germany">Germany</option>
                            <option value="Spain">Spain</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Has Credit Card?</label>
                        <select name="has_credit_card" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100">
                            <option value="1" selected>Yes</option>
                            <option value="0">No</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Is Active Member?</label>
                        <select name="is_active_member" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100">
                            <option value="1" selected>Yes</option>
                            <option value="0">No</option>
                        </select>
                    </div>
                    <div class="sm:col-span-2">
                        <label class="block text-xs font-medium text-slate-300 mb-1">Estimated Annual Salary ($)</label>
                        <input type="number" step="0.01" name="estimated_salary" value="{{ form_data.estimated_salary if form_data else 50000.00 }}" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100">
                    </div>
                    <div class="sm:col-span-2 mt-2">
                        <button type="submit" class="w-full py-3 bg-indigo-600 hover:bg-indigo-500 font-semibold rounded-lg text-white transition-all">
                            Predict Churn
                        </button>
                    </div>
                </form>
            </div>

            <div class="bg-slate-800/60 border border-slate-700/60 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
                <div>
                    <h2 class="text-lg font-semibold text-slate-200 mb-4">Prediction Results</h2>
                    {% if prediction_text %}
                        <div class="text-center py-6 border border-slate-700/50 rounded-xl bg-slate-900/50">
                            {% if is_churn %}
                                <h3 class="text-2xl font-bold text-red-400">High Risk</h3>
                                <p class="text-slate-400 text-xs mt-1">Customer likely to exit</p>
                            {% else %}
                                <h3 class="text-2xl font-bold text-emerald-400">Low Risk</h3>
                                <p class="text-slate-400 text-xs mt-1">Customer likely to stay</p>
                            {% endif %}
                            <div class="mt-6 text-3xl font-extrabold text-slate-100">{{ prediction_text }}</div>
                        </div>
                    {% elif error_text %}
                        <div class="p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-xs text-center">
                            {{ error_text }}
                        </div>
                    {% else %}
                        <div class="text-center py-12 text-slate-500">
                            <p class="text-xs">Fill details and click <strong>Predict Churn</strong>.</p>
                        </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </main>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE, form_data=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error_text="Model file 'Sequential_pkl.pkl' failed to load from pickle.", form_data=request.form)

    try:
        credit_score = float(request.form['credit_score'])
        age = float(request.form['age'])
        tenure = float(request.form['tenure'])
        balance = float(request.form['balance'])
        num_products = float(request.form['num_products'])
        has_credit_card = float(request.form['has_credit_card'])
        is_active_member = float(request.form['is_active_member'])
        estimated_salary = float(request.form['estimated_salary'])
        
        geography = request.form['geography']
        geo_germany = 1.0 if geography == 'Germany' else 0.0
        geo_spain = 1.0 if geography == 'Spain' else 0.0

        # Construct 10 inputs for the Sequential model
        input_features = np.array([[
            credit_score, age, tenure, balance, num_products,
            has_credit_card, is_active_member, estimated_salary,
            geo_germany, geo_spain
        ]], dtype=np.float32)

        # Model Prediction
        preds = model.predict(input_features)
        prediction_prob = float(preds[0][0])
        churn_risk = prediction_prob > 0.5
        percentage = round(prediction_prob * 100, 2)

        return render_template_string(
            HTML_TEMPLATE,
            prediction_text=f"{percentage}%",
            is_churn=churn_risk,
            probability=percentage,
            form_data=request.form
        )

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error_text=f"Prediction error: {str(e)}", form_data=request.form)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
