import os
import numpy as np
from flask import Flask, render_template_string, request
import keras

app = Flask(__name__)

# Load the saved model file
MODEL_PATH = "Sequential_pkl.pkl"

try:
    model = keras.models.load_model(MODEL_PATH)
    print("Sequential_pkl.pkl loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Embedded HTML & CSS Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bank Customer Churn Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen flex flex-col justify-between font-sans">

    <!-- Navigation Header -->
    <header class="border-b border-slate-800 bg-slate-950/50 backdrop-blur py-4 px-6">
        <div class="max-w-6xl mx-auto flex justify-between items-center">
            <div class="flex items-center space-x-3">
                <div class="p-2 bg-indigo-600 rounded-lg">
                    <i class="fa-solid fa-brain text-white text-xl"></i>
                </div>
                <h1 class="text-xl font-bold tracking-wide bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
                    ChurnGuard AI
                </h1>
            </div>
            <span class="text-xs px-3 py-1 bg-slate-800 border border-slate-700 text-slate-400 rounded-full font-mono">
                Sequential_pkl.pkl
            </span>
        </div>
    </header>

    <!-- Main Section -->
    <main class="max-w-5xl mx-auto my-8 px-4 w-full">
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            <!-- Left Input Form -->
            <div class="lg:col-span-2 bg-slate-800/60 border border-slate-700/60 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
                <h2 class="text-lg font-semibold text-slate-200 mb-1">Customer Profile Features</h2>
                <p class="text-xs text-slate-400 mb-6">Enter bank customer parameters to run neural network churn analysis.</p>

                <form action="/predict" method="POST" class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    
                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Credit Score</label>
                        <input type="number" name="credit_score" min="300" max="850" value="{{ form_data.credit_score if form_data else 650 }}" required 
                               class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Age</label>
                        <input type="number" name="age" min="18" max="100" value="{{ form_data.age if form_data else 38 }}" required 
                               class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Tenure (Years)</label>
                        <input type="number" name="tenure" min="0" max="10" value="{{ form_data.tenure if form_data else 5 }}" required 
                               class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Account Balance ($)</label>
                        <input type="number" step="0.01" name="balance" value="{{ form_data.balance if form_data else 75000.00 }}" required 
                               class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Number of Products</label>
                        <select name="num_products" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                            <option value="1" {% if form_data and form_data.num_products == '1' %}selected{% endif %}>1 Product</option>
                            <option value="2" {% if not form_data or form_data.num_products == '2' %}selected{% endif %}>2 Products</option>
                            <option value="3" {% if form_data and form_data.num_products == '3' %}selected{% endif %}>3 Products</option>
                            <option value="4" {% if form_data and form_data.num_products == '4' %}selected{% endif %}>4 Products</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Geography</label>
                        <select name="geography" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                            <option value="France" {% if not form_data or form_data.geography == 'France' %}selected{% endif %}>France</option>
                            <option value="Germany" {% if form_data and form_data.geography == 'Germany' %}selected{% endif %}>Germany</option>
                            <option value="Spain" {% if form_data and form_data.geography == 'Spain' %}selected{% endif %}>Spain</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Has Credit Card?</label>
                        <select name="has_credit_card" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                            <option value="1" {% if not form_data or form_data.has_credit_card == '1' %}selected{% endif %}>Yes</option>
                            <option value="0" {% if form_data and form_data.has_credit_card == '0' %}selected{% endif %}>No</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Is Active Member?</label>
                        <select name="is_active_member" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                            <option value="1" {% if not form_data or form_data.is_active_member == '1' %}selected{% endif %}>Yes</option>
                            <option value="0" {% if form_data and form_data.is_active_member == '0' %}selected{% endif %}>No</option>
                        </select>
                    </div>

                    <div class="sm:col-span-2">
                        <label class="block text-xs font-medium text-slate-300 mb-1">Estimated Annual Salary ($)</label>
                        <input type="number" step="0.01" name="estimated_salary" value="{{ form_data.estimated_salary if form_data else 50000.00 }}" required 
                               class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                    </div>

                    <div class="sm:col-span-2 mt-2">
                        <button type="submit" 
                                class="w-full py-3 bg-gradient-to-r from-indigo-500 to-cyan-500 hover:from-indigo-600 hover:to-cyan-600 font-semibold rounded-lg text-white shadow-lg shadow-indigo-500/20 transition-all">
                            Predict Churn
                        </button>
                    </div>
                </form>
            </div>

            <!-- Right Analytics Dashboard -->
            <div class="bg-slate-800/60 border border-slate-700/60 rounded-2xl p-6 shadow-xl flex flex-col justify-between backdrop-blur-sm">
                <div>
                    <h2 class="text-lg font-semibold text-slate-200 mb-4">Prediction Analysis</h2>

                    {% if prediction_text %}
                        <div class="text-center py-6 border border-slate-700/50 rounded-xl bg-slate-900/50">
                            {% if is_churn %}
                                <div class="inline-flex items-center justify-center w-16 h-16 rounded-full bg-red-500/10 text-red-400 mb-3 border border-red-500/20">
                                    <i class="fa-solid fa-user-xmark text-2xl"></i>
                                </div>
                                <h3 class="text-2xl font-bold text-red-400">High Churn Risk</h3>
                                <p class="text-slate-400 text-xs mt-1">Customer is likely to exit</p>
                            {% else %}
                                <div class="inline-flex items-center justify-center w-16 h-16 rounded-full bg-emerald-500/10 text-emerald-400 mb-3 border border-emerald-500/20">
                                    <i class="fa-solid fa-user-check text-2xl"></i>
                                </div>
                                <h3 class="text-2xl font-bold text-emerald-400">Low Churn Risk</h3>
                                <p class="text-slate-400 text-xs mt-1">Customer is likely to stay</p>
                            {% endif %}

                            <div class="mt-6 text-3xl font-extrabold text-slate-100">
                                {{ prediction_text }}
                            </div>
                            <span class="text-xs text-slate-500 uppercase tracking-widest font-medium">Churn Probability</span>
                        </div>
                    {% elif error_text %}
                        <div class="p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-xs text-center">
                            {{ error_text }}
                        </div>
                    {% else %}
                        <div class="text-center py-12 text-slate-500">
                            <i class="fa-solid fa-chart-line text-4xl mb-3 opacity-50"></i>
                            <p class="text-xs">Fill parameters and click <strong>Predict Churn</strong> to evaluate probability.</p>
                        </div>
                    {% endif %}
                </div>

                <div class="border-t border-slate-700/60 pt-4 mt-6 text-xs text-slate-500 flex justify-between">
                    <span>Loaded Model:</span>
                    <span class="font-mono text-slate-400">Sequential_pkl.pkl</span>
                </div>
            </div>

        </div>
    </main>

    <!-- Footer -->
    <footer class="border-t border-slate-800 py-4 text-center text-xs text-slate-500">
        Bank Customer Churn Predictor
    </footer>

</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE, form_data=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error_text="Model file 'Sequential_pkl.pkl' failed to load.", form_data=request.form)

    try:
        # Get raw form inputs
        credit_score = float(request.form['credit_score'])
        age = float(request.form['age'])
        tenure = float(request.form['tenure'])
        balance = float(request.form['balance'])
        num_products = float(request.form['num_products'])
        has_credit_card = float(request.form['has_credit_card'])
        is_active_member = float(request.form['is_active_member'])
        estimated_salary = float(request.form['estimated_salary'])
        
        # Categorical feature mapping for Geography (One-hot encoding matching 10 features)
        geography = request.form['geography']
        geo_germany = 1.0 if geography == 'Germany' else 0.0
        geo_spain = 1.0 if geography == 'Spain' else 0.0

        # Construct array for 10 sequential inputs
        input_features = np.array([[
            credit_score,
            age,
            tenure,
            balance,
            num_products,
            has_credit_card,
            is_active_member,
            estimated_salary,
            geo_germany,
            geo_spain
        ]])

        # Execute Neural Network Prediction
        prediction_prob = float(model.predict(input_features)[0][0])
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
        return render_template_string(HTML_TEMPLATE, error_text=f"Prediction Error: {str(e)}", form_data=request.form)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
