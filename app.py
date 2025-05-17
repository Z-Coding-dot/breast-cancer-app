from flask import Flask, request, render_template, url_for
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import json
from sklearn.datasets import load_breast_cancer

# -------------------- Load Artifacts --------------------
# Load feature names from the original dataset
data = load_breast_cancer()
feature_names = data.feature_names

# Load saved model and scaler
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load evaluation metrics
with open("model_metrics.json", "r") as f:
    model_metrics = json.load(f)

# -------------------- Flask App --------------------
app = Flask(__name__)

@app.route('/')
def index():
    # Render the form with no prediction yet
    return render_template(
        'index.html',
        prediction_text=None,
        metrics=model_metrics,
        download_link=None
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Read inputs
        values = [float(request.form[str(i)]) for i in range(len(feature_names))]

        # 2. Build DataFrame with the original column names
        df_input = pd.DataFrame([values], columns=feature_names)

        # 3. Scale using DataFrame (no warnings)
        scaled = scaler.transform(df_input)

        # 4. Predict
        pred = model.predict(scaled)[0]
        result = "Malignant (Cancer)" if pred == 0 else "Benign (No Cancer)"

        # 5. Save CSV and render
        df_out = df_input.copy()
        df_out["Prediction"] = result
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        fname = f"prediction_{ts}.csv"
        df_out.to_csv(f"static/{fname}", index=False)

        return render_template(
            'index.html',
            prediction_text=f"Result: {result}",
            metrics=model_metrics,
            download_link=url_for('static', filename=fname)
        )
    except Exception as e:
        import traceback
        print(f"Error in /predict: {e}")
        print(traceback.format_exc())
        return render_template(
            'index.html',
            prediction_text="⚠️ Error: Invalid input. Please ensure all fields are numeric.",
            metrics=model_metrics,
            download_link=None
        )


if __name__ == '__main__':
    # Debug=True for development, turn off in production
    app.run(debug=True)
