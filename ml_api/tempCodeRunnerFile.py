from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

# Load AI models
credit_model = joblib.load("credit_score_model.pkl")
finance_model = joblib.load("financial_advice_model.pkl")

logging.basicConfig(level=logging.INFO)

@app.route("/predict-score", methods=["POST"])
def predict_score():
    try:
        data = request.get_json()
        required_fields = ["on_time_payments", "credit_utilization", "total_debt", "credit_accounts"]

        for field in required_fields:
            if field not in data or not isinstance(data[field], (int, float)):
                return jsonify({"error": f"Invalid or missing value for {field}"}), 400

        features = np.array([
            data["on_time_payments"],
            data["credit_utilization"],
            data["total_debt"],
            data["credit_accounts"],
        ]).reshape(1, -1)

        predicted_score = credit_model.predict(features)[0]
        return jsonify({"predicted_credit_score": round(predicted_score, 2)})

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/financial-advice", methods=["POST"])
def financial_advice():
    try:
        data = request.get_json()
        required_fields = ["age", "salary", "expenses", "credit_score"]

        for field in required_fields:
            if field not in data or not isinstance(data[field], (int, float)):
                return jsonify({"error": f"Invalid or missing value for {field}"}), 400

        features = np.array([
            data["age"],
            data["salary"],
            data["expenses"],
            data["credit_score"],
        ]).reshape(1, -1)

        savings_advice = finance_model.predict(features)[0]

        response = {
            "savings_suggestion": f"Save at least ₹{round(savings_advice, 2)} per month.",
            "loan_eligibility": "Eligible for low-interest loans" if data["credit_score"] > 700 else "Improve credit score for better loan rates",
            "budgeting_tip": "Keep expenses below 50% of your salary to maintain financial health.",
        }
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
