import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# 🔹 Training Data for Credit Score Prediction
credit_data = {
    "on_time_payments": [50, 100, 200, 250, 300],
    "credit_utilization": [80, 60, 40, 30, 20],
    "total_debt": [50000, 40000, 30000, 20000, 10000],
    "credit_accounts": [2, 3, 5, 6, 8],
    "credit_score": [580, 640, 700, 750, 800],
}

df_credit = pd.DataFrame(credit_data)
X_credit = df_credit.drop(columns=["credit_score"])
y_credit = df_credit["credit_score"]

credit_model = LinearRegression()
credit_model.fit(X_credit, y_credit)

joblib.dump(credit_model, "credit_score_model.pkl")

# 🔹 Training Data for Financial Advice Model
financial_data = {
    "age": [25, 30, 40, 50, 60],
    "salary": [40000, 60000, 80000, 100000, 120000],
    "expenses": [20000, 25000, 30000, 35000, 40000],
    "credit_score": [580, 640, 700, 750, 800],
    "savings_advice": [10000, 20000, 30000, 40000, 50000],
}

df_finance = pd.DataFrame(financial_data)
X_finance = df_finance.drop(columns=["savings_advice"])
y_finance = df_finance["savings_advice"]

finance_model = LinearRegression()
finance_model.fit(X_finance, y_finance)

joblib.dump(finance_model, "financial_advice_model.pkl")

print("✅ Models trained and saved!")


# 🔥 Function to Predict Credit Score
def predict_credit_score(on_time_payments, credit_utilization, total_debt, credit_accounts):
    model = joblib.load("credit_score_model.pkl")
    new_data = pd.DataFrame([{
        "on_time_payments": on_time_payments,
        "credit_utilization": credit_utilization,
        "total_debt": total_debt,
        "credit_accounts": credit_accounts
    }])
    return model.predict(new_data)[0]


# 🔥 Function to Predict Savings Advice
def predict_savings_advice(age, salary, expenses, credit_score):
    model = joblib.load("financial_advice_model.pkl")
    new_data = pd.DataFrame([{
        "age": age,
        "salary": salary,
        "expenses": expenses,
        "credit_score": credit_score
    }])
    return model.predict(new_data)[0]


# 🏆 Example Predictions
predicted_credit = predict_credit_score(150, 50, 25000, 4)
predicted_savings = predict_savings_advice(35, 70000, 27000, 680)

print(f"🔹 Predicted Credit Score: {predicted_credit:.2f}")
print(f"🔹 Predicted Savings Advice: {predicted_savings:.2f}")
