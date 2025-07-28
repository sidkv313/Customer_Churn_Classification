import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("saved_models/random_forest.pkl")
scaler = joblib.load("saved_models/scaler.pkl")

# Expected feature order
expected_features = [
    "total_orders",
    "total_spend",
    "avg_spend_per_order",
    "avg_review_score",
    "num_payment_methods",
    "customer_tenure_days"
]

# Streamlit UI
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📉 Customer Churn Prediction")
st.markdown("Enter customer profile data to predict churn:")

# Collect inputs in correct order
total_orders = st.number_input("Total Orders", min_value=0)
total_spend = st.number_input("Total Spend (₹)", min_value=0.0, step=50.0)
avg_spend_per_order = st.number_input("Average Spend Per Order (₹)", min_value=0.0, step=10.0)
avg_review_score = st.slider("Average Review Score", 1.0, 5.0, step=0.1)
num_payment_methods = st.slider("Number of Payment Methods", 1, 5)
customer_tenure_days = st.number_input("Customer Tenure (Days)", min_value=0)

# Create input DataFrame with proper order
input_data = pd.DataFrame([[
    total_orders,
    total_spend,
    avg_spend_per_order,
    avg_review_score,
    num_payment_methods,
    customer_tenure_days
]], columns=expected_features)

# Transform and predict
X_scaled = scaler.transform(input_data)
if st.button("🔍 Predict Churn"):
    result = model.predict(X_scaled)[0]
    st.success("Prediction: **Churn**" if result == 1 else "Prediction: **No Churn**")
