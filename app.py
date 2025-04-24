import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


model = joblib.load('credit_ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')


st.title("Credit Risk Prediction App")
st.write("Upload loan applicant data")

# Input form
with st.form("input_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", options=["male", "female"])
    job = st.selectbox("Job", options=[0, 1, 2, 3])
    housing = st.selectbox("Housing", options=["own", "free", "rent"])
    saving_accounts = st.selectbox("Saving Accounts", options=["little", "moderate", "quite rich", "rich", "no_info"])
    checking_account = st.selectbox("Checking Account", options=["little", "moderate", "rich", "no_info"])
    credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
    duration = st.number_input("Duration (in months)", min_value=1, value=12)
    purpose = st.selectbox("Purpose", options=["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation/others"])

    submitted = st.form_submit_button("Predict")


label_mappings = {
    'Sex': {"male": 1, "female": 0},
    'Housing': {"own": 1, "free": 0, "rent": 2},
    'Saving accounts': {"little": 0, "moderate": 1, "quite rich": 2, "rich": 3, "no_info": 4},
    'Checking account': {"little": 0, "moderate": 1, "rich": 2, "no_info": 3},
    'Purpose': {"radio/TV": 5, "education": 3, "furniture/equipment": 4, "car": 1, "business": 0,
                "domestic appliances": 2, "repairs": 6, "vacation/others": 7}
}

if submitted:
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [label_mappings['Sex'][sex]],
        'Job': [job],
        'Housing': [label_mappings['Housing'][housing]],
        'Saving accounts': [label_mappings['Saving accounts'][saving_accounts]],
        'Checking account': [label_mappings['Checking account'][checking_account]],
        'Credit amount': [credit_amount],
        'Duration': [duration],
        'Purpose': [label_mappings['Purpose'][purpose]]
    })


    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    
    st.markdown("### Prediction Result")
    if prediction == 1:
        st.success(f"Good Credit Risk ✅ (Confidence: {probability:.2f})")
    else:
        st.error(f"Bad Credit Risk ⚠️ (Confidence: {1 - probability:.2f})")
