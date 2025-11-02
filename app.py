import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoder
model = joblib.load("no_show_model.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="🏥 Medical Appointment No-Show Prediction")

st.title("🏥 Medical Appointment No-Show Prediction")
st.write("This app predicts if a patient will miss their appointment based on provided information.")

# Sidebar for input
st.sidebar.header("Enter Patient Details")

Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Age = st.sidebar.slider("Age", 0, 100, 35)
Scholarship = st.sidebar.selectbox("Scholarship (Gov. support)", [0, 1])
Hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
Diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
Alcoholism = st.sidebar.selectbox("Alcoholism", [0, 1])
Handicap = st.sidebar.selectbox("Handicap", [0, 1])
SMS_received = st.sidebar.selectbox("Received SMS reminder", [0, 1])
waiting_days = st.sidebar.slider("Waiting Days", 0, 60, 5)
Neighbourhood = st.sidebar.text_input("Neighbourhood (type area name)", "JARDIM CAMBURI")

# Encode Gender and Neighbourhood
Gender_encoded = 1 if Gender == "Male" else 0
Neighbourhood_encoded = le.transform([Neighbourhood])[0] if Neighbourhood in le.classes_ else 0

# Create DataFrame for prediction
input_data = pd.DataFrame({
    'Gender': [Gender_encoded],
    'Age': [Age],
    'Scholarship': [Scholarship],
    'Hypertension': [Hypertension],
    'Diabetes': [Diabetes],
    'Alcoholism': [Alcoholism],
    'Handicap': [Handicap],
    'SMS_received': [SMS_received],
    'Neighbourhood': [Neighbourhood_encoded],
    'waiting_days': [waiting_days]
})

# Prediction
if st.button("🔍 Predict"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    if pred == 1:
        st.error(f"⚠️ Patient likely to MISS appointment ({prob:.2f}% chance)")
    else:
        st.success(f"✅ Patient likely to ATTEND appointment ({prob:.2f}% chance)")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit & Random Forest")
