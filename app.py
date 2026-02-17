
import streamlit as st
import joblib
import numpy as np

# 1. Load the saved model
model = joblib.load('malnutrition_model.pkl')

# 2. App Title and Description
st.title("👶 Child Malnutrition Prediction App")
st.write("Enter the details below to check the child's malnutrition risk status.")

# 3. Create Input Fields for the User
st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age_mother = st.number_input("Mother's Age", min_value=15, max_value=60, value=25)
    weight_child = st.number_input("Child's Weight (Kg)", min_value=0.5, max_value=30.0, value=2.5)
    income = st.selectbox("Household Income", options=[(1, "Low"), (2, "Medium"), (3, "High")], format_func=lambda x: x[1])

with col2:
    water = st.selectbox("Access to Clean Water", options=[(1, "Yes"), (0, "No")], format_func=lambda x: x[1])
    education = st.selectbox("Mother's Education Level", options=[(1, "Primary"), (2, "Secondary"), (3, "University")], format_func=lambda x: x[1])

# 4. Prediction Logic
if st.button("Predict Status"):
    # Arrange inputs in the same order as training
    # [Age, Weight, Income, Water, Education]
    features = np.array([[age_mother, weight_child, income[0], water[0], education[0]]])
    
    prediction = model.predict(features)
    
    # Handle both Classifier (0/1) and Regressor (decimal)
    result = 1 if prediction[0] >= 0.5 else 0
    
    st.subheader("Results:")
    if result == 1:
        st.error("⚠️ Prediction: The child is at HIGH RISK of Malnutrition.")
    else:
        st.success("✅ Prediction: The child is HEALTHY (Low Risk).")

