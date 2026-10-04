import streamlit as st
import pandas as pd
import joblib
import os

# Load the model directly from the repo (Streamlit Cloud clones the whole repo,
# so the model file committed by the Github Actions pipeline is already on disk)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best_pima_diabetes_model_v1.joblib")
model = joblib.load(MODEL_PATH)

# Streamlit UI for PIMA Diabetes Prediction
st.title("PIMA Diabetes Prediction App")
st.write("""
This application predicts the likelihood of a patient having diabetes based on their health attributes.
Please enter the health metrics below to get a prediction.
""")

# User inputs
preg = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
plas = st.number_input("Plasma Glucose Concentration", min_value=0, max_value=300, value=120)
pres = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
skin = st.number_input("Triceps Skinfold Thickness (mm)", min_value=0, max_value=100, value=20)
test = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
mass = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
pedi = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'preg': preg,
    'plas': plas,
    'pres': pres,
    'skin': skin,
    'test': test,
    'mass': mass,
    'pedi': pedi,
    'age': age
}])

# Prediction button
if st.button("Predict Diabetes"):
    prediction = model.predict(input_data)[0]
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
