import streamlit as st
import pandas as pd

from preprocess import (
    load_data,
    handle_missing_values,
    get_features_and_target,
    split_and_preprocess
)

from model import ridge_regression_model

# ------------------------------
# App Configuration
# ------------------------------
st.set_page_config(
    page_title="Treatment Cost Prediction",
    page_icon="💊",
    layout="centered"
)

st.title("💊 Medical Treatment Cost Prediction")
st.write("Predict estimated treatment cost using Ridge Regression")

# ------------------------------
# Load & Train Model (Cached)
# ------------------------------
@st.cache_resource
def load_and_train_model():
    df = load_data("data/TreatmentCostAmount.csv")
    df = handle_missing_values(df)

    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess(X, y)

    model = ridge_regression_model()
    model.fit(X_train, y_train)

    return model, preprocessor


model, preprocessor = load_and_train_model()

# ------------------------------
# User Input Section
# ------------------------------
st.header("📋 Enter Patient Details")

#  Manual input (NO spinner)
age_input = st.text_input("Age", placeholder="Enter age (e.g. 35)")
bmi_input = st.text_input("BMI", placeholder="Enter BMI (e.g. 24.5)")

sex_input = st.selectbox("Sex", ["male", "female"])
smoker_input = st.selectbox("Smoker", ["yes", "no"])
region_input = st.selectbox("Region", ["north", "south", "east", "west"])
disease_input = st.selectbox(
    "Previous Disease",
    ["none", "diabetes", "heart", "other"]
)

# ------------------------------
# Encode Inputs
# ------------------------------
sex = 1 if sex_input == "male" else 0
smoker = 1 if smoker_input == "yes" else 0

region_map = {"north": 0, "south": 1, "east": 2, "west": 3}
disease_map = {"none": 0, "diabetes": 1, "heart": 2, "other": 3}

region = region_map[region_input]
disease = disease_map[disease_input]

# ------------------------------
# Prediction Button
# ------------------------------
if st.button("🔮 Predict Treatment Cost"):

    try:
        #  Convert user input
        age = int(age_input)
        bmi = float(bmi_input)

        #  Validation
        if age <= 0 or age > 120:
            raise ValueError("Age must be between 1 and 120")
        if bmi < 10 or bmi > 60:
            raise ValueError("BMI must be between 10 and 60")

        input_df = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "smoker": [smoker],
            "region": [region],
            "disease": [disease]
        })

        input_processed = preprocessor.transform(input_df)
        prediction = model.predict(input_processed)[0]

        st.success(f"💰 **Estimated Treatment Cost:** ₹ {prediction:,.2f}")

    except ValueError as e:
        st.error(f"❌ Invalid input: {e}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
