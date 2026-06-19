import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import (
    load_data,
    handle_missing_values,
    get_features_and_target,
    split_and_preprocess
)

from model import get_model


# ------------------------------
# App Configuration
# ------------------------------
st.set_page_config(
    page_title="Treatment Cost Prediction",
    page_icon="💊",
    layout="centered"
)

st.title(" Medical Treatment Cost Prediction")
# st.write("Predict estimated treatment cost using Ridge Regression")
st.write("Predict estimated treatment cost ")

# ------------------------------
# Sidebar Navigation
# ------------------------------
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select Option", ["EDA Dashboard","Prediction", "Bulk Prediction"])


# ------------------------------
# Load & Train Model (Cached)
# ------------------------------
@st.cache_resource
def load_and_train_model():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "TreatmentCostAmount.csv")

    df = load_data(DATA_PATH)
    df = handle_missing_values(df)

    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess(X, y)

    model = get_model("ridge")
    model.fit(X_train, y_train)

    return model, preprocessor, X_test, y_test


model, preprocessor, X_test, y_test = load_and_train_model()


# ==============================
#  PREDICTION SECTION
# ==============================
if option == "Prediction":

    st.header("📋 Enter Patient Details")

    age = st.number_input("Age", 1, 120, 30)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    steps = st.number_input("Daily Steps", value=5000)
    sleep = st.number_input("Sleep Hours", value=6.0)
    stress = st.number_input("Stress Level (1-10)", 1, 10, 5)
    prev_cost = st.number_input("Previous Year Cost", value=5000)

    gender = st.selectbox("Gender", ["Male", "Female"])
    smoker = st.selectbox("Smoker", ["Yes", "No"])
    insurance = st.selectbox("Insurance Type", ["Private", "Government", "None"])
    city = st.selectbox("City Type", ["Urban", "Semi-Urban", "Rural"])
    activity = st.selectbox("Activity Level", ["Low", "Medium", "High"])

    if st.button("🔮 Predict Treatment Cost"):

        try:
            input_df = pd.DataFrame({
                "age": [age],
                "bmi": [bmi],
                "daily_steps": [steps],
                "sleep_hours": [sleep],
                "stress_level": [stress],
                "doctor_visits_per_year": [2],
                "hospital_admissions": [0],
                "medication_count": [1],
                "insurance_coverage_pct": [70],
                "previous_year_cost": [prev_cost],
                "diabetes": [0],
                "hypertension": [0],
                "heart_disease": [0],
                "asthma": [0],
                "gender": [gender],
                "smoker": [smoker],
                "insurance_type": [insurance],
                "city_type": [city],
                "physical_activity_level": [activity]
            })

            input_processed = preprocessor.transform(input_df)
            prediction = model.predict(input_processed)[0]

            st.success(f"💰 Estimated Treatment Cost: ₹ {prediction:,.2f}")

        except Exception as e:
            st.error(f"❌ Error: {e}")


# ==============================
#  EDA DASHBOARD SECTION
# ==============================
elif option == "EDA Dashboard":

    st.header("📊 Exploratory Data Analysis")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "TreatmentCostAmount.csv")

    df = load_data(DATA_PATH)
    df = handle_missing_values(df)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Cost Distribution
    st.subheader("💰 Cost Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["annual_medical_cost"], bins=30)
    ax.set_xlabel("Medical Cost (₹)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Smoker vs Cost
    st.subheader(" Smoker vs Cost")
    avg_cost = df.groupby("smoker")["annual_medical_cost"].mean()
    fig, ax = plt.subplots()
    ax.bar(avg_cost.index, avg_cost.values)
    ax.set_xlabel("Smoker")
    ax.set_ylabel("Average Cost")
    st.pyplot(fig)

    # Age vs Cost
    st.subheader("📈 Age vs Cost")
    fig, ax = plt.subplots()
    ax.scatter(df["age"], df["annual_medical_cost"])
    ax.set_xlabel("Age")
    ax.set_ylabel("Cost")
    st.pyplot(fig)

    
    st.subheader(" BMI vs Cost")
    fig, ax = plt.subplots()
    ax.scatter(df["bmi"], df["annual_medical_cost"], alpha=0.5)
    ax.set_xlabel("BMI")
    ax.set_ylabel("Medical Cost (₹)")
    ax.set_title("BMI vs Cost")
    st.pyplot(fig)

    st.subheader("📊 Previous Year Cost vs Current Cost")
    fig, ax = plt.subplots()
    ax.scatter(df["previous_year_cost"], df["annual_medical_cost"], alpha=0.5)
    ax.set_xlabel("Previous Year Cost")
    ax.set_ylabel("Current Cost")
    ax.set_title("Previous vs Current Cost")
    st.pyplot(fig)

    st.subheader("🛡️ Insurance Type vs Cost")
    avg_cost = df.groupby("insurance_type")["annual_medical_cost"].mean()
    fig, ax = plt.subplots()
    ax.bar(avg_cost.index, avg_cost.values)
    ax.set_xlabel("Insurance Type")
    ax.set_ylabel("Average Cost")
    ax.set_title("Insurance Impact")
    st.pyplot(fig)

    st.subheader("🏃 Physical Activity vs Cost")
    avg_cost = df.groupby("physical_activity_level")["annual_medical_cost"].mean()
    fig, ax = plt.subplots()
    ax.bar(avg_cost.index, avg_cost.values)
    ax.set_xlabel("Activity Level")
    ax.set_ylabel("Average Cost")
    ax.set_title("Activity Impact on Cost")
    st.pyplot(fig)

    st.subheader(" Sleep Hours vs Cost")
    fig, ax = plt.subplots()
    ax.scatter(df["sleep_hours"], df["annual_medical_cost"], alpha=0.5)
    ax.set_xlabel("Sleep Hours")
    ax.set_ylabel("Cost")
    ax.set_title("Sleep vs Cost")
    st.pyplot(fig)

    st.subheader(" Stress Level vs Cost")
    fig, ax = plt.subplots()
    ax.scatter(df["stress_level"], df["annual_medical_cost"], alpha=0.5)
    ax.set_xlabel("Stress Level")
    ax.set_ylabel("Cost")
    ax.set_title("Stress Impact")
    st.pyplot(fig)


    # Correlation Heatmap
    st.subheader(" Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax)
    st.pyplot(fig)


   # ==============================
    #  MODEL EVALUATION 
    # ==============================
    st.subheader("📈 Model Evaluation Metrics")

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("R² Score", f"{r2:.4f}")

    #  Actual vs Predicted
    st.subheader("📊 Actual vs Predicted")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)

    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color='red', linestyle='--'
    )

    ax.set_xlabel("Actual Cost")
    ax.set_ylabel("Predicted Cost")

    st.pyplot(fig)

    #  Residual Plot
    st.subheader("📉 Residual Plot")

    residuals = y_test - y_pred

    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, alpha=0.5)

    ax.axhline(y=0, color='red', linestyle='--')

    ax.set_xlabel("Predicted Cost")
    ax.set_ylabel("Residuals")

    st.pyplot(fig)



# ==============================
#  BULK PREDICTION SECTION
# ==============================
elif option == "Bulk Prediction":

    st.header("📂 Bulk Prediction (Upload CSV)")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)

            st.subheader("📄 Uploaded Data")
            st.dataframe(df.head())

            required_cols = [
                "age", "bmi", "daily_steps", "sleep_hours", "stress_level",
                "doctor_visits_per_year", "hospital_admissions",
                "medication_count", "insurance_coverage_pct",
                "previous_year_cost", "diabetes", "hypertension",
                "heart_disease", "asthma", "gender", "smoker",
                "insurance_type", "city_type", "physical_activity_level"
            ]

            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
                st.stop()

            processed_data = preprocessor.transform(df)
            predictions = model.predict(processed_data)

            df["Predicted_Treatment_Cost"] = predictions

            st.subheader(" Prediction Results")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="predicted_costs.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")


# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
