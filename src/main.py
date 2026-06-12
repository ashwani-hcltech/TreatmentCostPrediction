import pandas as pd

from preprocess import (
    load_data,
    handle_missing_values,
    get_features_and_target,
    split_and_preprocess
)

from model import (
    get_model
)

from evaluate import (
    regression_metrics,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residuals
)


def main():

    # ✅ 1. Load & Preprocess Data
    df = load_data("data/TreatmentCostAmount.csv")
    df = handle_missing_values(df)

    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess(X, y)

    # ✅ 2. Train Models (UPDATED ✅)
    lr_model = get_model("linear")
    lasso_model = get_model("lasso")
    ridge_model = get_model("ridge")

    lr_model.fit(X_train, y_train)
    lasso_model.fit(X_train, y_train)
    ridge_model.fit(X_train, y_train)

    # ✅ 3. Predictions
    lr_pred = lr_model.predict(X_test)
    lasso_pred = lasso_model.predict(X_test)
    ridge_pred = ridge_model.predict(X_test)

    # ✅ 4. Evaluation
    regression_metrics(y_test, lr_pred, "Linear Regression")
    regression_metrics(y_test, lasso_pred, "Lasso Regression")
    regression_metrics(y_test, ridge_pred, "Ridge Regression")

    plot_actual_vs_predicted(y_test, lr_pred, "Linear Regression")
    plot_actual_vs_predicted(y_test, lasso_pred, "Lasso Regression")
    plot_actual_vs_predicted(y_test, ridge_pred, "Ridge Regression")

    plot_residuals(y_test, lr_pred, "Residual Plot - Linear")
    plot_residuals(y_test, lasso_pred, "Residual Plot - Lasso")
    plot_residuals(y_test, ridge_pred, "Residual Plot - Ridge")

    # ✅ Feature Importance
    feature_names = preprocessor.get_feature_names_out()
    plot_feature_importance(lasso_model.coef_, feature_names)
    plot_feature_importance(ridge_model.coef_, feature_names)

    # ✅ 5. USER INPUT (UPDATED ✅)

    print("\nEnter patient details:")

    try:
        age = int(input("Age: "))
        bmi = float(input("BMI: "))
        steps = int(input("Daily Steps: "))
        sleep = float(input("Sleep Hours: "))
        stress = int(input("Stress Level (1-10): "))
        prev_cost = float(input("Previous Year Cost: "))

        gender = input("Gender (Male/Female): ").strip()
        smoker = input("Smoker (Yes/No): ").strip()
        insurance = input("Insurance Type (Private/Government/None): ").strip()
        city = input("City Type (Urban/Semi-Urban/Rural): ").strip()
        activity = input("Activity Level (Low/Medium/High): ").strip()

    except ValueError:
        print("❌ Invalid input")
        return

    # ✅ Create input dataframe (MATCH DATASET ✅)
    new_data = pd.DataFrame({
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

    # ✅ Transform using SAME pipeline
    new_data_processed = preprocessor.transform(new_data)

    # ✅ Predict (USE BEST MODEL ✅)
    predicted_cost = ridge_model.predict(new_data_processed)[0]

    print(f"\n💰 Estimated Treatment Cost: ₹ {predicted_cost:,.2f}")


if __name__ == "__main__":
    main()