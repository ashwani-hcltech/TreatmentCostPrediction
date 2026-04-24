import pandas as pd

from preprocess import (
    load_data,
    handle_missing_values,
    get_features_and_target,
    split_and_preprocess
)

from model import (
    build_linear_regression,
    build_lasso,
    ridge_regression_model
)

from evaluate import (
    regression_metrics,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residuals
)


def main():

    # 1. Load & Preprocess Data
    df = load_data("data/TreatmentCostAmount.csv")
    df = handle_missing_values(df)

    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess(X, y)

    # 2. Train Models
    lr_model = build_linear_regression()
    lasso_model = build_lasso()
    ridge_model = ridge_regression_model()   

    lr_model.fit(X_train, y_train)
    lasso_model.fit(X_train, y_train)
    ridge_model.fit(X_train, y_train)       

    # 3. Evaluate Models
    lr_pred = lr_model.predict(X_test)
    lasso_pred = lasso_model.predict(X_test)
    ridge_pred = ridge_model.predict(X_test) 

    regression_metrics(y_test, lr_pred, "Linear Regression")
    regression_metrics(y_test, lasso_pred, "Lasso Regression")
    regression_metrics(y_test, ridge_pred, "Ridge Regression")

    plot_actual_vs_predicted(y_test, lr_pred, "Linear Regression")
    plot_actual_vs_predicted(y_test, lasso_pred, "Lasso Regression")
    plot_actual_vs_predicted(y_test, ridge_pred, "Ridge Regression")

    plot_residuals(y_test, lr_pred, "Residual Plot - Linear Regression")
    plot_residuals(y_test, lasso_pred, "Residual Plot - Lasso Regression")
    plot_residuals(y_test, ridge_pred, "Residual Plot - Ridge Regression")

    feature_names = preprocessor.get_feature_names_out()
    plot_feature_importance(lasso_model.coef_, feature_names)
    plot_feature_importance(ridge_model.coef_, feature_names)  

    # 4. User Input → Prediction
    print("\nEnter customer details to predict Treatment Cost:")

    try:
        age = int(input("Age: "))
        sex_input = input("Sex (male/female): ").strip().lower()
        bmi = float(input("BMI: "))
        smoker_input = input("Smoker (yes/no): ").strip().lower()
        region_input = input("Region (north/south/east/west): ").strip().lower()
        disease_input = input(
            "Previous Disease (none/diabetes/heart/other): "
        ).strip().lower()

        # Validation
        if sex_input not in ["male", "female"]:
            raise ValueError("Gender must be male/female")
        if smoker_input not in ["yes", "no"]:
            raise ValueError("Smoker must be yes/no")
        if region_input not in ["north", "south", "east", "west"]:
            raise ValueError("Region must be north/south/east/west")

        disease_map = {
            "none": 0,
            "diabetes": 1,
            "heart": 2,
            "other": 3
        }

        if disease_input not in disease_map:
            raise ValueError("Disease must be none / diabetes / heart / other")

        # Numeric Encoding (MATCH CSV)
        sex = 1 if sex_input == "male" else 0
        smoker = 1 if smoker_input == "yes" else 0
        region = {"north": 0, "south": 1, "east": 2, "west": 3}[region_input]
        disease = disease_map[disease_input]

    except ValueError as e:
        print(f"\nInvalid input: {e}")
        return

    # Predict Treatment Cost (Using Ridge)
    new_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "smoker": [smoker],
        "region": [region],
        "disease": [disease]
    })

    new_data_processed = preprocessor.transform(new_data)
    predicted_cost = ridge_model.predict(new_data_processed)[0]

    print(f"\nEstimated Treatment Cost: ₹ {predicted_cost:,.2f}")


if __name__ == "__main__":
    main()
