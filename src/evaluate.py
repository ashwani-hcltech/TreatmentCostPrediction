import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#  Metrics
def regression_metrics(y_true, y_pred, model_name):
    print(f"\n====== {model_name} Evaluation ======")
    print(f"MAE  : {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"MSE  : {mean_squared_error(y_true, y_pred):.2f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"R²   : {r2_score(y_true, y_pred):.4f}")


#  Actual vs Predicted
def plot_actual_vs_predicted(y_true, y_pred, title):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_true, y=y_pred)

    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             color='red', linestyle='--')

    plt.xlabel("Actual Cost")
    plt.ylabel("Predicted Cost")
    plt.title(title)
    plt.tight_layout()
    plt.show()


#  Feature Importance 
def plot_feature_importance(coefficients, feature_names):

    importance = abs(coefficients)

    #  Ensure same length
    min_len = min(len(importance), len(feature_names))
    importance = importance[:min_len]
    feature_names = feature_names[:min_len]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importance, y=feature_names)

    plt.title("Feature Importance (Lasso / Ridge)")
    plt.xlabel("Absolute Coefficient Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


#  Residual Plot
def plot_residuals(y_true, y_pred, title="Residual Plot"):

    residuals = y_true - y_pred

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_pred, y=residuals)

    plt.axhline(y=0, color='red', linestyle='--')

    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(title)
    plt.tight_layout()
    plt.show()