import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def regression_metrics(y_true, y_pred, model_name):
    print(f"\n====== {model_name} Evaluation ======")
    print(f"MAE  : {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"MSE  : {mean_squared_error(y_true, y_pred):.2f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"R²   : {r2_score(y_true, y_pred):.4f}")


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


def plot_feature_importance(coefficients, feature_names):
    importance = abs(coefficients)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importance, y=feature_names)
    plt.title("Feature Importance (Lasso / Linear)")
    plt.xlabel("Absolute Coefficient Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred, title="Residual Plot"):
    """
    Residual Plot: Actual - Predicted vs Predicted Values
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_pred, y=residuals)

    # Zero reference line
    plt.axhline(y=0, color='red', linestyle='--')

    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(title)
    plt.tight_layout()
    plt.show()   