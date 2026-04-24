import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cost_distribution(df):
    """
    Histogram of treatment cost
    """
    plt.figure()
    plt.hist(df["charges"], bins=30)
    plt.title("Distribution of Treatment Cost")
    plt.xlabel("Treatment Cost")
    plt.ylabel("Number of Patients")
    plt.show()


def plot_smoker_vs_cost(df):
    """
    Average cost comparison between smokers and non-smokers
    """
    avg_cost = df.groupby("smoker")["charges"].mean()

    labels = ["Non-Smoker", "Smoker"]
    values = avg_cost.values

    plt.figure()
    plt.bar(labels, values)
    plt.title("Average Treatment Cost: Smoker vs Non-Smoker")
    plt.ylabel("Average Cost")
    plt.show()


def plot_disease_vs_cost(df):
    """
    Average cost by disease category
    """
    avg_cost = df.groupby("disease")["charges"].mean()
    labels = ["None", "Diabetes", "Heart", "Other"]

    plt.figure()
    plt.bar(labels, avg_cost.values)
    plt.title("Average Treatment Cost by Medical History")
    plt.ylabel("Average Cost")
    plt.show()


def plot_actual_vs_predicted(y_test, y_pred):
    """
    Scatter plot: Actual vs Predicted cost
    """
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()]
    )
    plt.xlabel("Actual Cost")
    plt.ylabel("Predicted Cost")
    plt.title("Actual vs Predicted Treatment Cost")
    plt.show()


def plot_residuals(y_test, y_pred):
    """
    Residual analysis plot
    """
    residuals = y_test - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0)
    plt.xlabel("Predicted Cost")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Cost")
    plt.show()


def plot_feature_importance(coefficients, feature_names):
    """
    Feature importance using Lasso coefficients
    """
    plt.figure()
    plt.barh(feature_names, coefficients)
    plt.title("Feature Importance (Lasso Regression)")
    plt.xlabel("Coefficient Value")
    plt.show()


def plot_age_vs_cost(df):
    """
    Scatter plot: Age vs Treatment Cost
    """
    plt.figure()
    plt.scatter(df["age"], df["charges"], alpha=0.5)
    plt.xlabel("Age")
    plt.ylabel("Treatment Cost")
    plt.title("Treatment Cost vs Age")
    plt.show()

def plot_age_group_vs_cost(df):
    """
    Bar chart: Average treatment cost by age group
    """
    age_bins = [0, 18, 30, 45, 60, 100]
    age_labels = ["<18", "18-30", "31-45", "46-60", "60+"]

    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)

    avg_cost = df.groupby("age_group")["charges"].mean()

    plt.figure()
    plt.bar(age_labels, avg_cost.values)
    plt.xlabel("Age Group")
    plt.ylabel("Average Treatment Cost")
    plt.title("Average Treatment Cost by Age Group")
    plt.show()