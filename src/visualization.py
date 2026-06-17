import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#  Target Distribution
def plot_cost_distribution(df):
    plt.figure()
    plt.hist(df["annual_medical_cost"], bins=30)
    plt.title("Distribution of Treatment Cost")
    plt.xlabel("Treatment Cost")
    plt.ylabel("Number of Patients")
    plt.show()


#  Smoker vs Cost
def plot_smoker_vs_cost(df):
    avg_cost = df.groupby("smoker")["annual_medical_cost"].mean()

    labels = avg_cost.index
    values = avg_cost.values

    plt.figure()
    plt.bar(labels, values)
    plt.title("Average Treatment Cost: Smoker vs Non-Smoker")
    plt.ylabel("Average Cost")
    plt.show()


#  Disease Impact 
def plot_disease_impact(df):

    diseases = ["diabetes", "hypertension", "heart_disease", "asthma"]

    plt.figure(figsize=(10, 6))

    for i, disease in enumerate(diseases):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(x=df[disease], y=df["annual_medical_cost"])
        plt.title(f"{disease} vs Cost")

    plt.tight_layout()
    plt.show()


#  Actual vs Predicted
def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red",
        linestyle="--"
    )

    plt.xlabel("Actual Cost")
    plt.ylabel("Predicted Cost")
    plt.title("Actual vs Predicted Treatment Cost")
    plt.show()


#  Residual Plot
def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")

    plt.xlabel("Predicted Cost")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Cost")
    plt.show()


#  Feature Importance 
def plot_feature_importance(coefficients, feature_names):

    importance = abs(coefficients)

    #  Fix mismatch
    min_len = min(len(importance), len(feature_names))
    importance = importance[:min_len]
    feature_names = feature_names[:min_len]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importance, y=feature_names)

    plt.title("Feature Importance")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


#  Age vs Cost
def plot_age_vs_cost(df):
    plt.figure()
    plt.scatter(df["age"], df["annual_medical_cost"], alpha=0.5)

    plt.xlabel("Age")
    plt.ylabel("Treatment Cost")
    plt.title("Treatment Cost vs Age")
    plt.show()


#  Age Group vs Cost
def plot_age_group_vs_cost(df):

    age_bins = [0, 18, 30, 45, 60, 100]
    age_labels = ["<18", "18-30", "31-45", "46-60", "60+"]

    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)

    avg_cost = df.groupby("age_group")["annual_medical_cost"].mean()

    plt.figure()
    plt.bar(age_labels, avg_cost.values)

    plt.xlabel("Age Group")
    plt.ylabel("Average Treatment Cost")
    plt.title("Average Treatment Cost by Age Group")
    plt.show()