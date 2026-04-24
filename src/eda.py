from preprocess import load_data, handle_missing_values
from visualization import (
    plot_cost_distribution,
    plot_smoker_vs_cost,
    plot_disease_vs_cost,
    plot_age_vs_cost,
    plot_age_group_vs_cost
)


def run_eda():
    df = load_data("data/TreatmentCostAmount.csv")
    df = handle_missing_values(df)

    print("\nRunning Exploratory Data Analysis...")

    plot_cost_distribution(df)
    plot_smoker_vs_cost(df)
    plot_disease_vs_cost(df)    
    
    plot_age_vs_cost(df)
    plot_age_group_vs_cost(df)


if __name__ == "__main__":
    run_eda()