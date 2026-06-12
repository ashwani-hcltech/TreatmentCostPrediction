from preprocess import load_data, handle_missing_values
from visualization import (
    plot_cost_distribution,
    plot_smoker_vs_cost,
    plot_age_vs_cost,
    plot_age_group_vs_cost,
    plot_disease_impact
)


def run_eda():
    df = load_data("data/TreatmentCostAmount.csv")
    df = handle_missing_values(df)

    print("\n✅ Running Exploratory Data Analysis...")

    # ✅ Target distribution
    plot_cost_distribution(df)

    # ✅ Key relationships
    plot_smoker_vs_cost(df)
    plot_age_vs_cost(df)
    plot_age_group_vs_cost(df)

    # ✅ New disease analysis (UPDATED ✅)
    plot_disease_impact(df)


if __name__ == "__main__":
    run_eda()