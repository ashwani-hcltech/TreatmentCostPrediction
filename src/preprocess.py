import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ✅ Load Data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


# ✅ Handle Missing Values (Improved)
def handle_missing_values(df):
    print("✅ Null values before:\n", df.isnull().sum().to_string())

    # Separate columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # Numerical → median
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Categorical → most frequent
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    print("\n✅ Null values after:\n", df.isnull().sum().to_string())

    return df


# ✅ Feature Selection (UPDATED FOR YOUR DATASET)
def get_features_and_target(df):

    features = [
        "age",
        "bmi",
        "smoker",
        "gender",
        "city_type",
        "insurance_type",
        "physical_activity_level",
        "daily_steps",
        "sleep_hours",
        "stress_level",
        "doctor_visits_per_year",
        "hospital_admissions",
        "medication_count",
        "insurance_coverage_pct",
        "previous_year_cost",
        "diabetes",
        "hypertension",
        "heart_disease",
        "asthma"
    ]

    X = df[features]
    y = df["annual_medical_cost"]

    return X, y


# ✅ Split + Preprocessing Pipeline (BEST PRACTICE ✅)
def split_and_preprocess(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # ✅ Numeric pipeline
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # ✅ Categorical pipeline
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # One-hot encoding inside pipeline
        ("onehot", __import__("sklearn").preprocessing.OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    # ✅ Combine pipelines
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    # ✅ Fit & transform
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor
