import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def handle_missing_values(df):
    print("Null values before:\n", df.isnull().sum().to_string())

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # Fill numeric columns only if needed
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    print("\nNull values after:\n", df.isnull().sum().to_string())
    return df


def get_features_and_target(df):
    
    features = [
        "age",
        "sex",
        "bmi",
        "smoker",
        "region",
        "disease"
    ]

    X = df[features]
    y = df["charges"]   

    return X, y


def split_and_preprocess(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    #  ALL FEATURES ARE NUMERIC NOW
    num_cols = X.columns.tolist()

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols)
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor