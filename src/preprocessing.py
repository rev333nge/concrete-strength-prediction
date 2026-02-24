import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/concrete_data.csv"

FEATURE_COLS = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age",
]

MODEL_FEATURE_COLS = FEATURE_COLS + ["water_cement_ratio"]
TARGET_COL = "Strength"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    return df


def add_features(df):
    df = df.copy()
    df["water_cement_ratio"] = df["Water"] / df["Cement"]
    return df


def cap_outliers(df):
    df = df.copy()
    for col in FEATURE_COLS:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def split_data(df, random_state=42):
    x = df[MODEL_FEATURE_COLS]
    y = df[TARGET_COL]

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, train_size=0.60, random_state=random_state
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, train_size=0.50, random_state=random_state
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def load_and_prepare(cap=True):
    df = load_data()
    df = add_features(df)
    if cap:
        df = cap_outliers(df)
    return split_data(df)
