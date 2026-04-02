import numpy as np
import pandas as pd


def add_time_features(df):
    df = df.copy()
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    df["weekofyear"] = df.index.isocalendar().week.astype(int)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df


def make_supervised(series):
    df = pd.DataFrame({"y": series})
    df = add_time_features(df)

    for lag in [1, 7, 14, 28, 365]:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    df["roll_mean_7"] = df["y"].shift(1).rolling(7).mean()
    df["roll_mean_28"] = df["y"].shift(1).rolling(28).mean()

    df = df.dropna()
    return df


def build_recursive_features(history, date):
    idx = pd.DatetimeIndex([date])
    df = pd.DataFrame({"y": [np.nan]}, index=idx)
    df = add_time_features(df)

    for lag in [1, 7, 14, 28, 365]:
        df[f"lag_{lag}"] = history[-lag]

    df["roll_mean_7"] = np.mean(history[-7:])
    df["roll_mean_28"] = np.mean(history[-28:])

    return df.drop(columns=["y"])
