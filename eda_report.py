import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path("time series")
OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_series(store, product):
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    train_df = pd.read_csv(train_path, parse_dates=["Date"])
    test_df = pd.read_csv(test_path, parse_dates=["Date"])

    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df[(df["store"] == store) & (df["product"] == product)]
    df = df.sort_values("Date")
    return df


def save_time_series_plot(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["number_sold"], linewidth=1)
    plt.title("Daily Sales Time Series")
    plt.xlabel("Date")
    plt.ylabel("Number Sold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_series.png")
    plt.close()


def save_dayofweek_plot(df):
    df = df.copy()
    df["dayofweek"] = df["Date"].dt.dayofweek
    avg = df.groupby("dayofweek")["number_sold"].mean()

    plt.figure(figsize=(8, 4))
    avg.plot(kind="bar")
    plt.title("Average Sales by Day of Week")
    plt.xlabel("Day of Week (0=Mon)")
    plt.ylabel("Average Number Sold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "avg_by_dayofweek.png")
    plt.close()


def save_month_plot(df):
    df = df.copy()
    df["month"] = df["Date"].dt.month
    avg = df.groupby("month")["number_sold"].mean()

    plt.figure(figsize=(8, 4))
    avg.plot(kind="bar")
    plt.title("Average Sales by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Number Sold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "avg_by_month.png")
    plt.close()


def save_descriptive_stats(df):
    stats = df["number_sold"].describe()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats.to_csv(OUTPUT_DIR / "descriptive_stats.csv")


def main(store, product):
    df = load_series(store, product)

    save_time_series_plot(df)
    save_dayofweek_plot(df)
    save_month_plot(df)
    save_descriptive_stats(df)

    print("Saved EDA outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=int, default=0)
    parser.add_argument("--product", type=int, default=0)
    args = parser.parse_args()

    main(args.store, args.product)
