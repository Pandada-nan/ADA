import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "original data"
OUTPUT_DIR = Path(__file__).resolve().parent

REGRESSOR_COLS = ["is_month_start", "is_month_end", "is_weekend"]


def add_date_regressors(df):
    out = df.copy()
    out["is_month_start"] = out["ds"].dt.is_month_start.astype(int)
    out["is_month_end"] = out["ds"].dt.is_month_end.astype(int)
    out["is_weekend"] = (out["ds"].dt.dayofweek >= 5).astype(int)
    return out


def load_full_series(store=0, product=0):
    train_df = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["Date"])
    test_df = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["Date"])

    train_df = train_df[(train_df["store"] == store) & (train_df["product"] == product)]
    test_df = test_df[(test_df["store"] == store) & (test_df["product"] == product)]

    all_df = pd.concat([train_df, test_df], ignore_index=True).sort_values("Date")
    all_df = all_df.set_index("Date")
    return all_df["number_sold"]


def build_model(params):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode=params["seasonality_mode"],
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        holidays_prior_scale=params["holidays_prior_scale"],
        changepoint_range=params["changepoint_range"],
        n_changepoints=params["n_changepoints"],
    )

    if params.get("monthly_fourier", 0) > 0:
        model.add_seasonality("monthly", period=30.5, fourier_order=params["monthly_fourier"])
    if params.get("quarterly_fourier", 0) > 0:
        model.add_seasonality("quarterly", period=91.25, fourier_order=params["quarterly_fourier"])
    if params.get("use_country_holidays", False):
        model.add_country_holidays(country_name="CN")

    for col in REGRESSOR_COLS:
        model.add_regressor(col)

    return model


def main():
    with open(OUTPUT_DIR / "best_postcal.json", "r", encoding="utf-8") as f:
        best = json.load(f)

    params = best["params"]
    slope = float(best["slope"])
    intercept = float(best["intercept"])

    full_series = load_full_series(store=0, product=0)
    train_df = full_series.reset_index()
    train_df.columns = ["ds", "y"]
    train_df = add_date_regressors(train_df)

    model = build_model(params)
    model.fit(train_df)

    forecast_dates_2020 = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    future = pd.DataFrame({"ds": forecast_dates_2020})
    future = add_date_regressors(future)
    forecast_raw = model.predict(future)["yhat"].values
    forecast_cal = slope * forecast_raw + intercept

    out_df = pd.DataFrame({"Date": forecast_dates_2020, "forecast": forecast_cal})
    out_df.to_csv(OUTPUT_DIR / "forecast_2020_postcal.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(forecast_dates_2020, forecast_cal, linewidth=1.2, label="Forecast")
    plt.title("2020 Forecast (Optimized CV + Post-Calibration)")
    plt.xlabel("Date")
    plt.ylabel("Forecasted Number Sold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_2020_postcal.png", dpi=160)
    plt.close()

    print("Saved:", OUTPUT_DIR / "forecast_2020_postcal.csv")
    print("Saved:", OUTPUT_DIR / "plot_2020_postcal.png")


if __name__ == "__main__":
    main()
