import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "time series"
OUTPUT_DIR = Path(__file__).resolve().parent

REGRESSOR_COLS = ["is_month_start", "is_month_end", "is_weekend"]


@dataclass
class SeriesData:
    train: pd.Series
    val: pd.Series
    test: pd.Series


def mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_series(store, product):
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    train_df = pd.read_csv(train_path, parse_dates=["Date"])
    test_df = pd.read_csv(test_path, parse_dates=["Date"])

    train_df = train_df[(train_df["store"] == store) & (train_df["product"] == product)]
    test_df = test_df[(test_df["store"] == store) & (test_df["product"] == product)]

    train_df = train_df.sort_values("Date").set_index("Date")
    test_df = test_df.sort_values("Date").set_index("Date")

    train = train_df["number_sold"]
    test = test_df["number_sold"]
    return SeriesData(
        train=train.loc[:"2017-12-31"],
        val=train.loc["2018-01-01":"2018-12-31"],
        test=test,
    )


def add_date_regressors(df):
    out = df.copy()
    out["is_month_start"] = out["ds"].dt.is_month_start.astype(int)
    out["is_month_end"] = out["ds"].dt.is_month_end.astype(int)
    out["is_weekend"] = (out["ds"].dt.dayofweek >= 5).astype(int)
    return out


def make_model(params):
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

    if params["monthly_fourier"] > 0:
        model.add_seasonality("monthly", period=30.5, fourier_order=params["monthly_fourier"])
    if params["quarterly_fourier"] > 0:
        model.add_seasonality("quarterly", period=91.25, fourier_order=params["quarterly_fourier"])
    if params["use_country_holidays"]:
        model.add_country_holidays(country_name="CN")

    for col in REGRESSOR_COLS:
        model.add_regressor(col)
    return model


def fit_predict(train_series, pred_dates, params):
    train_df = train_series.reset_index()
    train_df.columns = ["ds", "y"]
    train_df = add_date_regressors(train_df)

    model = make_model(params)
    model.fit(train_df)

    future = pd.DataFrame({"ds": pd.to_datetime(pred_dates)})
    future = add_date_regressors(future)
    forecast = model.predict(future)
    return forecast["yhat"].values


def get_cv_folds(pre2019_series):
    return [
        ("2014-12-31", "2015-01-01", "2015-12-31"),
        ("2015-12-31", "2016-01-01", "2016-12-31"),
        ("2016-12-31", "2017-01-01", "2017-12-31"),
        ("2017-12-31", "2018-01-01", "2018-12-31"),
    ]


def score_with_cv(pre2019_series, params):
    fold_rows = []
    for train_end, val_start, val_end in get_cv_folds(pre2019_series):
        tr = pre2019_series.loc[:train_end]
        va = pre2019_series.loc[val_start:val_end]
        pred = fit_predict(tr, va.index, params)
        fold_rows.append(
            {
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "fold_mape": mape(va.values, pred),
                "fold_rmse": rmse(va.values, pred),
            }
        )
    fold_df = pd.DataFrame(fold_rows)
    return float(fold_df["fold_mape"].mean()), float(fold_df["fold_rmse"].mean()), fold_df


def build_core_grid():
    return ParameterGrid(
        {
            "seasonality_mode": ["additive", "multiplicative"],
            "changepoint_prior_scale": [0.03, 0.1, 0.3],
            "seasonality_prior_scale": [5.0, 10.0],
            "holidays_prior_scale": [1.0, 5.0],
            "changepoint_range": [0.85, 0.95],
            "n_changepoints": [15, 30],
        }
    )


def build_augment_grid():
    return ParameterGrid(
        {
            "monthly_fourier": [0, 5, 8],
            "quarterly_fourier": [0, 3],
            "use_country_holidays": [False, True],
        }
    )


def tune_with_cv(pre2019_series):
    core_best = None
    search_rows = []
    fold_rows = []

    for core in build_core_grid():
        candidate = {**core, "monthly_fourier": 0, "quarterly_fourier": 0, "use_country_holidays": False}
        try:
            avg_mape, avg_rmse, fold_df = score_with_cv(pre2019_series, candidate)
            search_rows.append({**candidate, "cv_mape": avg_mape, "cv_rmse": avg_rmse, "status": "ok"})
            fold_df = fold_df.assign(stage="core", **candidate)
            fold_rows.extend(fold_df.to_dict("records"))
            if core_best is None or avg_mape < core_best["cv_mape"]:
                core_best = {"params": candidate, "cv_mape": avg_mape}
        except Exception as exc:
            search_rows.append({**candidate, "cv_mape": None, "cv_rmse": None, "status": f"error:{exc.__class__.__name__}"})

    if core_best is None:
        raise RuntimeError("Core CV tuning failed.")

    final_best = None
    for aug in build_augment_grid():
        candidate = {**core_best["params"], **aug}
        try:
            avg_mape, avg_rmse, fold_df = score_with_cv(pre2019_series, candidate)
            search_rows.append({**candidate, "cv_mape": avg_mape, "cv_rmse": avg_rmse, "status": "ok"})
            fold_df = fold_df.assign(stage="augment", **candidate)
            fold_rows.extend(fold_df.to_dict("records"))
            if final_best is None or avg_mape < final_best["cv_mape"]:
                final_best = {"params": candidate, "cv_mape": avg_mape}
        except Exception as exc:
            search_rows.append({**candidate, "cv_mape": None, "cv_rmse": None, "status": f"error:{exc.__class__.__name__}"})

    if final_best is None:
        raise RuntimeError("Augment CV tuning failed.")

    return final_best, pd.DataFrame(search_rows), pd.DataFrame(fold_rows)


def tune_baseline(train_series, val_series):
    grid = ParameterGrid(
        {
            "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.5],
            "seasonality_mode": ["additive", "multiplicative"],
            "seasonality_prior_scale": [1.0, 5.0, 10.0],
        }
    )
    best = None
    for g in grid:
        params = {
            "seasonality_mode": g["seasonality_mode"],
            "changepoint_prior_scale": g["changepoint_prior_scale"],
            "seasonality_prior_scale": g["seasonality_prior_scale"],
            "holidays_prior_scale": 10.0,
            "changepoint_range": 0.8,
            "n_changepoints": 25,
            "monthly_fourier": 0,
            "quarterly_fourier": 0,
            "use_country_holidays": False,
        }
        pred = fit_predict(train_series, val_series.index, params)
        score = mape(val_series.values, pred)
        if best is None or score < best["val_mape"]:
            best = {"params": params, "val_mape": score, "val_pred": pred}
    return best


def save_plot(dates, actual, base_pred, opt_pred, path):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, actual, label="Actual", linewidth=1.2)
    plt.plot(dates, base_pred, label="Baseline", linewidth=1.0)
    plt.plot(dates, opt_pred, label="OptimizedCV", linewidth=1.0)
    plt.title("2019 Forecast Comparison (Baseline vs OptimizedCV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main(store, product):
    series_data = load_series(store, product)
    pre2019 = pd.concat([series_data.train, series_data.val])

    baseline_best = tune_baseline(series_data.train, series_data.val)
    cv_best, search_log, fold_log = tune_with_cv(pre2019)

    base_test_pred = fit_predict(pre2019, series_data.test.index, baseline_best["params"])
    opt_test_pred = fit_predict(pre2019, series_data.test.index, cv_best["params"])

    metrics_df = pd.DataFrame(
        [
            {
                "model": "ProphetBaseline",
                "split": "val",
                "mape": mape(series_data.val.values, baseline_best["val_pred"]),
                "rmse": rmse(series_data.val.values, baseline_best["val_pred"]),
            },
            {
                "model": "ProphetBaseline",
                "split": "test",
                "mape": mape(series_data.test.values, base_test_pred),
                "rmse": rmse(series_data.test.values, base_test_pred),
            },
            {
                "model": "ProphetOptimizedCV",
                "split": "cv_mean",
                "mape": cv_best["cv_mape"],
                "rmse": np.nan,
            },
            {
                "model": "ProphetOptimizedCV",
                "split": "test",
                "mape": mape(series_data.test.values, opt_test_pred),
                "rmse": rmse(series_data.test.values, opt_test_pred),
            },
        ]
    )
    metrics_df.to_csv(OUTPUT_DIR / "metrics_cv.csv", index=False)

    pred_df = pd.DataFrame(index=series_data.test.index)
    pred_df["actual"] = series_data.test.values
    pred_df["ProphetBaseline"] = base_test_pred
    pred_df["ProphetOptimizedCV"] = opt_test_pred
    pred_df.to_csv(OUTPUT_DIR / "predictions_2019_cv.csv", index_label="Date")

    search_log.to_csv(OUTPUT_DIR / "prophet_tuning_log_cv.csv", index=False)
    fold_log.to_csv(OUTPUT_DIR / "prophet_tuning_folds_cv.csv", index=False)

    with open(OUTPUT_DIR / "best_params_cv.json", "w", encoding="utf-8") as f:
        json.dump({"baseline": baseline_best["params"], "optimized_cv": cv_best["params"]}, f, ensure_ascii=True, indent=2)

    save_plot(series_data.test.index, series_data.test.values, base_test_pred, opt_test_pred, OUTPUT_DIR / "plot_2019_compare_cv.png")

    full_series = pd.concat([series_data.train, series_data.val, series_data.test])
    forecast_dates_2020 = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    forecast_2020 = fit_predict(full_series, forecast_dates_2020, cv_best["params"])
    pd.DataFrame({"Date": forecast_dates_2020, "forecast": forecast_2020}).to_csv(
        OUTPUT_DIR / "forecast_2020_optimized_cv.csv", index=False
    )

    print("Saved CV optimization outputs to:", OUTPUT_DIR)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=int, default=0)
    parser.add_argument("--product", type=int, default=0)
    args = parser.parse_args()
    main(store=args.store, product=args.product)
