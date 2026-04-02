import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.statespace.sarimax import SARIMAX

import lightgbm as lgb
import xgboost as xgb

from preprocess import add_time_features, build_recursive_features, make_supervised

warnings.filterwarnings("ignore")


DATA_DIR = Path("time series")
OUTPUT_DIR = Path("outputs_light")
OUTPUT_DIR.mkdir(exist_ok=True)


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

    train_df = train_df.sort_values("Date")
    test_df = test_df.sort_values("Date")

    train_df = train_df.set_index("Date")
    test_df = test_df.set_index("Date")

    train = train_df["number_sold"]
    test = test_df["number_sold"]

    train_part = train.loc[:"2017-12-31"]
    val_part = train.loc["2018-01-01":"2018-12-31"]

    return SeriesData(train=train_part, val=val_part, test=test)


def seasonal_naive_forecast(history, horizon, season=7):
    history = list(history)
    if len(history) < season:
        raise ValueError("Not enough history for seasonal naive forecast")
    last_season = history[-season:]
    reps = int(np.ceil(horizon / season))
    forecast = (last_season * reps)[:horizon]
    return np.array(forecast)


def fit_sarima(train, order, seasonal_order):
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=50)


def tune_sarima(train, val):
    grid = ParameterGrid(
        {
            "order": [(1, 1, 1), (2, 1, 1)],
            "seasonal_order": [(1, 1, 1, 7), (0, 1, 1, 7)],
        }
    )

    best = None
    for params in grid:
        try:
            model = fit_sarima(train, params["order"], params["seasonal_order"])
            pred = model.forecast(steps=len(val))
            score = mape(val.values, pred.values)
            if best is None or score < best["mape"]:
                best = {"mape": score, "params": params, "model": model}
        except Exception:
            continue

    return best


def prophet_predict(train, dates, changepoint_prior_scale, seasonality_mode):
    df = train.reset_index().rename(columns={"Date": "ds", "number_sold": "y"})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
    )
    model.fit(df)

    future = pd.DataFrame({"ds": dates})
    forecast = model.predict(future)
    return model, forecast["yhat"].values


def tune_prophet(train, val_series):
    grid = ParameterGrid(
        {
            "changepoint_prior_scale": [0.05, 0.5],
            "seasonality_mode": ["additive", "multiplicative"],
        }
    )

    best = None
    for params in grid:
        try:
            model, pred = prophet_predict(
                train,
                val_series.index,
                params["changepoint_prior_scale"],
                params["seasonality_mode"],
            )
            score = mape(val_series.values, pred)
            if best is None or score < best["mape"]:
                best = {"mape": score, "params": params, "model": model}
        except Exception:
            continue

    return best


def tune_lgbm(X_train, y_train, X_val, y_val):
    grid = ParameterGrid(
        {
            "n_estimators": [300, 600],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [31, 63],
        }
    )

    best = None
    for params in grid:
        model = lgb.LGBMRegressor(
            **params,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = mape(y_val, pred)
        if best is None or score < best["mape"]:
            best = {"mape": score, "params": params, "model": model}
    return best


def tune_xgb(X_train, y_train, X_val, y_val):
    grid = ParameterGrid(
        {
            "n_estimators": [300, 600],
            "learning_rate": [0.05, 0.1],
            "max_depth": [4, 6],
        }
    )

    best = None
    for params in grid:
        model = xgb.XGBRegressor(
            **params,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror",
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = mape(y_val, pred)
        if best is None or score < best["mape"]:
            best = {"mape": score, "params": params, "model": model}
    return best


def recursive_forecast(model, history, start_date, periods):
    history = list(history)
    preds = []
    dates = pd.date_range(start=start_date, periods=periods, freq="D")

    for date in dates:
        features = build_recursive_features(history, date)
        pred = float(model.predict(features)[0])
        preds.append(pred)
        history.append(pred)

    return dates, np.array(preds)


def save_plot(dates, actual, pred, title, path):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, actual, label="Actual", linewidth=1)
    plt.plot(dates, pred, label="Predicted", linewidth=1)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main(store, product):
    series_data = load_series(store, product)

    metrics = []
    predictions_2019 = pd.DataFrame(index=series_data.test.index)
    predictions_2019["actual"] = series_data.test.values

    # Seasonal Naive baseline
    naive_val = seasonal_naive_forecast(series_data.train.values, len(series_data.val), 7)
    naive_history = list(series_data.train.values) + list(series_data.val.values)
    naive_test = seasonal_naive_forecast(naive_history, len(series_data.test), 7)
    metrics.append(
        {
            "model": "SeasonalNaive",
            "split": "val",
            "mape": mape(series_data.val.values, naive_val),
            "rmse": rmse(series_data.val.values, naive_val),
        }
    )
    metrics.append(
        {
            "model": "SeasonalNaive",
            "split": "test",
            "mape": mape(series_data.test.values, naive_test),
            "rmse": rmse(series_data.test.values, naive_test),
        }
    )
    predictions_2019["SeasonalNaive"] = naive_test

    # SARIMA (tuning)
    sarima_best = tune_sarima(series_data.train, series_data.val)
    if sarima_best:
        sarima_model = sarima_best["model"]
        sarima_val = sarima_model.forecast(steps=len(series_data.val))
        sarima_model_full = fit_sarima(
            pd.concat([series_data.train, series_data.val]),
            sarima_best["params"]["order"],
            sarima_best["params"]["seasonal_order"],
        )
        sarima_test = sarima_model_full.forecast(steps=len(series_data.test))
        metrics.append(
            {
                "model": "SARIMA",
                "split": "val",
                "mape": mape(series_data.val.values, sarima_val.values),
                "rmse": rmse(series_data.val.values, sarima_val.values),
            }
        )
        metrics.append(
            {
                "model": "SARIMA",
                "split": "test",
                "mape": mape(series_data.test.values, sarima_test.values),
                "rmse": rmse(series_data.test.values, sarima_test.values),
            }
        )
        predictions_2019["SARIMA"] = sarima_test.values

    # Prophet (tuning)
    prophet_best = tune_prophet(series_data.train.to_frame("number_sold"), series_data.val)
    if prophet_best:
        prophet_model = prophet_best["model"]
        prophet_val = prophet_model.predict(pd.DataFrame({"ds": series_data.val.index}))
        prophet_val_pred = prophet_val["yhat"].values

        full_df = pd.concat([series_data.train, series_data.val]).reset_index()
        full_df.columns = ["ds", "y"]
        prophet_model_full = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=prophet_best["params"]["changepoint_prior_scale"],
            seasonality_mode=prophet_best["params"]["seasonality_mode"],
        )
        prophet_model_full.fit(full_df)
        prophet_test = prophet_model_full.predict(pd.DataFrame({"ds": series_data.test.index}))
        prophet_test_pred = prophet_test["yhat"].values

        metrics.append(
            {
                "model": "Prophet",
                "split": "val",
                "mape": mape(series_data.val.values, prophet_val_pred),
                "rmse": rmse(series_data.val.values, prophet_val_pred),
            }
        )
        metrics.append(
            {
                "model": "Prophet",
                "split": "test",
                "mape": mape(series_data.test.values, prophet_test_pred),
                "rmse": rmse(series_data.test.values, prophet_test_pred),
            }
        )
        predictions_2019["Prophet"] = prophet_test_pred

    # ML features
    full_series = pd.concat([series_data.train, series_data.val])
    sup = make_supervised(full_series)
    train_mask = sup.index <= "2017-12-31"
    val_mask = (sup.index >= "2018-01-01") & (sup.index <= "2018-12-31")

    X_train = sup.loc[train_mask].drop(columns=["y"])
    y_train = sup.loc[train_mask]["y"]
    X_val = sup.loc[val_mask].drop(columns=["y"])
    y_val = sup.loc[val_mask]["y"]

    lgb_best = tune_lgbm(X_train, y_train, X_val, y_val)
    if lgb_best:
        lgb_model = lgb_best["model"]
        hist = list(full_series.values)
        dates, lgb_test_pred = recursive_forecast(
            lgb_model,
            hist,
            series_data.test.index[0],
            len(series_data.test),
        )
        metrics.append(
            {
                "model": "LightGBM",
                "split": "test",
                "mape": mape(series_data.test.values, lgb_test_pred),
                "rmse": rmse(series_data.test.values, lgb_test_pred),
            }
        )
        metrics.append(
            {
                "model": "LightGBM",
                "split": "val",
                "mape": lgb_best["mape"],
                "rmse": rmse(y_val, lgb_model.predict(X_val)),
            }
        )
        predictions_2019["LightGBM"] = lgb_test_pred

    xgb_best = tune_xgb(X_train, y_train, X_val, y_val)
    if xgb_best:
        xgb_model = xgb_best["model"]
        hist = list(full_series.values)
        dates, xgb_test_pred = recursive_forecast(
            xgb_model,
            hist,
            series_data.test.index[0],
            len(series_data.test),
        )
        metrics.append(
            {
                "model": "XGBoost",
                "split": "test",
                "mape": mape(series_data.test.values, xgb_test_pred),
                "rmse": rmse(series_data.test.values, xgb_test_pred),
            }
        )
        metrics.append(
            {
                "model": "XGBoost",
                "split": "val",
                "mape": xgb_best["mape"],
                "rmse": rmse(y_val, xgb_model.predict(X_val)),
            }
        )
        predictions_2019["XGBoost"] = xgb_test_pred

    metrics_df = pd.DataFrame(metrics).sort_values(["split", "mape"])
    metrics_path = OUTPUT_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    predictions_path = OUTPUT_DIR / "predictions_2019.csv"
    predictions_2019.to_csv(predictions_path, index_label="Date")

    # Select best model by test MAPE
    test_metrics = metrics_df[metrics_df["split"] == "test"]
    best_row = test_metrics.sort_values("mape").iloc[0]
    best_model_name = best_row["model"]
    best_mape = float(best_row["mape"])
    best_rmse = float(best_row["rmse"])

    # Forecast 2020 with best model using full data (2010-2019)
    full_data = pd.concat([series_data.train, series_data.val, series_data.test])
    forecast_dates_2020 = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")

    forecast_2020 = None
    if best_model_name == "SARIMA" and sarima_best:
        sarima_full = fit_sarima(
            full_data,
            sarima_best["params"]["order"],
            sarima_best["params"]["seasonal_order"],
        )
        forecast_2020 = sarima_full.forecast(steps=len(forecast_dates_2020)).values
    elif best_model_name == "Prophet" and prophet_best:
        full_df = full_data.reset_index()
        full_df.columns = ["ds", "y"]
        prophet_full = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=prophet_best["params"]["changepoint_prior_scale"],
            seasonality_mode=prophet_best["params"]["seasonality_mode"],
        )
        prophet_full.fit(full_df)
        forecast_2020 = prophet_full.predict(pd.DataFrame({"ds": forecast_dates_2020}))["yhat"].values
    elif best_model_name == "LightGBM" and lgb_best:
        sup_full = make_supervised(full_data)
        X_full = sup_full.drop(columns=["y"])
        y_full = sup_full["y"]
        lgb_full = lgb.LGBMRegressor(
            **lgb_best["params"],
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        lgb_full.fit(X_full, y_full)
        hist = list(full_data.values)
        _, forecast_2020 = recursive_forecast(
            lgb_full,
            hist,
            forecast_dates_2020[0],
            len(forecast_dates_2020),
        )
    elif best_model_name == "XGBoost" and xgb_best:
        sup_full = make_supervised(full_data)
        X_full = sup_full.drop(columns=["y"])
        y_full = sup_full["y"]
        xgb_full = xgb.XGBRegressor(
            **xgb_best["params"],
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror",
        )
        xgb_full.fit(X_full, y_full)
        hist = list(full_data.values)
        _, forecast_2020 = recursive_forecast(
            xgb_full,
            hist,
            forecast_dates_2020[0],
            len(forecast_dates_2020),
        )

    if forecast_2020 is not None:
        forecast_df = pd.DataFrame(
            {"Date": forecast_dates_2020, "forecast": forecast_2020}
        ).set_index("Date")
        forecast_df.to_csv(OUTPUT_DIR / "forecast_2020_best.csv")

    # Plots
    best_pred_2019 = predictions_2019[best_model_name]
    save_plot(
        predictions_2019.index,
        predictions_2019["actual"],
        best_pred_2019,
        f"2019 Actual vs Predicted ({best_model_name}) | MAPE {best_mape:.2f}% | RMSE {best_rmse:.2f}",
        OUTPUT_DIR / "plot_2019_best.png",
    )

    if forecast_2020 is not None:
        save_plot(
            forecast_dates_2020,
            np.full(len(forecast_dates_2020), np.nan),
            forecast_2020,
            f"2020 Forecast ({best_model_name}) | MAPE {best_mape:.2f}% | RMSE {best_rmse:.2f}",
            OUTPUT_DIR / "plot_2020_forecast.png",
        )

    print("Best model:", best_model_name)
    print("Saved:")
    print("-", metrics_path)
    print("-", predictions_path)
    print("-", OUTPUT_DIR / "forecast_2020_best.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=int, default=0)
    parser.add_argument("--product", type=int, default=0)
    args = parser.parse_args()

    main(args.store, args.product)
