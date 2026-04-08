import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.statespace.sarimax import SARIMAX

import lightgbm as lgb
import xgboost as xgb

from preprocess import add_time_features, make_supervised

warnings.filterwarnings("ignore")


DATA_DIR = Path("original data")
OUTPUT_DIR = Path("param_search_outputs_full")
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class SeriesData:
    train: pd.Series
    val: pd.Series


def mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_series(store, product):
    train_path = DATA_DIR / "train.csv"
    train_df = pd.read_csv(train_path, parse_dates=["Date"])
    train_df = train_df[(train_df["store"] == store) & (train_df["product"] == product)]
    train_df = train_df.sort_values("Date").set_index("Date")

    train = train_df["number_sold"]
    train_part = train.loc[:"2017-12-31"]
    val_part = train.loc["2018-01-01":"2018-12-31"]

    return SeriesData(train=train_part, val=val_part)


def fit_sarima(train, order, seasonal_order):
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=50)


def prophet_predict(train, dates, changepoint_prior_scale, seasonality_mode, seasonality_prior_scale=None):
    df = train.reset_index().rename(columns={"Date": "ds", "number_sold": "y"})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale if seasonality_prior_scale is not None else 10.0,
    )
    model.fit(df)
    future = pd.DataFrame({"ds": dates})
    forecast = model.predict(future)
    return forecast["yhat"].values


def sarima_grid(light):
    if light:
        return {
            "order": [(1, 1, 1), (2, 1, 1)],
            "seasonal_order": [(1, 1, 1, 7), (0, 1, 1, 7)],
        }
    return {
        "order": [
            (0, 1, 1),
            (1, 1, 0),
            (1, 1, 1),
            (2, 1, 1),
            (2, 1, 2),
            (3, 1, 1),
        ],
        "seasonal_order": [
            (0, 1, 1, 7),
            (1, 1, 0, 7),
            (1, 1, 1, 7),
            (2, 1, 1, 7),
            (2, 1, 2, 7),
        ],
    }


def prophet_grid(light):
    if light:
        return {
            "changepoint_prior_scale": [0.05, 0.5],
            "seasonality_mode": ["additive", "multiplicative"],
        }
    return {
        "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.5],
        "seasonality_mode": ["additive", "multiplicative"],
        "seasonality_prior_scale": [1.0, 5.0, 10.0],
    }


def lgbm_grid(light):
    if light:
        return {
            "n_estimators": [300, 600],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [31, 63],
        }
    return {
        "n_estimators": [300, 600, 900],
        "learning_rate": [0.03, 0.05, 0.1],
        "num_leaves": [31, 63, 127],
        "max_depth": [-1, 8, 12],
        "min_child_samples": [10, 20, 50],
    }


def xgb_grid(light):
    if light:
        return {
            "n_estimators": [300, 600],
            "learning_rate": [0.05, 0.1],
            "max_depth": [4, 6],
        }
    return {
        "n_estimators": [300, 600, 900],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "min_child_weight": [1, 5, 10],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    }


def run_sarima(series_data, light):
    results = []
    for params in ParameterGrid(sarima_grid(light)):
        try:
            model = fit_sarima(series_data.train, params["order"], params["seasonal_order"])
            pred = model.forecast(steps=len(series_data.val))
            results.append(
                {
                    "order": params["order"],
                    "seasonal_order": params["seasonal_order"],
                    "mape": mape(series_data.val.values, pred.values),
                    "rmse": rmse(series_data.val.values, pred.values),
                    "status": "ok",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "order": params["order"],
                    "seasonal_order": params["seasonal_order"],
                    "mape": None,
                    "rmse": None,
                    "status": f"error: {exc.__class__.__name__}",
                }
            )
    return pd.DataFrame(results)


def run_prophet(series_data, light):
    results = []
    grid = prophet_grid(light)
    for params in ParameterGrid(grid):
        try:
            pred = prophet_predict(
                series_data.train.to_frame("number_sold"),
                series_data.val.index,
                params["changepoint_prior_scale"],
                params["seasonality_mode"],
                params.get("seasonality_prior_scale"),
            )
            results.append(
                {
                    "changepoint_prior_scale": params["changepoint_prior_scale"],
                    "seasonality_mode": params["seasonality_mode"],
                    "seasonality_prior_scale": params.get("seasonality_prior_scale"),
                    "mape": mape(series_data.val.values, pred),
                    "rmse": rmse(series_data.val.values, pred),
                    "status": "ok",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "changepoint_prior_scale": params["changepoint_prior_scale"],
                    "seasonality_mode": params["seasonality_mode"],
                    "seasonality_prior_scale": params.get("seasonality_prior_scale"),
                    "mape": None,
                    "rmse": None,
                    "status": f"error: {exc.__class__.__name__}",
                }
            )
    return pd.DataFrame(results)


def run_lgbm(series_data, light):
    full_series = pd.concat([series_data.train, series_data.val])
    sup = make_supervised(full_series)
    train_mask = sup.index <= "2017-12-31"
    val_mask = (sup.index >= "2018-01-01") & (sup.index <= "2018-12-31")

    X_train = sup.loc[train_mask].drop(columns=["y"])
    y_train = sup.loc[train_mask]["y"]
    X_val = sup.loc[val_mask].drop(columns=["y"])
    y_val = sup.loc[val_mask]["y"]

    results = []
    for params in ParameterGrid(lgbm_grid(light)):
        model = lgb.LGBMRegressor(
            **params,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        results.append(
            {
                **params,
                "mape": mape(y_val, pred),
                "rmse": rmse(y_val, pred),
                "status": "ok",
            }
        )
    return pd.DataFrame(results)


def run_xgb(series_data, light):
    full_series = pd.concat([series_data.train, series_data.val])
    sup = make_supervised(full_series)
    train_mask = sup.index <= "2017-12-31"
    val_mask = (sup.index >= "2018-01-01") & (sup.index <= "2018-12-31")

    X_train = sup.loc[train_mask].drop(columns=["y"])
    y_train = sup.loc[train_mask]["y"]
    X_val = sup.loc[val_mask].drop(columns=["y"])
    y_val = sup.loc[val_mask]["y"]

    results = []
    for params in ParameterGrid(xgb_grid(light)):
        model = xgb.XGBRegressor(
            **params,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        results.append(
            {
                **params,
                "mape": mape(y_val, pred),
                "rmse": rmse(y_val, pred),
                "status": "ok",
            }
        )
    return pd.DataFrame(results)


def main(store, product, light):
    series_data = load_series(store, product)

    sarima_df = run_sarima(series_data, light)
    sarima_df.to_csv(OUTPUT_DIR / "sarima_param_search.csv", index=False)

    prophet_df = run_prophet(series_data, light)
    prophet_df.to_csv(OUTPUT_DIR / "prophet_param_search.csv", index=False)

    lgbm_df = run_lgbm(series_data, light)
    lgbm_df.to_csv(OUTPUT_DIR / "lightgbm_param_search.csv", index=False)

    xgb_df = run_xgb(series_data, light)
    xgb_df.to_csv(OUTPUT_DIR / "xgboost_param_search.csv", index=False)

    print("Saved parameter search logs to:", OUTPUT_DIR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=int, default=0)
    parser.add_argument("--product", type=int, default=0)
    parser.add_argument("--light", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.store, args.product, args.light)
