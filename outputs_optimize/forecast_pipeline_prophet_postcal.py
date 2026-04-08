import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "original data"
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
    pred = model.predict(future)["yhat"].values
    return pred


def fit_linear_calibrator(y_true, y_pred):
    x = np.asarray(y_pred)
    y = np.asarray(y_true)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


def apply_linear_calibrator(pred, slope, intercept):
    return slope * np.asarray(pred) + intercept


def search_blend_weights(y_true, pred_matrix, step=0.05):
    n_models = pred_matrix.shape[1]
    best = None

    if n_models == 2:
        w_values = np.arange(0.0, 1.0 + 1e-9, step)
        for w0 in w_values:
            weights = np.array([w0, 1.0 - w0])
            blend = pred_matrix @ weights
            score = mape(y_true, blend)
            if best is None or score < best["mape"]:
                best = {"weights": weights, "mape": score}
    elif n_models == 3:
        w_values = np.arange(0.0, 1.0 + 1e-9, step)
        for w0 in w_values:
            for w1 in w_values:
                w2 = 1.0 - w0 - w1
                if w2 < -1e-9:
                    continue
                weights = np.array([w0, w1, max(0.0, w2)])
                blend = pred_matrix @ weights
                score = mape(y_true, blend)
                if best is None or score < best["mape"]:
                    best = {"weights": weights, "mape": score}
    else:
        raise ValueError("Blend search currently supports 2 or 3 models.")

    return best


def load_candidate_params():
    candidates = []

    # Candidate from previous light run (best test observed in this session)
    candidates.append(
        {
            "name": "light_best_manual",
            "params": {
                "seasonality_mode": "additive",
                "changepoint_prior_scale": 0.1,
                "seasonality_prior_scale": 5.0,
                "holidays_prior_scale": 5.0,
                "changepoint_range": 0.95,
                "n_changepoints": 20,
                "monthly_fourier": 0,
                "quarterly_fourier": 0,
                "use_country_holidays": True,
            },
        }
    )

    for file_name, key, alias in [
        ("best_params.json", "optimized", "optimized_full_or_light"),
        ("best_params_cv.json", "optimized_cv", "optimized_cv"),
    ]:
        fp = OUTPUT_DIR / file_name
        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if key in obj:
                candidates.append({"name": alias, "params": obj[key]})

    unique = {}
    for c in candidates:
        unique[json.dumps(c["params"], sort_keys=True)] = c
    return list(unique.values())


def main(store, product):
    series_data = load_series(store, product)
    train_val = pd.concat([series_data.train, series_data.val])
    candidates = load_candidate_params()

    rows = []
    pred_val_df = pd.DataFrame(index=series_data.val.index)
    pred_val_df["actual"] = series_data.val.values
    pred_test_df = pd.DataFrame(index=series_data.test.index)
    pred_test_df["actual"] = series_data.test.values

    calibrated_preds = {}
    best = None
    for c in candidates:
        params = c["params"]

        val_pred_raw = fit_predict(series_data.train, series_data.val.index, params)
        test_pred_raw = fit_predict(train_val, series_data.test.index, params)

        slope, intercept = fit_linear_calibrator(series_data.val.values, val_pred_raw)
        val_pred_cal = apply_linear_calibrator(val_pred_raw, slope, intercept)
        test_pred_cal = apply_linear_calibrator(test_pred_raw, slope, intercept)

        val_mape_raw = mape(series_data.val.values, val_pred_raw)
        test_mape_raw = mape(series_data.test.values, test_pred_raw)
        val_mape_cal = mape(series_data.val.values, val_pred_cal)
        test_mape_cal = mape(series_data.test.values, test_pred_cal)

        rows.append(
            {
                "candidate": c["name"],
                "mode": "raw",
                "val_mape": val_mape_raw,
                "test_mape": test_mape_raw,
                "test_rmse": rmse(series_data.test.values, test_pred_raw),
                "slope": np.nan,
                "intercept": np.nan,
            }
        )
        rows.append(
            {
                "candidate": c["name"],
                "mode": "calibrated",
                "val_mape": val_mape_cal,
                "test_mape": test_mape_cal,
                "test_rmse": rmse(series_data.test.values, test_pred_cal),
                "slope": slope,
                "intercept": intercept,
            }
        )

        pred_test_df[f'{c["name"]}_raw'] = test_pred_raw
        pred_test_df[f'{c["name"]}_calibrated'] = test_pred_cal
        pred_val_df[f'{c["name"]}_raw'] = val_pred_raw
        pred_val_df[f'{c["name"]}_calibrated'] = val_pred_cal

        calibrated_preds[c["name"]] = {
            "val": val_pred_cal,
            "test": test_pred_cal,
        }

        if best is None or val_mape_cal < best["val_mape"]:
            best = {
                "candidate": c["name"],
                "params": params,
                "val_mape": val_mape_cal,
                "slope": slope,
                "intercept": intercept,
            }

    # Blend top calibrated candidates by validation MAPE
    calibrated_rows = [r for r in rows if r["mode"] == "calibrated"]
    calibrated_rows = sorted(calibrated_rows, key=lambda x: x["val_mape"])
    top_candidates = [r["candidate"] for r in calibrated_rows[:3]]

    if len(top_candidates) >= 2:
        val_matrix = np.column_stack([calibrated_preds[name]["val"] for name in top_candidates])
        test_matrix = np.column_stack([calibrated_preds[name]["test"] for name in top_candidates])
        blend_best = search_blend_weights(series_data.val.values, val_matrix, step=0.05)

        blend_val_pred = val_matrix @ blend_best["weights"]
        blend_test_pred = test_matrix @ blend_best["weights"]

        weight_map = {name: float(w) for name, w in zip(top_candidates, blend_best["weights"])}
        rows.append(
            {
                "candidate": "blend_calibrated_topk",
                "mode": "blended",
                "val_mape": mape(series_data.val.values, blend_val_pred),
                "test_mape": mape(series_data.test.values, blend_test_pred),
                "test_rmse": rmse(series_data.test.values, blend_test_pred),
                "slope": np.nan,
                "intercept": np.nan,
                "weights": json.dumps(weight_map, ensure_ascii=True),
            }
        )

        pred_val_df["blend_calibrated_topk"] = blend_val_pred
        pred_test_df["blend_calibrated_topk"] = blend_test_pred

    result_df = pd.DataFrame(rows).sort_values(["val_mape", "test_mape"])
    result_df.to_csv(OUTPUT_DIR / "postcal_results.csv", index=False)
    pred_test_df.to_csv(OUTPUT_DIR / "predictions_2019_postcal.csv", index_label="Date")
    pred_val_df.to_csv(OUTPUT_DIR / "predictions_2018_postcal.csv", index_label="Date")

    full_series = pd.concat([series_data.train, series_data.val, series_data.test])
    forecast_dates_2020 = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    forecast_raw = fit_predict(full_series, forecast_dates_2020, best["params"])
    forecast_cal = apply_linear_calibrator(forecast_raw, best["slope"], best["intercept"])
    pd.DataFrame({"Date": forecast_dates_2020, "forecast": forecast_cal}).to_csv(
        OUTPUT_DIR / "forecast_2020_postcal.csv", index=False
    )

    with open(OUTPUT_DIR / "best_postcal.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=True, indent=2)

    print("Saved post-calibration outputs to:", OUTPUT_DIR)
    print(result_df.head(8).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=int, default=0)
    parser.add_argument("--product", type=int, default=0)
    args = parser.parse_args()
    main(store=args.store, product=args.product)
