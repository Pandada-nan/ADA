# COMP5152 Retail Sales Forecasting (Group 40)

This is an end-to-end time series forecasting project. We first analyze trend and seasonality, then compare multiple models. We run two grid-search variants (“light” and “full”) to tune and evaluate on 2019, and finally generate the 2020 forecast. All steps are implemented in this repository for easy reproduction.

## What you need to provide

Put the data under `time series/`:
- `train.csv`: historical data (the code splits it by time: train <= 2017, validation 2018)
- `test.csv`: the test year data (by default, this corresponds to 2019)

The default experiments use a single series: `store=0` and `product=0` (aligned with our report/PPT).

## Environment setup

```powershell
pip install -U pip
pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt` (pandas, Prophet, statsmodels, LightGBM, XGBoost, etc.).

## How to run (recommended order)

```powershell
# 1) EDA
python eda_report.py --store 0 --product 0


# 2) Light grid search vs full grid search
python forecast_pipeline.py --store 0 --product 0
python forecast_pipeline_full.py --store 0 --product 0


# 3) Export per-model grid search logs (on the validation set)
python param_search_log_light.py --store 0 --product 0
python param_search_log_full.py --store 0 --product 0

# 4) Prophet further optimization (rolling CV + linear post-calibration + 2020 forecast)
python outputs_optimize\forecast_pipeline_prophet_optimize_cv.py --store 0 --product 0
python outputs_optimize\forecast_pipeline_prophet_postcal.py --store 0 --product 0
python outputs_optimize\generate_2020_postcal_forecast.py
```

## Where the outputs are saved

- `outputs_light/`: results for light tuning (`metrics.csv`, 2019 predictions, and the best 2020 forecast + plots)
- `outputs_full/`: results for full tuning (`metrics.csv`, 2019 predictions, and plots)
- `param_search_outputs_light/` / `param_search_outputs_full/`: grid-search tables for each model (CSV)
- `tuning_compare_plots/`: `compare_*.png` (visual comparison between light vs full)
- `eda_outputs/`: EDA artifacts (line/bar plots and `descriptive_stats.csv`)
- `outputs_optimize/`: Prophet optimization artifacts (`best_postcal.json`, `final_results.csv`, `forecast_2020_postcal.csv`, etc.)

## Scripts overview

- `preprocess.py`: feature engineering for tree-based models (time features, lags, rolling statistics)
- `forecast_pipeline*.py`: train/validate/test, select the best model, and generate the 2020 forecast
- `param_search_log*.py`: export grid-search results as tables (useful for writing the report)
- `compare_tuning_plots.py`: plot light/full `metrics.csv` for comparison
- `eda_report.py`: EDA plots and descriptive statistics

## Experimental setup

- Data split: train <= 2017, validation 2018, test 2019
- Metrics: MAPE (%) and RMSE
- Baseline: Seasonal Naive (7-day weekly cycle)