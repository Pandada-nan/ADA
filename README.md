# COMP5152 Forecasting Pipeline

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -r requirements.txt
```

## Run

```powershell
.venv\Scripts\python forecast_pipeline.py --store 0 --product 0
```

## Outputs

- outputs/metrics.csv
- outputs/predictions_2019.csv
- outputs/forecast_2020_best.csv
- outputs/plot_2019_best.png
- outputs/plot_2020_forecast.png
