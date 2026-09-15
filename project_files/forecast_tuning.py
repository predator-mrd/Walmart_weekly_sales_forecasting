"""Small, reproducible grid searches using expanding-window time-series CV."""

from dataclasses import dataclass
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def default_grids():
    """Keep the search small enough to run interactively on the Walmart data."""
    return {
        "SARIMAX": {"order": [(1, 1, 1), (1, 0, 1)], "seasonal_ar": [0, 1]},
        "STL": {"trend": ["add", None], "robust": [True, False]},
        "Prophet": {
            "changepoint_prior_scale": [0.01, 0.1],
            "seasonality_prior_scale": [1.0, 10.0],
        },
        "UCM": {
            "level": ["local level", "local linear trend"],
            "harmonics": [3, 6],
        },
        "Theta": {"theta": [1.0, 2.0, 3.0], "deseasonalize": [True, False]},
    }


def expanding_folds(y, n_splits=2, horizon=13, seasonal_period=52):
    """Require two annual cycles in the smallest training window."""
    if n_splits < 2 or horizon < 1 or seasonal_period < 2:
        raise ValueError("Use at least two folds, a positive horizon, and period >= 2.")
    minimum = 2 * seasonal_period
    if len(y) - n_splits * horizon < minimum:
        raise ValueError(
            f"Need at least {minimum + n_splits * horizon} training weeks "
            f"for {n_splits} folds of {horizon} weeks and two seasonal cycles."
        )
    if not isinstance(y.index, pd.DatetimeIndex):
        raise ValueError("Sales must use a DatetimeIndex.")
    if not y.index.is_monotonic_increasing or not y.index.is_unique:
        raise ValueError("Sales dates must be unique and sorted chronologically.")
    expected = pd.date_range(y.index[0], periods=len(y), freq="7D")
    if not y.index.equals(expected) or not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("Sales must be finite observations on a complete weekly index.")
    return list(TimeSeriesSplit(n_splits=n_splits, test_size=horizon).split(y))


def forecast_model(name, train_y, train_x, future_x, params, seasonal_period=52, seed=42):
    """Fit only on train_y; future_x contains known calendar features, never sales."""
    if not train_y.index.equals(train_x.index):
        raise ValueError("Training sales and calendar features must have matching dates.")
    if len(future_x) == 0 or future_x.index.min() <= train_y.index.max():
        raise ValueError("Forecast dates must follow the training window.")
    scale = 1e6  # Fixed dollar-to-million conversion improves optimizer conditioning.
    y = train_y / scale
    steps = len(future_x)
    with warnings.catch_warnings():
        # A failed optimizer must not silently win the search.
        warnings.simplefilter("error", ConvergenceWarning)
        if name == "SARIMAX":
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            model = SARIMAX(
                y, exog=train_x, order=tuple(params["order"]),
                seasonal_order=(params["seasonal_ar"], 0, 0, seasonal_period),
                enforce_stationarity=False, enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=500)
            values = fit.get_forecast(steps=steps, exog=future_x).predicted_mean
        elif name == "STL":
            from statsmodels.tsa.forecasting.stl import STLForecast
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            fit = STLForecast(
                y, ExponentialSmoothing, period=seasonal_period,
                robust=params["robust"],
                model_kwargs={"trend": params["trend"],
                              "damped_trend": params["trend"] is not None,
                              "initialization_method": "estimated"},
            ).fit()
            values = fit.forecast(steps)
        elif name == "Prophet":
            from prophet import Prophet
            train = pd.DataFrame({"ds": train_y.index, "y": y.to_numpy(),
                                  "Holiday_Flag": train_x["Holiday_Flag"].to_numpy()})
            future = pd.DataFrame({"ds": future_x.index,
                                   "Holiday_Flag": future_x["Holiday_Flag"].to_numpy()})
            model = Prophet(
                yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, uncertainty_samples=0, **params,
            )
            model.add_regressor("Holiday_Flag")
            model.fit(train, seed=seed)
            values = model.predict(future)["yhat"]
        elif name == "UCM":
            from statsmodels.tsa.statespace.structural import UnobservedComponents
            fit = UnobservedComponents(
                y, level=params["level"],
                freq_seasonal=[{"period": seasonal_period,
                                "harmonics": params["harmonics"]}],
                exog=train_x,
            ).fit(disp=False, maxiter=500)
            values = fit.get_forecast(steps=steps, exog=future_x).predicted_mean
        elif name == "Theta":
            from statsmodels.tsa.forecasting.theta import ThetaModel
            fit = ThetaModel(
                y, period=seasonal_period, deseasonalize=params["deseasonalize"],
                use_test=False,
            ).fit()
            values = fit.forecast(steps, theta=params["theta"])
        else:
            raise ValueError(f"Unknown model: {name}")
    values = np.asarray(values, dtype=float) * scale
    if values.shape != (steps,) or not np.isfinite(values).all():
        raise ValueError(f"{name} produced invalid forecasts.")
    return pd.Series(values, index=future_x.index, name=name)


@dataclass
class TuningResult:
    best_params: dict
    leaderboard: pd.DataFrame
    fold_scores: pd.DataFrame
    folds: pd.DataFrame
    cv_predictions: pd.DataFrame


def tune_models(train_y, train_x, *, grids=None, n_splits=2, horizon=13,
                seasonal_period=52, seed=42, verbose=True):
    """Select minimum mean fold RMSE; require every fold to succeed.

    Only pass the development/training data here. The final holdout belongs to
    the caller and is never used for selecting hyperparameters.
    """
    if not train_y.index.equals(train_x.index):
        raise ValueError("Training sales and calendar features must align.")
    if not np.isfinite(train_x.to_numpy(dtype=float)).all():
        raise ValueError("Calendar features must be finite.")
    splits = expanding_folds(train_y, n_splits, horizon, seasonal_period)
    fold_rows = []
    for fold, (past, future) in enumerate(splits, 1):
        fold_rows.append({
            "fold": fold, "train_start": train_y.index[past[0]],
            "train_end": train_y.index[past[-1]], "train_weeks": len(past),
            "validation_start": train_y.index[future[0]],
            "validation_end": train_y.index[future[-1]],
            "validation_weeks": len(future),
        })
    best_params, score_rows, search_rows, best_predictions = {}, [], [], []
    for name, grid in (default_grids() if grids is None else grids).items():
        candidates = list(ParameterGrid(grid))
        if verbose:
            print(f"Tuning {name}: {len(candidates)} candidates x {n_splits} folds", flush=True)
        best_score = np.inf
        for candidate, params in enumerate(candidates, 1):
            scores, predictions = [], []
            encoded = json.dumps(params, sort_keys=True)
            for fold, (past, future) in enumerate(splits, 1):
                row = {"model": name, "candidate": candidate, "params": encoded, "fold": fold}
                try:
                    pred = forecast_model(
                        name, train_y.iloc[past], train_x.iloc[past], train_x.iloc[future],
                        params, seasonal_period, seed,
                    )
                    actual = train_y.iloc[future]
                    if not pred.index.equals(actual.index) or not np.isfinite(pred).all():
                        raise ValueError("Forecast dates or values are invalid.")
                    rmse = float(np.sqrt(np.mean((actual.to_numpy() - pred.to_numpy()) ** 2)))
                    scores.append(rmse)
                    predictions.append(pd.DataFrame({
                        "model": name, "fold": fold, "Date": actual.index,
                        "Actual": actual.to_numpy(), "Prediction": pred.to_numpy(),
                    }))
                    row.update(rmse=rmse, status="ok", error="")
                except (ValueError, RuntimeError, np.linalg.LinAlgError, ConvergenceWarning) as error:
                    row.update(rmse=np.nan, status="failed", error=f"{type(error).__name__}: {error}")
                score_rows.append(row)
            complete = len(scores) == n_splits
            mean_score = float(np.mean(scores)) if complete else np.nan
            search_rows.append({"model": name, "candidate": candidate, "params": encoded,
                                "mean_cv_rmse": mean_score,
                                "std_cv_rmse": float(np.std(scores)) if complete else np.nan,
                                "successful_folds": len(scores),
                                "status": "ok" if complete else "failed"})
            if complete and mean_score < best_score:
                best_score = mean_score
                best_params[name] = params
                selected_predictions = predictions
        if name not in best_params:
            errors = [row["error"] for row in score_rows if row["model"] == name and row["status"] == "failed"]
            raise RuntimeError(f"All {name} candidates failed at least one fold: {errors}")
        best_predictions.extend(selected_predictions)
        if verbose:
            print(f"  Selected {best_params[name]}; mean CV RMSE ${best_score:,.0f}", flush=True)
    if not best_params:
        raise ValueError("Provide at least one model search grid.")
    return TuningResult(
        best_params, pd.DataFrame(search_rows).sort_values(["model", "mean_cv_rmse"]),
        pd.DataFrame(score_rows), pd.DataFrame(fold_rows),
        pd.concat(best_predictions, ignore_index=True),
    )
