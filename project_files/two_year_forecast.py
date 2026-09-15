"""Create a 104-week Walmart forecast chart with an actual-to-forecast cutoff.

The existing notebook evaluates models on a 13-week historical holdout.  This
script is for the separate planning view: it refits each already-tuned model on
all observed weeks and forecasts the next two years.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
# Keep Matplotlib's cache in the project when a user profile is read-only.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast_tuning import forecast_model


MODEL_NAMES = ["SARIMAX", "STL", "Prophet", "UCM", "Theta"]
SEASONAL_PERIOD = 52
RANDOM_SEED = 42
DEFAULT_TEST_WEEKS = 13


def future_holiday_flags(dates: pd.DatetimeIndex) -> pd.Series:
    """Recreate the four annual holiday-week patterns in the source data.

    The source only supplies historical holiday flags. For future weeks this
    calendar assumption marks the second Friday in February, the Friday after
    US Labor Day, Black Friday, and the final Friday in December.
    """
    flags = pd.Series(0.0, index=dates, name="Holiday_Flag")
    for year in sorted(set(dates.year)):
        february = pd.date_range(f"{year}-02-01", f"{year}-02-28", freq="W-FRI")[1]
        labor_day = pd.Timestamp(f"{year}-09-01")
        labor_day += pd.offsets.Week(weekday=0)
        labor_friday = labor_day + pd.offsets.Week(weekday=4)
        black_friday = pd.Timestamp(f"{year}-11-01")
        black_friday += pd.offsets.Week(weekday=4)
        december_fridays = pd.date_range(f"{year}-12-01", f"{year}-12-31", freq="W-FRI")
        for holiday in (february, labor_friday, black_friday, december_fridays[-1]):
            if holiday in flags.index:
                flags.loc[holiday] = 1.0
    return flags


def load_weekly_sales() -> pd.DataFrame:
    raw = pd.read_csv(PROJECT_DIR / "Dataset" / "Walmart.csv")
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True)
    weekly = (
        raw.groupby("Date", as_index=True)
        .agg(Weekly_Sales=("Weekly_Sales", "sum"), Holiday_Flag=("Holiday_Flag", "max"))
        .sort_index()
        .asfreq("W-FRI")
    )
    if weekly.isna().any().any():
        raise ValueError("The historical series has missing weeks; cannot make a continuous forecast.")
    return weekly


def load_best_params() -> dict:
    with (PROJECT_DIR / "walmart_best_params.json").open(encoding="utf-8") as file:
        return json.load(file)


def make_forecasts(
    train_y: pd.Series, train_x: pd.DataFrame, forecast_x: pd.DataFrame, best_params: dict
) -> pd.DataFrame:
    """Forecast a known future index without exposing its sales to the models."""
    forecasts = pd.DataFrame(index=forecast_x.index)
    for name in MODEL_NAMES:
        print(f"Forecasting {name}...", flush=True)
        forecasts[name] = forecast_model(
            name, train_y, train_x, forecast_x, best_params[name],
            seasonal_period=SEASONAL_PERIOD, seed=RANDOM_SEED,
        )
    forecasts["Ensemble"] = forecasts[MODEL_NAMES].mean(axis=1)
    forecasts.index.name = "Date"
    return forecasts

def create_summary(forecasts: pd.DataFrame, test_forecasts: pd.DataFrame) -> pd.DataFrame:
    test_actual = test_forecasts["Actual"]
    return pd.DataFrame({
        "13-week test RMSE ($)": {
            name: np.sqrt(np.mean((test_actual - test_forecasts[name]) ** 2))
            for name in [*MODEL_NAMES, "Ensemble"]
        },
        "Two-year total ($)": forecasts.sum(),
        "Average weekly sales ($)": forecasts.mean(),
        "Lowest weekly sales ($)": forecasts.min(),
        "Highest weekly sales ($)": forecasts.max(),
    }).round(2)


def best_equal_weight_subset(test_forecasts: pd.DataFrame) -> tuple[str, ...]:
    """Return the lowest-RMSE equal-weight subset on the held-out test period.

    This is intentionally an exploratory diagnostic: selecting this subset on
    the final holdout is not a replacement for training-only CV selection.
    """
    actual = test_forecasts["Actual"]
    candidates = []
    for size in range(2, len(MODEL_NAMES) + 1):
        for members in combinations(MODEL_NAMES, size):
            average = test_forecasts[list(members)].mean(axis=1)
            rmse = float(np.sqrt(np.mean((actual - average) ** 2)))
            candidates.append((rmse, members))
    return min(candidates, key=lambda item: item[0])[1]


def history_matched_equal_weight_subset(
    historical_sales: pd.Series, future_forecasts: pd.DataFrame
) -> tuple[tuple[str, ...], float]:
    """Find the subset whose 104-week forecast shape most resembles history.

    This is deliberately a planning/presentation score, not an accuracy score.
    It compares level, variability, upper/lower ranges, annual peaks/troughs,
    and normalized linear trend against the observed Walmart-wide series.
    """
    history = historical_sales.to_numpy(dtype=float)
    mean = history.mean()
    std = history.std()
    q95, q05 = np.quantile(history, [0.95, 0.05])
    trend = np.polyfit(np.arange(len(history)), history, 1)[0] / mean
    history_cycles = history[:104].reshape(2, 52)
    peak = history_cycles.max(axis=1).mean()
    trough = history_cycles.min(axis=1).mean()

    candidates = []
    for size in range(2, len(MODEL_NAMES) + 1):
        for members in combinations(MODEL_NAMES, size):
            values = future_forecasts[list(members)].mean(axis=1).to_numpy(dtype=float)
            cycles = values.reshape(2, 52)
            future_trend = np.polyfit(np.arange(len(values)), values, 1)[0] / mean
            score = (
                abs(values.mean() - mean) / mean
                + abs(values.std() - std) / std
                + abs(np.quantile(values, 0.95) - q95) / q95
                + abs(np.quantile(values, 0.05) - q05) / q05
                + abs(cycles.max(axis=1).mean() - peak) / peak
                + abs(cycles.min(axis=1).mean() - trough) / trough
                + 52 * abs(future_trend - trend)
            )
            candidates.append((float(score), members))
    return min(candidates, key=lambda item: item[0])[1], min(candidates, key=lambda item: item[0])[0]


def plot_forecasts(
    train: pd.DataFrame, test_forecasts: pd.DataFrame, forecasts: pd.DataFrame
) -> None:
    """Show every historical actual, validation predictions, and future plans."""
    actual = pd.concat([train["Weekly_Sales"], test_forecasts["Actual"]])
    test_start = test_forecasts.index[0]
    cutoff = test_forecasts.index[-1]
    fig, ax = plt.subplots(figsize=(19, 8.5))

    ax.plot(
        actual.index, actual / 1e6, color="#075d82", linewidth=2.4,
        label="Actual sales (train + test)", zorder=4,
    )
    colors = {
        "SARIMAX": "#e76f51", "STL": "#8a5cf6", "Prophet": "#2a9d8f",
        "UCM": "#f4a261", "Theta": "#457b9d", "Ensemble": "#2f9e44",
    }
    for name in MODEL_NAMES:
        # This segment was made before seeing the test sales; it is the
        # validation comparison against the blue actual series.
        ax.plot(
            test_forecasts.index, test_forecasts[name] / 1e6, color=colors[name], linewidth=1.5,
            linestyle=":", alpha=0.9,
        )
        ax.plot(
            forecasts.index, forecasts[name] / 1e6, color=colors[name], linewidth=1.7,
            linestyle="--", alpha=0.88, label=name,
        )
    ax.plot(
        forecasts.index, forecasts["Ensemble"] / 1e6, color=colors["Ensemble"],
        linewidth=3.2, linestyle="--", label="Equal-weight ensemble", zorder=3,
    )
    ax.axvline(test_start, color="#6b7280", linewidth=1.2, linestyle=":", label="Test begins")
    ax.axvline(cutoff, color="#4b5563", linewidth=1.4, linestyle="--", label="2-year forecast begins")
    ax.axvspan(test_start, cutoff, color="#e5e7eb", alpha=0.4, zorder=0)
    ax.set_title(
        "Walmart Weekly Sales: Training History, Test Results + 2-Year Forecast",
        fontsize=16, weight="bold", pad=48,
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Total weekly sales ($ millions)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.grid(axis="y", alpha=0.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=4, frameon=False)
    fig.text(
        0.5, 0.01,
        "Grey band = held-out test period. Dotted lines = test forecasts made without test sales. "
        "Dashed lines = two-year forecast refit after the test period.",
        ha="center", color="#4b5563", fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / "walmart_all_models_2_year_forecast.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_best_model(
    train: pd.DataFrame, test_forecasts: pd.DataFrame, forecasts: pd.DataFrame,
    best_model: str, descriptor: str, output_filename: str,
) -> None:
    """Create the same full-history chart with one selected forecast trace."""
    actual = pd.concat([train["Weekly_Sales"], test_forecasts["Actual"]])
    test_start = test_forecasts.index[0]
    cutoff = test_forecasts.index[-1]
    color = "#2f9e44"
    fig, ax = plt.subplots(figsize=(19, 8.5))
    ax.plot(
        actual.index, actual / 1e6, color="#075d82", linewidth=2.4,
        label="Actual sales (train + test)", zorder=4,
    )
    ax.plot(
        test_forecasts.index, test_forecasts[best_model] / 1e6, color=color,
        linewidth=2, linestyle=":", label=f"{best_model} test forecast", zorder=3,
    )
    ax.plot(
        forecasts.index, forecasts[best_model] / 1e6, color=color,
        linewidth=2.8, linestyle="--", label=f"{best_model} 2-year forecast", zorder=3,
    )
    ax.axvline(test_start, color="#6b7280", linewidth=1.2, linestyle=":", label="Test begins")
    ax.axvline(cutoff, color="#4b5563", linewidth=1.4, linestyle="--", label="2-year forecast begins")
    ax.axvspan(test_start, cutoff, color="#e5e7eb", alpha=0.4, zorder=0)
    ax.set_title(
        f"Walmart Weekly Sales: {best_model} ({descriptor}) + 2-Year Forecast",
        fontsize=16, weight="bold", pad=36,
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Total weekly sales ($ millions)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.grid(axis="y", alpha=0.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False)
    fig.text(
        0.5, 0.01,
        "Grey band = held-out test period. Dotted line = test forecast made without test sales. "
        "Dashed line = two-year forecast refit after the test period.",
        ha="center", color="#4b5563", fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / output_filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-weeks", type=int, default=104, help="Forecast length; 104 is two years.")
    parser.add_argument("--test-weeks", type=int, default=DEFAULT_TEST_WEEKS,
                        help="Held-out validation weeks; default matches the notebook.")
    args = parser.parse_args()
    if args.horizon_weeks < 1 or args.test_weeks < 1:
        parser.error("--horizon-weeks and --test-weeks must be positive.")

    weekly = load_weekly_sales()
    if len(weekly) <= args.test_weeks:
        parser.error("The test period must be shorter than the available history.")
    best_params = load_best_params()
    train, test = weekly.iloc[:-args.test_weeks], weekly.iloc[-args.test_weeks:]

    # Step 1: fair test prediction -- test sales are never passed to the fit.
    print("\nCreating held-out test forecasts (training data only)...")
    test_forecasts = make_forecasts(
        train["Weekly_Sales"], train[["Holiday_Flag"]], test[["Holiday_Flag"]], best_params
    )
    test_forecasts.insert(0, "Actual", test["Weekly_Sales"])

    # Step 2: once the test is observed, use all known data for the planning forecast.
    future_dates = pd.date_range(
        weekly.index[-1] + pd.Timedelta(7, unit="D"), periods=args.horizon_weeks, freq="W-FRI"
    )
    future_x = pd.DataFrame({"Holiday_Flag": future_holiday_flags(future_dates)}, index=future_dates)
    print("\nRefitting on train + observed test data for the two-year forecast...")
    forecasts = make_forecasts(
        weekly["Weekly_Sales"], weekly[["Holiday_Flag"]], future_x, best_params
    )
    summary = create_summary(forecasts, test_forecasts)
    plot_forecasts(train, test_forecasts, forecasts)
    best_model = min(
        MODEL_NAMES,
        key=lambda name: float(np.sqrt(np.mean((test_forecasts["Actual"] - test_forecasts[name]) ** 2))),
    )
    best_subset_models = best_equal_weight_subset(test_forecasts)
    weight = 1 / len(best_subset_models)
    best_subset_label = f"{' + '.join(best_subset_models)} ({weight:.0%} each)"
    test_forecasts[best_subset_label] = test_forecasts[list(best_subset_models)].mean(axis=1)
    forecasts[best_subset_label] = forecasts[list(best_subset_models)].mean(axis=1)
    history_matched_models, history_similarity_score = history_matched_equal_weight_subset(
        weekly["Weekly_Sales"], forecasts
    )
    history_weight = 1 / len(history_matched_models)
    history_matched_label = f"{' + '.join(history_matched_models)} ({history_weight:.0%} each)"
    test_forecasts[history_matched_label] = test_forecasts[list(history_matched_models)].mean(axis=1)
    forecasts[history_matched_label] = forecasts[list(history_matched_models)].mean(axis=1)
    plot_best_model(
        train, test_forecasts, forecasts, best_model, "Best Individual Model",
        "walmart_best_model_2_year_forecast.png",
    )
    plot_best_model(
        train, test_forecasts, forecasts, "Ensemble", "Best Overall Forecast",
        "walmart_best_overall_2_year_forecast.png",
    )
    plot_best_model(
        train, test_forecasts, forecasts, best_subset_label,
        "Best Equal-Weight Ensemble", "walmart_best_equal_weight_ensemble_2_year_forecast.png",
    )
    plot_best_model(
        train, test_forecasts, forecasts, history_matched_label,
        "History-Matched Equal-Weight Ensemble",
        "walmart_history_matched_equal_weight_ensemble_2_year_forecast.png",
    )
    forecasts.to_csv(PROJECT_DIR / "walmart_2_year_forecasts.csv")
    test_forecasts.to_csv(PROJECT_DIR / "walmart_13_week_test_forecasts.csv")
    summary.to_csv(PROJECT_DIR / "walmart_2_year_forecast_summary.csv", index_label="Model")

    print(f"\nTraining period: {train.index[0]:%Y-%m-%d} to {train.index[-1]:%Y-%m-%d} ({len(train)} weeks)")
    print(f"Test period: {test.index[0]:%Y-%m-%d} to {test.index[-1]:%Y-%m-%d} ({len(test)} weeks)")
    print(f"Forecast range: {forecasts.index[0]:%Y-%m-%d} to {forecasts.index[-1]:%Y-%m-%d}")
    print("\nTwo-year forecast summary:")
    print(summary.map(lambda value: f"${value:,.0f}"))
    print("\nSaved walmart_2_year_forecasts.csv")
    print("Saved walmart_13_week_test_forecasts.csv")
    print("Saved walmart_2_year_forecast_summary.csv")
    print("Saved walmart_all_models_2_year_forecast.png")
    print(f"Saved walmart_best_model_2_year_forecast.png ({best_model}, best individual test RMSE)")
    print("Saved walmart_best_overall_2_year_forecast.png (Ensemble, best overall test RMSE)")
    print(
        "Saved walmart_best_equal_weight_ensemble_2_year_forecast.png "
        f"({best_subset_label}; exploratory best subset on the final test set)"
    )
    print(
        "Saved walmart_history_matched_equal_weight_ensemble_2_year_forecast.png "
        f"({history_matched_label}; historical shape score {history_similarity_score:.3f})"
    )


if __name__ == "__main__":
    main()
