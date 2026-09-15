"""Checks for chronological isolation and valid candidate selection."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecast_tuning import expanding_folds, tune_models


class TuningTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2010-02-05", periods=130, freq="W-FRI")
        self.y = pd.Series(np.arange(130, dtype=float), index=dates)
        self.x = pd.DataFrame({"Holiday_Flag": 0.0}, index=dates)

    def test_folds_preserve_seasons_and_time_order(self):
        folds = expanding_folds(self.y)
        self.assertEqual([len(past) for past, _ in folds], [104, 117])
        self.assertEqual([len(future) for _, future in folds], [13, 13])
        for past, future in folds:
            self.assertLess(past[-1], future[0])
        with self.assertRaisesRegex(ValueError, "Need at least"):
            expanding_folds(self.y.iloc[:-1])

    def test_selection_uses_all_folds_and_rejects_partial_failure(self):
        calls = []

        def fake_forecast(name, y, x, future, params, period, seed):
            self.assertTrue(y.index.equals(x.index))
            self.assertLess(y.index.max(), future.index.min())
            calls.append((len(y), len(future)))
            if params["offset"] == 0 and len(y) == 117:
                raise ValueError("intentional second-fold failure")
            return self.y.loc[future.index] + params["offset"]

        with patch("forecast_tuning.forecast_model", side_effect=fake_forecast):
            result = tune_models(self.y, self.x, grids={"Fake": {"offset": [0, 1, 3]}}, verbose=False)
        self.assertEqual(result.best_params, {"Fake": {"offset": 1}})
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(result.cv_predictions), 26)
        self.assertEqual(result.leaderboard.query("status == 'failed'").shape[0], 1)
        self.assertEqual(result.fold_scores.query("status == 'failed'").shape[0], 1)
        np.testing.assert_allclose(
            result.cv_predictions.Prediction - result.cv_predictions.Actual, 1.0
        )

    def test_all_failures_are_reported(self):
        with patch("forecast_tuning.forecast_model", side_effect=ValueError("invalid fit")):
            with self.assertRaisesRegex(RuntimeError, "All Fake candidates failed"):
                tune_models(self.y, self.x, grids={"Fake": {"offset": [0]}}, verbose=False)

    def test_misaligned_calendar_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must align"):
            tune_models(self.y, self.x.iloc[1:], verbose=False)


if __name__ == "__main__":
    unittest.main()
