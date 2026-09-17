"""Unit tests for evaluation metrics, circular clock calculations, and error analyzer."""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_circular_hour_mae, PredictionEvaluator
from src.evaluation.error_analysis import ErrorAnalyzer


def test_circular_hour_mae():
    # 23:00 to 01:00 is 2 hours circular distance
    y_true = np.array([23, 12, 0])
    y_pred = np.array([1, 14, 23])
    mae = compute_circular_hour_mae(y_true, y_pred)
    # diffs: |23-1|=22 -> min(22, 2) = 2; |12-14|=2 -> 2; |0-23|=23 -> min(23, 1) = 1
    # mean = (2 + 2 + 1) / 3 = 1.666...
    assert pytest.approx(mae, 0.01) == 5.0 / 3.0


def test_prediction_evaluator_exact_and_partial_matches():
    # Create synthetic test pairs
    now = datetime(2026, 6, 1, 18, 0, 0)
    y_true_df = pd.DataFrame(
        [
            {
                "target_facility": "Gym",
                "target_usage_day": "Fri",
                "target_usage_hour": 7,
                "target_lead_time_hours": 12.0,
                "target_booking_timestamp": (now - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S"),
            },
            {
                "target_facility": "Gym",
                "target_usage_day": "Fri",
                "target_usage_hour": 19,
                "target_lead_time_hours": 24.0,
                "target_booking_timestamp": (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S"),
            },
        ]
    )

    y_pred_df = pd.DataFrame(
        [
            # Exact match row (4 of 4)
            {
                "pred_facility": "Gym",
                "pred_usage_day": "Fri",
                "pred_usage_hour": 7,
                "pred_lead_time_hours": 12.0,
                "pred_nudge_dt": now - timedelta(hours=12),
            },
            # Partial match row (facility matches, day matches, hour differs 19 vs 20, nudge differs)
            {
                "pred_facility": "Gym",
                "pred_usage_day": "Fri",
                "pred_usage_hour": 20,
                "pred_lead_time_hours": 48.0,
                "pred_nudge_dt": now - timedelta(hours=48),
            },
        ]
    )

    evaluator = PredictionEvaluator(nudge_tolerance_minutes=60)
    results = evaluator.evaluate(y_true_df, y_pred_df, pd.DataFrame())

    summary = results["summary"]
    details = results["details"]

    assert summary["facility_accuracy"] == 1.0
    assert summary["usage_day_accuracy"] == 1.0
    assert summary["usage_hour_exact_accuracy"] == 0.5
    assert summary["exact_4_of_4_match_rate"] == 0.5

    # Check match labels per PDF format
    assert "YES\n4 of 4" in details["match_label"].iloc[0]
    assert "NO\n2 of 4" in details["match_label"].iloc[1]
