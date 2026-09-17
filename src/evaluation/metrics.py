"""Comprehensive evaluation metrics and matching engine per Anacity specifications."""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

from src.features.pipeline import DAY_NAMES, FACILITY_LIST


def compute_circular_hour_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes circular MAE for 24-hour periodic clock."""
    diff = np.abs(y_true - y_pred)
    circ_diff = np.minimum(diff, 24.0 - diff)
    return float(np.mean(circ_diff))


class PredictionEvaluator:
    """Calculates all per-output metrics, composite match scores, and error diagnostics."""

    def __init__(self, nudge_tolerance_minutes: int = 60):
        self.nudge_tolerance_minutes = nudge_tolerance_minutes

    def evaluate(
        self,
        y_true_df: pd.DataFrame,
        y_pred_df: pd.DataFrame,
        meta_df: pd.DataFrame,
    ) -> Dict[str, any]:
        """Evaluates predictions against holdout ground truth.

        Returns comprehensive metrics dictionary and sample-level match records.
        """
        n_samples = len(y_true_df)
        if n_samples == 0:
            return {"error": "Empty evaluation set"}

        # -------------------------------------------------------------
        # Output 1: Facility Metrics
        # -------------------------------------------------------------
        y_true_fac = y_true_df["target_facility"].astype(str).values
        y_pred_fac = y_pred_df["pred_facility"].astype(str).values

        fac_acc = accuracy_score(y_true_fac, y_pred_fac)
        fac_f1_macro = f1_score(y_true_fac, y_pred_fac, average="macro", zero_division=0)
        fac_f1_weighted = f1_score(y_true_fac, y_pred_fac, average="weighted", zero_division=0)

        fac_match = (y_true_fac == y_pred_fac).astype(int)

        # -------------------------------------------------------------
        # Output 2: Usage Day Metrics
        # -------------------------------------------------------------
        y_true_day = y_true_df["target_usage_day"].astype(str).values
        y_pred_day = y_pred_df["pred_usage_day"].astype(str).values

        day_acc = accuracy_score(y_true_day, y_pred_day)
        day_f1_macro = f1_score(y_true_day, y_pred_day, average="macro", zero_division=0)
        day_match = (y_true_day == y_pred_day).astype(int)

        # -------------------------------------------------------------
        # Output 3: Usage Hour Metrics
        # -------------------------------------------------------------
        y_true_hour = y_true_df["target_usage_hour"].astype(int).values
        y_pred_hour = y_pred_df["pred_usage_hour"].astype(int).values

        hour_exact_acc = accuracy_score(y_true_hour, y_pred_hour)
        hour_within_1_acc = np.mean(np.abs(y_true_hour - y_pred_hour) <= 1)
        hour_mae = float(np.mean(np.abs(y_true_hour - y_pred_hour)))
        hour_circ_mae = compute_circular_hour_mae(y_true_hour, y_pred_hour)

        hour_match = (y_true_hour == y_pred_hour).astype(int)

        # -------------------------------------------------------------
        # Output 4: Notification (Nudge) Time Metrics
        # -------------------------------------------------------------
        # Actual booking datetime
        actual_book_dts = pd.to_datetime(y_true_df["target_booking_timestamp"]).dt.to_pydatetime()
        pred_nudge_dts = pd.to_datetime(y_pred_df["pred_nudge_dt"]).dt.to_pydatetime()

        lead_mae = float(np.mean(np.abs(
            y_true_df["target_lead_time_hours"].values - y_pred_df["pred_lead_time_hours"].values
        )))

        # Nudge timing match:
        # Match if day of nudge matches actual booking day AND time is within tolerance window (e.g. 60 min)
        nudge_match = []
        is_proactive = []
        time_diff_hours = []

        for idx in range(n_samples):
            act_b_dt = actual_book_dts[idx]
            pr_n_dt = pred_nudge_dts[idx]

            # Time delta between actual booking and predicted nudge
            delta_min = abs((pr_n_dt - act_b_dt).total_seconds()) / 60.0
            time_diff_hours.append(delta_min / 60.0)

            # Proactive: nudge arrived before or within 15 min of actual booking
            proactive = pr_n_dt <= (act_b_dt + timedelta(minutes=15))
            is_proactive.append(1 if proactive else 0)

            # Day and time window match
            same_day = (pr_n_dt.weekday() == act_b_dt.weekday())
            window_ok = (delta_min <= self.nudge_tolerance_minutes)
            nudge_match.append(1 if (same_day and window_ok) else 0)

        nudge_match = np.array(nudge_match)
        nudge_window_acc = float(np.mean(nudge_match))
        proactive_rate = float(np.mean(is_proactive))

        # -------------------------------------------------------------
        # Overall Composite Measures (PDF §3.4 & §3.5)
        # -------------------------------------------------------------
        total_matched_outputs = fac_match + day_match + hour_match + nudge_match
        exact_4_of_4 = (total_matched_outputs == 4).astype(int)
        partial_3_plus = (total_matched_outputs >= 3).astype(int)
        partial_2_plus = (total_matched_outputs >= 2).astype(int)

        match_labels = []
        for m in total_matched_outputs:
            if m == 4:
                match_labels.append("YES\n4 of 4")
            else:
                match_labels.append(f"NO\n{m} of 4")

        summary_metrics = {
            "n_samples": n_samples,
            # Per-output
            "facility_accuracy": round(fac_acc, 4),
            "facility_macro_f1": round(fac_f1_macro, 4),
            "facility_weighted_f1": round(fac_f1_weighted, 4),
            "usage_day_accuracy": round(day_acc, 4),
            "usage_day_macro_f1": round(day_f1_macro, 4),
            "usage_hour_exact_accuracy": round(hour_exact_acc, 4),
            "usage_hour_within_1hr_accuracy": round(hour_within_1_acc, 4),
            "usage_hour_mae": round(hour_mae, 2),
            "usage_hour_circular_mae": round(hour_circ_mae, 2),
            "lead_time_mae_hours": round(lead_mae, 2),
            "nudge_window_accuracy": round(nudge_window_acc, 4),
            "nudge_proactive_actionability_rate": round(proactive_rate, 4),
            # Overall Composite
            "exact_4_of_4_match_rate": round(float(np.mean(exact_4_of_4)), 4),
            "partial_3_plus_match_rate": round(float(np.mean(partial_3_plus)), 4),
            "partial_2_plus_match_rate": round(float(np.mean(partial_2_plus)), 4),
            "average_outputs_matched": round(float(np.mean(total_matched_outputs)), 2),
        }

        detailed_matches = pd.DataFrame(
            {
                "fac_match": fac_match,
                "day_match": day_match,
                "hour_match": hour_match,
                "nudge_match": nudge_match,
                "total_matched": total_matched_outputs,
                "match_label": match_labels,
                "is_proactive": is_proactive,
            }
        )

        return {"summary": summary_metrics, "details": detailed_matches}
