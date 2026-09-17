"""In-depth error analysis and cohort segmentation for facility usage prediction."""

from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.features.pipeline import FACILITY_LIST


class ErrorAnalyzer:
    """Performs deep failure mode analysis across user engagement cohorts and drift phases."""

    @staticmethod
    def analyze_by_engagement_tier(
        X_df: pd.DataFrame, eval_details: pd.DataFrame
    ) -> pd.DataFrame:
        """Breaks down performance by user history volume (Cold-start, Medium, Power users)."""
        df = pd.DataFrame()
        hist_count = X_df["user_history_count"].values

        tiers = []
        for c in hist_count:
            if c < 3:
                tiers.append("Cold-Start (<3 bookings)")
            elif c <= 15:
                tiers.append("Regular (3-15 bookings)")
            else:
                tiers.append("Power User (>15 bookings)")

        df["tier"] = tiers
        df["fac_match"] = eval_details["fac_match"]
        df["day_match"] = eval_details["day_match"]
        df["hour_match"] = eval_details["hour_match"]
        df["nudge_match"] = eval_details["nudge_match"]
        df["exact_4_of_4"] = (eval_details["total_matched"] == 4).astype(int)
        df["avg_matched"] = eval_details["total_matched"]

        grouped = df.groupby("tier").agg(
            sample_count=("fac_match", "count"),
            facility_accuracy=("fac_match", "mean"),
            day_accuracy=("day_match", "mean"),
            hour_accuracy=("hour_match", "mean"),
            nudge_accuracy=("nudge_match", "mean"),
            exact_4_of_4_rate=("exact_4_of_4", "mean"),
            avg_outputs_matched=("avg_matched", "mean"),
        ).round(3)

        return grouped.reset_index()

    @staticmethod
    def facility_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
        """Computes labeled confusion matrix across facilities."""
        labels = sorted(list(set(y_true.unique()).union(set(y_pred.unique()))))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return pd.DataFrame(cm, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
