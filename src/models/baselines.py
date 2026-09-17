"""Baseline prediction models for benchmark comparison against the ML pipeline.

Author: Prajwal Rao
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.features.pipeline import DAY_NAMES, FACILITY_LIST


class HabitualBaseline:
    """Predicts each resident's most frequent historical facility, day, hour, and median lead time."""

    def __init__(self):
        self.user_modes: Dict[str, dict] = {}
        self.global_mode_facility = "Gym"
        self.global_mode_day = "Mon"
        self.global_mode_hour = 18
        self.global_median_lead = 24.0

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fits habitual modes from training data."""
        if len(train_df) == 0:
            return

        u_dt = pd.to_datetime(train_df["usage_timestamp"])
        b_dt = pd.to_datetime(train_df["booking_timestamp"])
        lead_series = (u_dt - b_dt).dt.total_seconds() / 3600.0

        # Global modes
        self.global_mode_facility = train_df["facility_id"].mode()[0]
        self.global_mode_day = DAY_NAMES[u_dt.dt.weekday.mode()[0]]
        self.global_mode_hour = int(u_dt.dt.hour.mode()[0])
        self.global_median_lead = float(lead_series.median())

        df_work = train_df.copy()
        df_work["_u_dt"] = u_dt
        df_work["_lead"] = lead_series

        # User-specific modes
        for rid, group in df_work.groupby("resident_id"):
            fac_mode = group["facility_id"].mode()[0]
            day_mode = DAY_NAMES[group["_u_dt"].dt.weekday.mode()[0]]
            hour_mode = int(group["_u_dt"].dt.hour.mode()[0])
            lead_median = float(group["_lead"].median())

            self.user_modes[rid] = {
                "facility": fac_mode,
                "usage_day": day_mode,
                "usage_hour": hour_mode,
                "lead_time": lead_median,
            }

    def predict(self, meta_df: pd.DataFrame) -> pd.DataFrame:
        """Generates predictions for a dataframe of records."""
        if len(meta_df) == 0:
            return pd.DataFrame(
                columns=[
                    "pred_facility",
                    "pred_usage_day",
                    "pred_usage_hour",
                    "pred_usage_time",
                    "pred_lead_time_hours",
                    "pred_nudge_str",
                    "pred_nudge_dt",
                ]
            )

        preds = []
        for row in meta_df.itertuples():
            rid = row.resident_id
            mode = self.user_modes.get(
                rid,
                {
                    "facility": self.global_mode_facility,
                    "usage_day": self.global_mode_day,
                    "usage_hour": self.global_mode_hour,
                    "lead_time": self.global_median_lead,
                },
            )

            u_dt = pd.to_datetime(row.usage_timestamp)
            lead_h = mode["lead_time"]
            nudge_dt = u_dt - timedelta(hours=lead_h)
            nudge_day = DAY_NAMES[nudge_dt.weekday()]
            nudge_hm = nudge_dt.strftime("%H:%M")

            preds.append(
                {
                    "pred_facility": mode["facility"],
                    "pred_usage_day": mode["usage_day"],
                    "pred_usage_hour": mode["usage_hour"],
                    "pred_usage_time": f"{mode['usage_hour']:02d}:00",
                    "pred_lead_time_hours": lead_h,
                    "pred_nudge_str": f"Nudge {nudge_day} / {nudge_hm}",
                    "pred_nudge_dt": nudge_dt,
                }
            )

        return pd.DataFrame(preds)


class GlobalPopularityBaseline:
    """Predicts community-wide top facility and peak hours for all residents."""

    def __init__(self):
        self.facility = "Gym"
        self.usage_day = "Mon"
        self.usage_hour = 18
        self.lead_time = 24.0

    def fit(self, train_df: pd.DataFrame) -> None:
        if len(train_df) == 0:
            return
        u_dt = pd.to_datetime(train_df["usage_timestamp"])
        b_dt = pd.to_datetime(train_df["booking_timestamp"])
        leads = (u_dt - b_dt).dt.total_seconds() / 3600.0

        self.facility = train_df["facility_id"].mode()[0]
        self.usage_day = DAY_NAMES[u_dt.dt.weekday.mode()[0]]
        self.usage_hour = int(u_dt.dt.hour.mode()[0])
        self.lead_time = float(leads.median())

    def predict(self, meta_df: pd.DataFrame) -> pd.DataFrame:
        if len(meta_df) == 0:
            return pd.DataFrame(
                columns=[
                    "pred_facility",
                    "pred_usage_day",
                    "pred_usage_hour",
                    "pred_usage_time",
                    "pred_lead_time_hours",
                    "pred_nudge_str",
                    "pred_nudge_dt",
                ]
            )
        preds = []
        for row in meta_df.itertuples():
            u_dt = pd.to_datetime(row.usage_timestamp)
            nudge_dt = u_dt - timedelta(hours=self.lead_time)
            nudge_day = DAY_NAMES[nudge_dt.weekday()]
            nudge_hm = nudge_dt.strftime("%H:%M")
            preds.append(
                {
                    "pred_facility": self.facility,
                    "pred_usage_day": self.usage_day,
                    "pred_usage_hour": self.usage_hour,
                    "pred_usage_time": f"{self.usage_hour:02d}:00",
                    "pred_lead_time_hours": self.lead_time,
                    "pred_nudge_str": f"Nudge {nudge_day} / {nudge_hm}",
                    "pred_nudge_dt": nudge_dt,
                }
            )
        return pd.DataFrame(preds)
