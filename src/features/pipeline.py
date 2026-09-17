"""Leakage-safe feature extraction pipeline for facility booking prediction.

Author: Prajwal Rao
Extracts causal historical features strictly prior to each prediction cutoff,
guaranteeing zero future data leakage and zero target leakage.
"""

import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.community_config import COMMUNITY_FACILITIES

FACILITY_LIST = list(COMMUNITY_FACILITIES.keys())
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class FeatureExtractor:
    """Causal, stateful feature extractor ensuring zero lookahead and zero target leakage."""

    def __init__(self, bayes_prior_weight: float = 3.0):
        self.bayes_prior_weight = bayes_prior_weight
        self.global_facility_priors: Dict[str, float] = {}
        self.global_facility_median_lead: Dict[str, float] = {}
        self.is_fitted = False

    def fit_global_priors(self, train_df: pd.DataFrame) -> None:
        """Fits global population priors strictly on training data to prevent data leakage."""
        total = len(train_df)
        counts = train_df["facility_id"].value_counts().to_dict()
        self.global_facility_priors = {
            f: counts.get(f, 1) / max(1, total) for f in FACILITY_LIST
        }

        # Calculate lead times dynamically from timestamps
        u_dt = pd.to_datetime(train_df["usage_timestamp"])
        b_dt = pd.to_datetime(train_df["booking_timestamp"])
        leads = (u_dt - b_dt).dt.total_seconds() / 3600.0

        temp_df = pd.DataFrame({"facility_id": train_df["facility_id"], "lead": leads})
        lead_medians = temp_df.groupby("facility_id")["lead"].median().to_dict()
        self.global_facility_median_lead = {
            f: float(lead_medians.get(f, 24.0)) for f in FACILITY_LIST
        }
        self.is_fitted = True

    def extract_features(
        self, df: pd.DataFrame, initial_history: Optional[Dict[str, List[dict]]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[dict]]]:
        """Extracts leakage-free tabular features and targets."""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted with fit_global_priors before extracting features.")

        df_sorted = df.copy()
        df_sorted["booking_dt_parsed"] = pd.to_datetime(df_sorted["booking_timestamp"])
        df_sorted["usage_dt_parsed"] = pd.to_datetime(df_sorted["usage_timestamp"])
        df_sorted = df_sorted.sort_values("booking_dt_parsed").reset_index(drop=True)

        user_history: Dict[str, List[dict]] = {}
        if initial_history is not None:
            user_history = {k: list(v) for k, v in initial_history.items()}

        feature_rows: List[dict] = []
        target_rows: List[dict] = []
        metadata_rows: List[dict] = []

        M = self.bayes_prior_weight
        priors = self.global_facility_priors

        for row in df_sorted.itertuples():
            rid = row.resident_id
            b_dt: datetime = row.booking_dt_parsed.to_pydatetime()
            u_dt: datetime = row.usage_dt_parsed.to_pydatetime()
            lead_h = (u_dt - b_dt).total_seconds() / 3600.0

            if rid not in user_history:
                user_history[rid] = []

            hist = user_history[rid]
            n_hist = len(hist)

            # 1. User activity volume & Cutoff context
            cut_dow = b_dt.weekday()
            cut_hour = b_dt.hour
            cut_month = b_dt.month

            feat = {
                "user_history_count": n_hist,
                "is_cold_start": 1 if n_hist < 3 else 0,
                "cut_dow": cut_dow,
                "cut_hour": cut_hour,
                "cut_month": cut_month,
                "is_cut_weekend": 1 if cut_dow in [5, 6] else 0,
            }

            # 2. Sequential & Recency features
            if n_hist > 0:
                last_event = hist[-1]
                feat["days_since_last_booking"] = (b_dt - last_event["booking_dt"]).total_seconds() / 86400.0
                feat["days_since_last_usage"] = (b_dt - last_event["usage_dt"]).total_seconds() / 86400.0
                feat["last_facility"] = last_event["facility"]
                feat["last_usage_dow"] = last_event["usage_dt"].weekday()
                feat["last_usage_hour"] = last_event["usage_dt"].hour
                feat["last_lead_time_hours"] = last_event["lead_time_hours"]
            else:
                feat["days_since_last_booking"] = 30.0
                feat["days_since_last_usage"] = 30.0
                feat["last_facility"] = "None"
                feat["last_usage_dow"] = cut_dow
                feat["last_usage_hour"] = 18
                feat["last_lead_time_hours"] = 24.0

            # 3. User Facility Preference with Empirical Bayes Shrinkage
            fac_counts = {f: 0 for f in FACILITY_LIST}
            dow_counts = {d: 0 for d in range(7)}
            hour_counts = []
            lead_times = []

            for h in hist:
                fac_counts[h["facility"]] = fac_counts.get(h["facility"], 0) + 1
                dow_counts[h["usage_dt"].weekday()] += 1
                hour_counts.append(h["usage_dt"].hour)
                lead_times.append(h["lead_time_hours"])

            entropy = 0.0
            top_fac = "None"
            max_c = -1
            for f in FACILITY_LIST:
                c = fac_counts[f]
                smoothed_ratio = (c + M * priors.get(f, 0.16)) / (n_hist + M)
                feat[f"user_fac_ratio_{f}"] = smoothed_ratio
                entropy -= smoothed_ratio * math.log(smoothed_ratio + 1e-9)
                if c > max_c:
                    max_c = c
                    top_fac = f

            feat["user_facility_entropy"] = entropy
            feat["user_top_facility"] = top_fac if n_hist > 0 else "Gym"

            # 4. Day-of-week distribution
            top_dow = 0
            max_d_c = -1
            for d in range(7):
                dow_ratio = (dow_counts[d] + 1.0) / (n_hist + 7.0)
                feat[f"user_dow_ratio_{d}"] = dow_ratio
                if dow_counts[d] > max_d_c:
                    max_d_c = dow_counts[d]
                    top_dow = d
            feat["user_top_dow"] = top_dow if n_hist > 0 else cut_dow

            # 5. Usage hour distributions
            if n_hist > 0:
                feat["user_mean_hour"] = float(np.mean(hour_counts))
                morning = sum(1 for h in hour_counts if h < 12) / n_hist
                afternoon = sum(1 for h in hour_counts if 12 <= h < 17) / n_hist
                evening = sum(1 for h in hour_counts if h >= 17) / n_hist
                weekend = (dow_counts[5] + dow_counts[6]) / n_hist
                feat["user_median_lead_hours"] = float(np.median(lead_times))
            else:
                feat["user_mean_hour"] = 12.0
                morning = 0.33
                afternoon = 0.33
                evening = 0.34
                weekend = 0.28
                feat["user_median_lead_hours"] = 24.0

            feat["user_morning_ratio"] = morning
            feat["user_afternoon_ratio"] = afternoon
            feat["user_evening_ratio"] = evening
            feat["user_weekend_ratio"] = weekend

            feature_rows.append(feat)

            # Targets
            target_dow_str = DAY_NAMES[u_dt.weekday()]
            target_rows.append(
                {
                    "target_facility": row.facility_id,
                    "target_usage_day": target_dow_str,
                    "target_usage_day_int": u_dt.weekday(),
                    "target_usage_hour": u_dt.hour,
                    "target_lead_time_hours": lead_h,
                    "target_usage_timestamp": row.usage_timestamp,
                    "target_booking_timestamp": row.booking_timestamp,
                }
            )

            # Format past bookings for PDF §3.5 table
            past_bookings_fmt = []
            for h in hist[-3:]:
                h_day = DAY_NAMES[h["usage_dt"].weekday()]
                h_use_time = h["usage_dt"].strftime("%H:%M")
                h_book_time = f"{DAY_NAMES[h['booking_dt'].weekday()]} {h['booking_dt'].strftime('%H:%M')}"
                past_bookings_fmt.append(f"{h['facility']} / {h_day} / {h_use_time} / {h_book_time}")

            metadata_rows.append(
                {
                    "resident_id": rid,
                    "booking_timestamp": row.booking_timestamp,
                    "usage_timestamp": row.usage_timestamp,
                    "past_bookings_summary": "\n".join(past_bookings_fmt) if past_bookings_fmt else "No prior history (Cold-start)",
                }
            )

            # Update historical state
            hist.append(
                {
                    "facility": row.facility_id,
                    "booking_dt": b_dt,
                    "usage_dt": u_dt,
                    "lead_time_hours": lead_h,
                }
            )

        X_df = pd.DataFrame(feature_rows)
        y_df = pd.DataFrame(target_rows)
        meta_df = pd.DataFrame(metadata_rows)

        return X_df, y_df, meta_df, user_history
