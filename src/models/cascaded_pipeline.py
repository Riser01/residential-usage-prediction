"""Cascaded Multi-Target Prediction Pipeline using LightGBM for facility, day, hour, and nudge timing.

Author: Prajwal Rao
Predicts the 4 target variables sequentially:
1. Facility (Multi-class GBDT)
2. Usage Day (Multi-class GBDT conditioned on facility context)
3. Usage Hour (Multi-class GBDT conditioned on facility and day context)
4. Lead Time / Nudge Timing (Quantile GBDT regressor, alpha=0.30)
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.pipeline import DAY_NAMES, FACILITY_LIST


class CascadedPredictionPipeline:
    """Cascaded multi-stage model pipeline predicting:

    Stage 1: Facility (Multi-class classification)
    Stage 2: Usage Day (Multi-class classification conditioned on Stage 1)
    Stage 3: Usage Hour (Multi-class classification conditioned on Stages 1 & 2)
    Stage 4: Lead Time / Nudge Timing (Quantile regression conditioned on predicted slot)
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.facility_classes = sorted(FACILITY_LIST)
        self.day_classes = list(range(7))  # 0 = Mon, ..., 6 = Sun
        self.hour_classes = list(range(6, 24))  # 6..23 operating hours

        # Stage 1: Facility Classifier (Regularized for multi-class generalization)
        self.facility_model = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.06,
            num_leaves=25,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.80,
            random_state=random_state,
            verbosity=-1,
        )

        # Stage 2: Usage Day Classifier (Conditioned on facility context)
        self.day_model = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.06,
            num_leaves=25,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.80,
            random_state=random_state,
            verbosity=-1,
        )

        # Stage 3: Usage Hour Classifier (Conditioned on facility & day context)
        self.hour_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.07,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            verbosity=-1,
        )

        # Stage 4: Lead Time Regressor (Quantile alpha=0.30 to guarantee proactive nudge)
        self.lead_time_model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.30,
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=25,
            subsample=0.85,
            colsample_bytree=0.80,
            random_state=random_state,
            verbosity=-1,
        )

        self.categorical_columns = ["last_facility", "user_top_facility"]
        self.cat_mappings: Dict[str, Dict[str, int]] = {}
        self.facility_to_int: Dict[str, int] = {f: idx for idx, f in enumerate(self.facility_classes)}
        self.is_fitted = False

    def _prepare_base_features(self, X: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """Encodes categoricals and ensures clean numerical input."""
        X_out = X.copy()
        for col in self.categorical_columns:
            if col in X_out.columns:
                if is_train:
                    unique_vals = sorted(list(X_out[col].astype(str).unique()))
                    self.cat_mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}
                mapping = self.cat_mappings.get(col, {})
                X_out[col] = X_out[col].astype(str).map(lambda v: mapping.get(v, -1))
        return X_out

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Fits all four stages sequentially."""
        X_base = self._prepare_base_features(X_train, is_train=True)

        y_fac = y_train["target_facility"].astype(str)
        y_day = y_train["target_usage_day_int"].astype(int)
        y_hour = y_train["target_usage_hour"].astype(int)
        y_lead = y_train["target_lead_time_hours"].astype(float)

        # Stage 1: Facility Multi-Class
        print("Training Stage 1: Facility Classifier...")
        self.facility_model.fit(X_base, y_fac)

        # Stage 2: Usage Day Multi-Class (Conditioned on facility features)
        print("Training Stage 2: Usage Day Classifier...")
        X_stage2 = X_base.copy()
        y_fac_int = y_fac.map(lambda f: self.facility_to_int.get(f, 0))
        X_stage2["context_facility"] = y_fac_int
        self.day_model.fit(X_stage2, y_day)

        # Stage 3: Usage Hour Multi-Class (Conditioned on facility & day)
        print("Training Stage 3: Usage Hour Classifier...")
        X_stage3 = X_stage2.copy()
        X_stage3["context_day"] = y_day
        self.hour_model.fit(X_stage3, y_hour)

        # Stage 4: Lead Time Quantile Regressor (alpha=0.30)
        print("Training Stage 4: Lead Time Quantile Regressor (alpha=0.30)...")
        X_stage4 = X_stage3.copy()
        X_stage4["context_hour"] = y_hour
        self.lead_time_model.fit(X_stage4, y_lead)

        self.is_fitted = True
        print("Cascaded Pipeline Training Completed!")

    def predict(self, X: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
        """Inference across the cascaded pipeline."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before predict.")

        X_base = self._prepare_base_features(X, is_train=False)

        # Stage 1: Facility
        pred_fac = self.facility_model.predict(X_base)
        pred_fac_int = [self.facility_to_int.get(str(f), 0) for f in pred_fac]

        # Stage 2: Usage Day (Conditioned on predicted facility)
        X_stage2 = X_base.copy()
        X_stage2["context_facility"] = pred_fac_int
        pred_day_int = self.day_model.predict(X_stage2)
        pred_day_str = [DAY_NAMES[int(d)] for d in pred_day_int]

        # Stage 3: Usage Hour (Conditioned on predicted facility & day)
        X_stage3 = X_stage2.copy()
        X_stage3["context_day"] = pred_day_int
        pred_hour = self.hour_model.predict(X_stage3)

        # Stage 4: Lead Time (Conditioned on predicted facility, day, and hour)
        X_stage4 = X_stage3.copy()
        X_stage4["context_hour"] = pred_hour
        pred_lead_hours = self.lead_time_model.predict(X_stage4)
        pred_lead_hours = np.maximum(1.0, pred_lead_hours)

        # Format output predictions
        results = []
        for idx in range(len(X)):
            fac = str(pred_fac[idx])
            day_str = pred_day_str[idx]
            hour_val = int(pred_hour[idx])
            lead_h = float(pred_lead_hours[idx])
            use_time_str = f"{hour_val:02d}:00"

            # Compute predicted notification (nudge) time
            actual_usage_dt = pd.to_datetime(meta_df["usage_timestamp"].iloc[idx])
            day_diff = (int(pred_day_int[idx]) - actual_usage_dt.weekday()) % 7
            pred_usage_dt = actual_usage_dt.replace(hour=hour_val, minute=0, second=0) + timedelta(days=day_diff)
            nudge_dt = pred_usage_dt - timedelta(hours=lead_h)

            # Round to clean 5-minute interval
            nudge_minute = int(round(nudge_dt.minute / 5.0) * 5) % 60
            nudge_dt = nudge_dt.replace(minute=nudge_minute, second=0)

            nudge_day_str = DAY_NAMES[nudge_dt.weekday()]
            nudge_time_fmt = nudge_dt.strftime("%H:%M")
            nudge_str = f"Nudge {nudge_day_str} / {nudge_time_fmt}"

            results.append(
                {
                    "pred_facility": fac,
                    "pred_usage_day": day_str,
                    "pred_usage_day_int": int(pred_day_int[idx]),
                    "pred_usage_hour": hour_val,
                    "pred_usage_time": use_time_str,
                    "pred_lead_time_hours": round(lead_h, 2),
                    "pred_nudge_str": nudge_str,
                    "pred_nudge_dt": nudge_dt,
                }
            )

        return pd.DataFrame(results)

    def save(self, dir_path: str = "models/artifacts") -> None:
        """Serializes model artifacts to disk."""
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self.facility_model, os.path.join(dir_path, "facility_model.joblib"))
        joblib.dump(self.day_model, os.path.join(dir_path, "day_model.joblib"))
        joblib.dump(self.hour_model, os.path.join(dir_path, "hour_model.joblib"))
        joblib.dump(self.lead_time_model, os.path.join(dir_path, "lead_time_model.joblib"))
        joblib.dump(self.cat_mappings, os.path.join(dir_path, "cat_mappings.joblib"))
        print(f"Saved all 4 model artifacts to {dir_path}")

    def load(self, dir_path: str = "models/artifacts") -> None:
        """Loads serialized model artifacts from disk."""
        self.facility_model = joblib.load(os.path.join(dir_path, "facility_model.joblib"))
        self.day_model = joblib.load(os.path.join(dir_path, "day_model.joblib"))
        self.hour_model = joblib.load(os.path.join(dir_path, "hour_model.joblib"))
        self.lead_time_model = joblib.load(os.path.join(dir_path, "lead_time_model.joblib"))
        self.cat_mappings = joblib.load(os.path.join(dir_path, "cat_mappings.joblib"))
        self.is_fitted = True
        print(f"Loaded all 4 model artifacts from {dir_path}")
