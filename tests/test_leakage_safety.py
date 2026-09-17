"""Automated tests verifying zero temporal lookahead and zero target leakage."""

import pandas as pd
import pytest

from src.data.generator import CommunityDatasetGenerator
from src.features.temporal_splitter import split_dataset_chronologically
from src.features.pipeline import FeatureExtractor


@pytest.fixture(scope="module")
def split_data():
    gen = CommunityDatasetGenerator(num_residents=40, num_days=90, seed=99)
    df = gen.generate()
    train_df, val_df, test_df = split_dataset_chronologically(
        df, timestamp_col="booking_timestamp"
    )
    return df, train_df, val_df, test_df


def test_chronological_split_monotonicity(split_data):
    _, train_df, val_df, test_df = split_data
    if len(train_df) > 0 and len(val_df) > 0:
        max_train = pd.to_datetime(train_df["booking_timestamp"]).max()
        min_val = pd.to_datetime(val_df["booking_timestamp"]).min()
        assert max_train <= min_val, f"Train ({max_train}) leaks into Val ({min_val})"

    if len(val_df) > 0 and len(test_df) > 0:
        max_val = pd.to_datetime(val_df["booking_timestamp"]).max()
        min_test = pd.to_datetime(test_df["booking_timestamp"]).min()
        assert max_val <= min_test, f"Val ({max_val}) leaks into Test ({min_test})"


def test_no_target_columns_in_features(split_data):
    _, train_df, _, _ = split_data
    extractor = FeatureExtractor()
    extractor.fit_global_priors(train_df)
    X_train, y_train, _, _ = extractor.extract_features(train_df)

    forbidden_cols = {
        "target_facility",
        "target_usage_day",
        "target_usage_hour",
        "target_lead_time_hours",
        "lead_time_hours",
        "usage_timestamp",
    }
    assert not forbidden_cols.intersection(set(X_train.columns)), "Target column leaked into X_train!"


def test_future_independence_leakage_assertion(split_data):
    """Modifying future records must NOT change features of prior records."""
    df, train_df, _, _ = split_data
    extractor = FeatureExtractor()
    extractor.fit_global_priors(train_df)

    # Extract on first 100 rows
    subset_100 = df.iloc[:100].copy()
    X_100, _, _, _ = extractor.extract_features(subset_100)

    # Extract on 200 rows where rows 100..200 have perturbed values
    subset_200 = df.iloc[:200].copy()
    # Perturb rows 100..200
    subset_200.loc[100:, "facility_id"] = "Hall"

    X_200, _, _, _ = extractor.extract_features(subset_200)

    # The features of the first 100 rows in X_200 must be EXACTLY identical to X_100
    pd.testing.assert_frame_equal(X_100, X_200.iloc[:100])


def test_zero_nans_in_features(split_data):
    _, train_df, _, _ = split_data
    extractor = FeatureExtractor()
    extractor.fit_global_priors(train_df)
    X_train, _, _, _ = extractor.extract_features(train_df)

    nan_count = X_train.isna().sum().sum()
    assert nan_count == 0, f"Found {nan_count} NaNs in feature matrix!"
