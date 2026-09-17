"""Unit tests for ML models, baselines, and cascaded pipeline."""

import os
import shutil
import pandas as pd
import pytest

from src.data.generator import CommunityDatasetGenerator
from src.features.pipeline import FeatureExtractor, FACILITY_LIST, DAY_NAMES
from src.features.temporal_splitter import split_dataset_chronologically, SplitBoundaries
from src.models.cascaded_pipeline import CascadedPredictionPipeline
from src.models.baselines import HabitualBaseline, GlobalPopularityBaseline


@pytest.fixture(scope="module")
def prepared_data():
    gen = CommunityDatasetGenerator(num_residents=30, num_days=60, seed=77)
    df = gen.generate()
    # Boundaries tailored for 60-day dataset
    boundaries = SplitBoundaries(
        train_end_date="2026-02-15 23:59:59",
        val_end_date="2026-02-24 23:59:59"
    )
    train_df, val_df, test_df = split_dataset_chronologically(df, boundaries=boundaries)

    extractor = FeatureExtractor()
    extractor.fit_global_priors(train_df)
    X_train, y_train, meta_train, hist = extractor.extract_features(train_df)
    X_val, y_val, meta_val, _ = extractor.extract_features(val_df, initial_history=hist)

    return train_df, X_train, y_train, meta_train, X_val, y_val, meta_val


def test_cascaded_pipeline_fit_predict(prepared_data):
    _, X_train, y_train, meta_train, X_val, y_val, meta_val = prepared_data

    # Use smaller estimators for fast test execution
    pipeline = CascadedPredictionPipeline(random_state=42)
    pipeline.facility_model.set_params(n_estimators=10)
    pipeline.day_model.set_params(n_estimators=10)
    pipeline.hour_model.set_params(n_estimators=10)
    pipeline.lead_time_model.set_params(n_estimators=10)

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val, meta_val)

    assert len(preds) == len(X_val)
    assert len(preds) > 0
    expected_cols = {
        "pred_facility",
        "pred_usage_day",
        "pred_usage_hour",
        "pred_usage_time",
        "pred_nudge_str",
    }
    assert expected_cols.issubset(set(preds.columns))

    # Check bounds
    assert preds["pred_facility"].isin(FACILITY_LIST).all()
    assert preds["pred_usage_day"].isin(DAY_NAMES).all()
    assert (preds["pred_usage_hour"] >= 6).all()
    assert (preds["pred_usage_hour"] <= 23).all()
    assert preds["pred_nudge_str"].str.startswith("Nudge ").all()


def test_baselines_execution(prepared_data):
    train_df, _, _, _, _, _, meta_val = prepared_data

    # Test HabitualBaseline
    hab_base = HabitualBaseline()
    hab_base.fit(train_df)
    hab_preds = hab_base.predict(meta_val)
    assert len(hab_preds) == len(meta_val)
    assert len(hab_preds) > 0
    assert hab_preds["pred_facility"].isin(FACILITY_LIST).all()

    # Test GlobalPopularityBaseline
    glob_base = GlobalPopularityBaseline()
    glob_base.fit(train_df)
    glob_preds = glob_base.predict(meta_val)
    assert len(glob_preds) == len(meta_val)
    assert len(glob_preds) > 0


def test_model_serialization(prepared_data, tmp_path):
    _, X_train, y_train, meta_train, X_val, _, meta_val = prepared_data

    pipeline = CascadedPredictionPipeline(random_state=42)
    pipeline.facility_model.set_params(n_estimators=10)
    pipeline.day_model.set_params(n_estimators=10)
    pipeline.hour_model.set_params(n_estimators=10)
    pipeline.lead_time_model.set_params(n_estimators=10)

    pipeline.fit(X_train, y_train)
    preds_before = pipeline.predict(X_val, meta_val)

    save_dir = str(tmp_path / "test_artifacts")
    pipeline.save(save_dir)

    new_pipeline = CascadedPredictionPipeline()
    new_pipeline.load(save_dir)
    preds_after = new_pipeline.predict(X_val, meta_val)

    pd.testing.assert_frame_equal(preds_before, preds_after)
