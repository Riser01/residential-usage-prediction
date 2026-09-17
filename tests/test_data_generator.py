"""Unit tests for synthetic dataset generator and physical domain constraints."""

from datetime import datetime
import pandas as pd
import pytest

from src.data.generator import CommunityDatasetGenerator
from src.data.community_config import COMMUNITY_FACILITIES


@pytest.fixture(scope="module")
def sample_dataset() -> pd.DataFrame:
    generator = CommunityDatasetGenerator(num_residents=50, num_days=60, seed=123)
    return generator.generate()


def test_schema_and_non_empty(sample_dataset: pd.DataFrame):
    required_cols = {"resident_id", "facility_id", "booking_timestamp", "usage_timestamp"}
    assert required_cols.issubset(set(sample_dataset.columns))
    assert len(sample_dataset) > 500


def test_physical_causality_booking_before_usage(sample_dataset: pd.DataFrame):
    booking_dt = pd.to_datetime(sample_dataset["booking_timestamp"])
    usage_dt = pd.to_datetime(sample_dataset["usage_timestamp"])
    # Strictly booking must precede usage
    diff_minutes = (usage_dt - booking_dt).dt.total_seconds() / 60.0
    assert (diff_minutes > 0).all(), "Found booking timestamp >= usage timestamp!"
    # Lead time should be at least 30 minutes
    assert (diff_minutes >= 30).all(), "Found booking lead time < 30 minutes!"


def test_facility_operating_hours(sample_dataset: pd.DataFrame):
    usage_dt = pd.to_datetime(sample_dataset["usage_timestamp"])
    usage_hours = usage_dt.dt.hour

    for facility_name, cfg in COMMUNITY_FACILITIES.items():
        fac_mask = sample_dataset["facility_id"] == facility_name
        if fac_mask.any():
            hours = usage_hours[fac_mask]
            assert hours.min() >= cfg.open_hour, f"{facility_name} usage before opening hour {cfg.open_hour}"
            assert hours.max() <= cfg.close_hour, f"{facility_name} usage after closing hour {cfg.close_hour}"


def test_no_double_booking_same_resident_same_hour(sample_dataset: pd.DataFrame):
    # Check no duplicate (resident_id, usage_timestamp)
    duplicates = sample_dataset.duplicated(subset=["resident_id", "usage_timestamp"])
    assert not duplicates.any(), f"Found {duplicates.sum()} double bookings for same resident at same hour!"


def test_facility_popularity_imbalance(sample_dataset: pd.DataFrame):
    counts = sample_dataset["facility_id"].value_counts(normalize=True)
    # Gym and Pool should account for more than 40% of bookings combined
    gym_pool_share = counts.get("Gym", 0) + counts.get("Pool", 0)
    assert gym_pool_share > 0.40, f"Gym + Pool share too low: {gym_pool_share}"
    # Multipurpose Hall should be rare (< 10%)
    hall_share = counts.get("Hall", 0)
    assert hall_share < 0.10, f"Hall share too high: {hall_share}"
