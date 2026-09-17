"""Chronological temporal train/validation/test splitter ensuring zero future lookahead."""

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import pandas as pd


@dataclass
class SplitBoundaries:
    train_end_date: str = "2026-04-30 23:59:59"  # Months 1-4 (Jan - Apr)
    val_end_date: str = "2026-05-31 23:59:59"    # Month 5 (May)
    # Test set is strictly Month 6 (June 1 - June 30, 2026)


def split_dataset_chronologically(
    df: pd.DataFrame,
    boundaries: SplitBoundaries = SplitBoundaries(),
    timestamp_col: str = "booking_timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits dataset chronologically based on booking timestamp.

    Returns:
        (train_df, val_df, test_df)
    """
    df_sorted = df.copy()
    df_sorted["_split_dt"] = pd.to_datetime(df_sorted[timestamp_col])
    df_sorted = df_sorted.sort_values("_split_dt").reset_index(drop=True)

    t_train_end = pd.to_datetime(boundaries.train_end_date)
    t_val_end = pd.to_datetime(boundaries.val_end_date)

    train_mask = df_sorted["_split_dt"] <= t_train_end
    val_mask = (df_sorted["_split_dt"] > t_train_end) & (df_sorted["_split_dt"] <= t_val_end)
    test_mask = df_sorted["_split_dt"] > t_val_end

    train_df = df_sorted[train_mask].drop(columns=["_split_dt"]).reset_index(drop=True)
    val_df = df_sorted[val_mask].drop(columns=["_split_dt"]).reset_index(drop=True)
    test_df = df_sorted[test_mask].drop(columns=["_split_dt"]).reset_index(drop=True)

    return train_df, val_df, test_df
