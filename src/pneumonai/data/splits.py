"""Deterministic, stratified dataset splitting."""

from math import isclose, isfinite

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {"sample_id", "label"}
VALID_LABELS = {0, 1}


def create_splits(
    images: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return deterministic, stratified train, validation, and test sets."""
    _validate_inputs(
        images,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    train_df, remaining_df = train_test_split(
        images,
        test_size=validation_ratio + test_ratio,
        random_state=seed,
        stratify=images["label"],
    )
    test_share_of_remaining = test_ratio / (
        validation_ratio + test_ratio
    )
    validation_df, test_df = train_test_split(
        remaining_df,
        test_size=test_share_of_remaining,
        random_state=seed,
        stratify=remaining_df["label"],
    )

    splits = tuple(
        split.sort_values("sample_id").reset_index(drop=True)
        for split in (train_df, validation_df, test_df)
    )
    _verify_splits(images, *splits)
    return splits


def _validate_inputs(
    images: pd.DataFrame,
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    missing_columns = REQUIRED_COLUMNS - set(images.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")
    if images.empty:
        raise ValueError("Cannot split an empty dataset.")
    if images["sample_id"].isna().any():
        raise ValueError("sample_id values must not be missing.")
    if images["sample_id"].astype(str).str.strip().eq("").any():
        raise ValueError("sample_id values must not be empty.")
    if images["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be unique before splitting.")
    if images["label"].isna().any():
        raise ValueError("label values must not be missing.")

    labels = set(images["label"].unique())
    if not labels.issubset(VALID_LABELS):
        raise ValueError("label values must be binary integers: 0 or 1.")
    if labels != VALID_LABELS:
        raise ValueError("Both label classes are required for stratification.")

    ratios = (train_ratio, validation_ratio, test_ratio)
    if not all(isfinite(ratio) and 0 < ratio < 1 for ratio in ratios):
        raise ValueError("Each split ratio must be finite and between 0 and 1.")
    if not isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1.")

    class_counts = images["label"].value_counts()
    remaining_ratio = validation_ratio + test_ratio
    for label, count in class_counts.items():
        remaining_count = round(count * remaining_ratio)
        if remaining_count < 2:
            raise ValueError(
                f"Label {label} has too few samples for validation and test."
            )


def _verify_splits(
    source: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    source_ids = set(source["sample_id"])
    train_ids = set(train["sample_id"])
    validation_ids = set(validation["sample_id"])
    test_ids = set(test["sample_id"])

    if train_ids & validation_ids or train_ids & test_ids:
        raise RuntimeError("Generated dataset splits overlap.")
    if validation_ids & test_ids:
        raise RuntimeError("Generated dataset splits overlap.")
    if train_ids | validation_ids | test_ids != source_ids:
        raise RuntimeError("Generated dataset splits do not cover the source.")
    if len(train) + len(validation) + len(test) != len(source):
        raise RuntimeError("Generated dataset splits changed the row count.")
