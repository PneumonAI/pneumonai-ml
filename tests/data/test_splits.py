"""Tests for deterministic, stratified dataset splitting."""

import pandas as pd
import pytest

from pneumonai.data.splits import create_splits


def _images(rows_per_class: int = 100) -> pd.DataFrame:
    rows = []
    for label in (0, 1):
        for index in range(rows_per_class):
            sample_id = f"class-{label}-{index:03d}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": f"images/{sample_id}.dcm",
                    "label": label,
                    "source_dataset": "rsna_pneumonia_2018",
                    "raw_class": (
                        "Lung Opacity" if label == 1 else "Normal"
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_create_splits_is_deterministic_and_complete() -> None:
    images = _images()

    first = create_splits(images, 0.70, 0.15, 0.15, seed=42)
    second = create_splits(images, 0.70, 0.15, 0.15, seed=42)

    for first_split, second_split in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(first_split, second_split)

    train, validation, test = first
    assert len(train) == 140
    assert len(validation) == 30
    assert len(test) == 30
    assert set(pd.concat(first)["sample_id"]) == set(images["sample_id"])


def test_create_splits_are_disjoint_and_stratified() -> None:
    train, validation, test = create_splits(
        _images(),
        0.70,
        0.15,
        0.15,
        seed=42,
    )

    train_ids = set(train["sample_id"])
    validation_ids = set(validation["sample_id"])
    test_ids = set(test["sample_id"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    for split in (train, validation, test):
        assert split["label"].value_counts().to_dict() == {
            0: len(split) // 2,
            1: len(split) // 2,
        }
        assert split["sample_id"].is_monotonic_increasing


@pytest.mark.parametrize(
    ("train_ratio", "validation_ratio", "test_ratio"),
    [
        (0.70, 0.15, 0.10),
        (0.0, 0.50, 0.50),
        (0.70, -0.10, 0.40),
        (float("nan"), 0.50, 0.50),
    ],
)
def test_create_splits_rejects_invalid_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    with pytest.raises(ValueError):
        create_splits(
            _images(),
            train_ratio,
            validation_ratio,
            test_ratio,
            seed=42,
        )


def test_create_splits_rejects_duplicate_sample_ids() -> None:
    images = _images()
    images.loc[1, "sample_id"] = images.loc[0, "sample_id"]

    with pytest.raises(ValueError, match="must be unique"):
        create_splits(images, 0.70, 0.15, 0.15, seed=42)


def test_create_splits_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        create_splits(
            pd.DataFrame({"sample_id": ["sample-1"]}),
            0.70,
            0.15,
            0.15,
            seed=42,
        )


def test_create_splits_rejects_non_binary_labels() -> None:
    images = _images()
    images.loc[0, "label"] = 2

    with pytest.raises(ValueError, match="binary integers"):
        create_splits(images, 0.70, 0.15, 0.15, seed=42)


def test_create_splits_rejects_dataset_too_small_for_stratification() -> None:
    with pytest.raises(ValueError, match="too few samples"):
        create_splits(
            _images(rows_per_class=3),
            0.70,
            0.15,
            0.15,
            seed=42,
        )
