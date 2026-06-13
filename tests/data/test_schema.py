"""Tests for canonical dataset record definitions."""

from dataclasses import FrozenInstanceError

import pytest

from pneumonai.data.schema import (
    BoundingBoxRecord,
    ImageRecord,
    IngestionReport,
    RejectedRecord,
)


def test_image_record_construction() -> None:
    record = ImageRecord(
        sample_id="sample-1",
        image_path=(
            "data/raw/rsna_pneumonia_2018/train/images/sample-1.dcm"
        ),
        label=1,
        source_dataset="rsna_pneumonia_2018",
        raw_class="Lung Opacity",
    )

    assert record.sample_id == "sample-1"
    assert record.label == 1
    assert record.raw_class == "Lung Opacity"


def test_image_record_uses_integer_label_contract() -> None:
    assert ImageRecord.__annotations__["label"] is int


def test_image_record_serialization() -> None:
    record = ImageRecord(
        sample_id="sample-1",
        image_path=(
            "data/raw/rsna_pneumonia_2018/train/images/sample-1.dcm"
        ),
        label=0,
        source_dataset="rsna_pneumonia_2018",
        raw_class="Normal",
    )

    assert record.to_dict() == {
        "sample_id": "sample-1",
        "image_path": (
            "data/raw/rsna_pneumonia_2018/train/images/sample-1.dcm"
        ),
        "label": 0,
        "source_dataset": "rsna_pneumonia_2018",
        "raw_class": "Normal",
    }


def test_image_record_is_immutable() -> None:
    record = ImageRecord(
        sample_id="sample-1",
        image_path="data/raw/sample-1.dcm",
        label=0,
        source_dataset="rsna_pneumonia_2018",
        raw_class="Normal",
    )

    with pytest.raises(FrozenInstanceError):
        record.label = 1


def test_bounding_box_record_construction_and_serialization() -> None:
    record = BoundingBoxRecord(
        sample_id="sample-1",
        x=264.0,
        y=152.0,
        width=213.0,
        height=379.0,
    )

    assert record.to_dict() == {
        "sample_id": "sample-1",
        "x": 264.0,
        "y": 152.0,
        "width": 213.0,
        "height": 379.0,
    }


def test_bounding_box_record_is_immutable() -> None:
    record = BoundingBoxRecord(
        sample_id="sample-1",
        x=264.0,
        y=152.0,
        width=213.0,
        height=379.0,
    )

    with pytest.raises(FrozenInstanceError):
        record.width = 100.0


def test_rejected_record_construction_and_serialization() -> None:
    record = RejectedRecord(
        sample_id="sample-1",
        reason="conflicting_targets",
    )

    assert record.to_dict() == {
        "sample_id": "sample-1",
        "reason": "conflicting_targets",
    }


def test_ingestion_report_construction() -> None:
    report = IngestionReport(
        source_rows=30_227,
        accepted_images=26_684,
        positive_images=6_012,
        negative_images=20_672,
        bounding_boxes=9_555,
        duplicate_boxes_removed=0,
        rejected_records=0,
        rejection_reasons={},
    )

    assert report.accepted_images == 26_684
    assert report.positive_images + report.negative_images == 26_684
    assert report.rejection_reasons == {}


def test_ingestion_report_is_immutable() -> None:
    report = IngestionReport(
        source_rows=30_227,
        accepted_images=26_684,
        positive_images=6_012,
        negative_images=20_672,
        bounding_boxes=9_555,
        duplicate_boxes_removed=0,
        rejected_records=0,
        rejection_reasons={},
    )

    with pytest.raises(FrozenInstanceError):
        report.rejected_records = 1
