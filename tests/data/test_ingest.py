"""Tests for RSNA ingestion orchestration and canonical outputs."""

from pathlib import Path

import pandas as pd

from pneumonai.data.ingest import ingest_rsna
from pneumonai.data.schema import (
    BoundingBoxRecord,
    ImageRecord,
    RejectedRecord,
)


def test_ingest_writes_valid_records_and_builds_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    labels_path = tmp_path / "labels.csv"
    classes_path = tmp_path / "classes.csv"
    images_output_path = tmp_path / "metadata" / "images.csv"
    boxes_output_path = tmp_path / "metadata" / "bounding_boxes.csv"
    pd.DataFrame(
        [
            {"patientId": "positive-1"},
            {"patientId": "positive-1"},
            {"patientId": "negative-1"},
        ]
    ).to_csv(labels_path, index=False)
    pd.DataFrame(
        [{"patientId": "positive-1", "class": "Lung Opacity"}]
    ).to_csv(classes_path, index=False)

    mapped_images = [
        ImageRecord(
            sample_id="positive-1",
            image_path="positive-1.dcm",
            label=1,
            source_dataset="rsna_pneumonia_2018",
            raw_class="Lung Opacity",
        )
    ]
    mapped_boxes = [
        BoundingBoxRecord(
            sample_id="positive-1",
            x=10.0,
            y=20.0,
            width=30.0,
            height=40.0,
        )
    ]
    mapping_rejections = [
        RejectedRecord("negative-1", "missing_class")
    ]

    monkeypatch.setattr(
        "pneumonai.data.ingest.map_rsna_records",
        lambda labels, classes, image_directory: (
            mapped_images,
            mapped_boxes,
            mapping_rejections,
        ),
    )
    monkeypatch.setattr(
        "pneumonai.data.ingest.validate_records",
        lambda images, boxes, rejections: (
            images,
            boxes,
            rejections,
            2,
        ),
    )

    report = ingest_rsna(
        labels_path,
        classes_path,
        tmp_path / "images",
        images_output_path,
        boxes_output_path,
    )

    images_output = pd.read_csv(images_output_path)
    boxes_output = pd.read_csv(boxes_output_path)

    assert images_output.to_dict("records") == [
        mapped_images[0].to_dict()
    ]
    assert boxes_output.to_dict("records") == [
        mapped_boxes[0].to_dict()
    ]
    assert report.source_rows == 3
    assert report.accepted_images == 1
    assert report.positive_images == 1
    assert report.negative_images == 0
    assert report.bounding_boxes == 1
    assert report.duplicate_boxes_removed == 2
    assert report.rejected_records == 1
    assert report.rejection_reasons == {"missing_class": 1}


def test_ingest_writes_headers_when_no_boxes_are_valid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    labels_path = tmp_path / "labels.csv"
    classes_path = tmp_path / "classes.csv"
    pd.DataFrame([{"patientId": "negative-1"}]).to_csv(
        labels_path,
        index=False,
    )
    pd.DataFrame(
        [{"patientId": "negative-1", "class": "Normal"}]
    ).to_csv(classes_path, index=False)
    image = ImageRecord(
        sample_id="negative-1",
        image_path="negative-1.dcm",
        label=0,
        source_dataset="rsna_pneumonia_2018",
        raw_class="Normal",
    )

    monkeypatch.setattr(
        "pneumonai.data.ingest.map_rsna_records",
        lambda labels, classes, image_directory: ([image], [], []),
    )
    monkeypatch.setattr(
        "pneumonai.data.ingest.validate_records",
        lambda images, boxes, rejections: (images, [], [], 0),
    )

    boxes_output_path = tmp_path / "metadata" / "bounding_boxes.csv"
    report = ingest_rsna(
        labels_path,
        classes_path,
        tmp_path / "images",
        tmp_path / "metadata" / "images.csv",
        boxes_output_path,
    )

    boxes_output = pd.read_csv(boxes_output_path)

    assert list(boxes_output.columns) == [
        "sample_id",
        "x",
        "y",
        "width",
        "height",
    ]
    assert boxes_output.empty
    assert report.negative_images == 1
