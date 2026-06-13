"""Tests for RSNA source parsing and canonical field mapping."""

from pathlib import Path

import pandas as pd

from pneumonai.data.rsna import map_rsna_records


def test_negative_image_creates_image_record_without_boxes() -> None:
    labels = pd.DataFrame(
        [
            {
                "patientId": "negative-1",
                "x": None,
                "y": None,
                "width": None,
                "height": None,
                "Target": 0,
            }
        ]
    )
    classes = pd.DataFrame(
        [
            {
                "patientId": "negative-1",
                "class": "No Lung Opacity / Not Normal",
            }
        ]
    )

    images, boxes, rejections = map_rsna_records(
        labels,
        classes,
        Path("data/raw/rsna_pneumonia_2018/train/images"),
    )

    assert len(images) == 1
    assert images[0].to_dict() == {
        "sample_id": "negative-1",
        "image_path": (
            "data/raw/rsna_pneumonia_2018/train/images/negative-1.dcm"
        ),
        "label": 0,
        "source_dataset": "rsna_pneumonia_2018",
        "raw_class": "No Lung Opacity / Not Normal",
    }
    assert boxes == []
    assert rejections == []


def test_positive_image_creates_image_record_and_box() -> None:
    labels = pd.DataFrame(
        [
            {
                "patientId": "positive-1",
                "x": 264.0,
                "y": 152.0,
                "width": 213.0,
                "height": 379.0,
                "Target": 1,
            }
        ]
    )
    classes = pd.DataFrame(
        [{"patientId": "positive-1", "class": "Lung Opacity"}]
    )

    images, boxes, rejections = map_rsna_records(
        labels,
        classes,
        Path("data/raw/rsna_pneumonia_2018/train/images"),
    )

    assert len(images) == 1
    assert images[0].label == 1
    assert images[0].raw_class == "Lung Opacity"
    assert len(boxes) == 1
    assert boxes[0].to_dict() == {
        "sample_id": "positive-1",
        "x": 264.0,
        "y": 152.0,
        "width": 213.0,
        "height": 379.0,
    }
    assert rejections == []


def test_multiple_annotations_create_one_image_and_multiple_boxes() -> None:
    labels = pd.DataFrame(
        [
            {
                "patientId": "positive-1",
                "x": 264.0,
                "y": 152.0,
                "width": 213.0,
                "height": 379.0,
                "Target": 1,
            },
            {
                "patientId": "positive-1",
                "x": 562.0,
                "y": 152.0,
                "width": 256.0,
                "height": 453.0,
                "Target": 1,
            },
        ]
    )
    classes = pd.DataFrame(
        [
            {"patientId": "positive-1", "class": "Lung Opacity"},
            {"patientId": "positive-1", "class": "Lung Opacity"},
        ]
    )

    images, boxes, rejections = map_rsna_records(
        labels,
        classes,
        Path("data/raw/rsna_pneumonia_2018/train/images"),
    )

    assert len(images) == 1
    assert len(boxes) == 2
    assert [box.sample_id for box in boxes] == [
        "positive-1",
        "positive-1",
    ]
    assert [(box.x, box.y) for box in boxes] == [
        (264.0, 152.0),
        (562.0, 152.0),
    ]
    assert rejections == []


def test_conflicting_targets_reject_the_patient_group() -> None:
    labels = pd.DataFrame(
        [
            {
                "patientId": "conflict-1",
                "x": None,
                "y": None,
                "width": None,
                "height": None,
                "Target": 0,
            },
            {
                "patientId": "conflict-1",
                "x": 10.0,
                "y": 20.0,
                "width": 30.0,
                "height": 40.0,
                "Target": 1,
            },
        ]
    )
    classes = pd.DataFrame(
        [{"patientId": "conflict-1", "class": "Lung Opacity"}]
    )

    images, boxes, rejections = map_rsna_records(
        labels,
        classes,
        Path("data/raw/rsna_pneumonia_2018/train/images"),
    )

    assert images == []
    assert boxes == []
    assert [record.to_dict() for record in rejections] == [
        {
            "sample_id": "conflict-1",
            "reason": "conflicting_targets",
        }
    ]


def test_incomplete_positive_box_rejects_the_patient_group() -> None:
    labels = pd.DataFrame(
        [
            {
                "patientId": "incomplete-1",
                "x": 10.0,
                "y": 20.0,
                "width": None,
                "height": 40.0,
                "Target": 1,
            }
        ]
    )
    classes = pd.DataFrame(
        [{"patientId": "incomplete-1", "class": "Lung Opacity"}]
    )

    images, boxes, rejections = map_rsna_records(
        labels,
        classes,
        Path("data/raw/rsna_pneumonia_2018/train/images"),
    )

    assert images == []
    assert boxes == []
    assert rejections[0].reason == "incomplete_box"
