"""Tests for dataset validation and rejection rules."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pneumonai.data.schema import BoundingBoxRecord, ImageRecord
from pneumonai.data.validation import validate_records


def _image(path: Path, *, sample_id: str = "sample-1") -> ImageRecord:
    return ImageRecord(
        sample_id=sample_id,
        image_path=str(path),
        label=1,
        source_dataset="rsna_pneumonia_2018",
        raw_class="Lung Opacity",
    )


def _box(*, sample_id: str = "sample-1") -> BoundingBoxRecord:
    return BoundingBoxRecord(
        sample_id=sample_id,
        x=10.0,
        y=20.0,
        width=30.0,
        height=40.0,
    )


def test_readable_dicom_and_in_bounds_box_are_accepted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "sample-1.dcm"
    image_path.touch()
    monkeypatch.setattr(
        "pneumonai.data.validation.pydicom.dcmread",
        lambda path: SimpleNamespace(
            pixel_array=np.zeros((100, 100), dtype=np.uint8)
        ),
    )

    images, boxes, rejections, duplicates = validate_records(
        [_image(image_path)],
        [_box()],
    )

    assert len(images) == 1
    assert len(boxes) == 1
    assert rejections == []
    assert duplicates == 0


def test_missing_image_is_rejected(tmp_path: Path) -> None:
    images, boxes, rejections, duplicates = validate_records(
        [_image(tmp_path / "missing.dcm")],
        [_box()],
    )

    assert images == []
    assert boxes == []
    assert rejections[0].reason == "image_file_not_found"
    assert duplicates == 0


def test_unreadable_dicom_is_rejected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "sample-1.dcm"
    image_path.touch()

    def fail_to_read(path: Path) -> None:
        raise ValueError("corrupt DICOM")

    monkeypatch.setattr(
        "pneumonai.data.validation.pydicom.dcmread",
        fail_to_read,
    )

    images, boxes, rejections, _ = validate_records(
        [_image(image_path)],
        [_box()],
    )

    assert images == []
    assert boxes == []
    assert rejections[0].reason == "unreadable_dicom"


def test_out_of_bounds_box_rejects_complete_image(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "sample-1.dcm"
    image_path.touch()
    monkeypatch.setattr(
        "pneumonai.data.validation.pydicom.dcmread",
        lambda path: SimpleNamespace(
            pixel_array=np.zeros((50, 50), dtype=np.uint8)
        ),
    )

    images, boxes, rejections, _ = validate_records(
        [_image(image_path)],
        [_box()],
    )

    assert images == []
    assert boxes == []
    assert rejections[0].reason == "box_outside_image"


def test_unknown_box_reference_is_rejected() -> None:
    images, boxes, rejections, _ = validate_records(
        [],
        [_box(sample_id="unknown")],
    )

    assert images == []
    assert boxes == []
    assert rejections[0].to_dict() == {
        "sample_id": "unknown",
        "reason": "unknown_image_reference",
    }


def test_duplicate_box_is_removed_and_counted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "sample-1.dcm"
    image_path.touch()
    monkeypatch.setattr(
        "pneumonai.data.validation.pydicom.dcmread",
        lambda path: SimpleNamespace(
            pixel_array=np.zeros((100, 100), dtype=np.uint8)
        ),
    )
    box = _box()

    images, boxes, rejections, duplicates = validate_records(
        [_image(image_path)],
        [box, box],
    )

    assert len(images) == 1
    assert boxes == [box]
    assert rejections == []
    assert duplicates == 1
