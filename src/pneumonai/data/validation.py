"""Validation rules for canonical records and DICOM source files."""

from collections import Counter, defaultdict
from math import isfinite
from pathlib import Path
from typing import Any

import pydicom

from .schema import BoundingBoxRecord, ImageRecord, RejectedRecord


def validate_records(
    images: list[ImageRecord],
    boxes: list[BoundingBoxRecord],
    rejections: list[RejectedRecord] | None = None,
) -> tuple[
    list[ImageRecord],
    list[BoundingBoxRecord],
    list[RejectedRecord],
    int,
]:
    """Return records backed by readable DICOM files and valid boxes."""
    rejected_records = list(rejections or [])
    image_id_counts = Counter(image.sample_id for image in images)
    known_image_ids = set(image_id_counts)
    image_dimensions: dict[str, tuple[int, int]] = {}
    candidate_images: list[ImageRecord] = []

    for image in images:
        if image_id_counts[image.sample_id] > 1:
            if not _is_rejected(
                rejected_records,
                image.sample_id,
                "duplicate_image_record",
            ):
                rejected_records.append(
                    RejectedRecord(
                        image.sample_id,
                        "duplicate_image_record",
                    )
                )
            continue

        image_path = Path(image.image_path)
        if not image_path.is_file():
            rejected_records.append(
                RejectedRecord(image.sample_id, "image_file_not_found")
            )
            continue

        try:
            dicom = pydicom.dcmread(image_path)
            pixels = dicom.pixel_array
        except Exception:
            rejected_records.append(
                RejectedRecord(image.sample_id, "unreadable_dicom")
            )
            continue

        if pixels.ndim != 2 or pixels.size == 0:
            rejected_records.append(
                RejectedRecord(image.sample_id, "invalid_image_dimensions")
            )
            continue

        height, width = pixels.shape
        if height <= 0 or width <= 0:
            rejected_records.append(
                RejectedRecord(image.sample_id, "invalid_image_dimensions")
            )
            continue

        candidate_images.append(image)
        image_dimensions[image.sample_id] = (int(width), int(height))

    unique_boxes: list[BoundingBoxRecord] = []
    seen_boxes: set[BoundingBoxRecord] = set()
    duplicate_boxes_removed = 0
    boxes_by_sample: defaultdict[str, list[BoundingBoxRecord]] = defaultdict(
        list
    )

    for box in boxes:
        if box in seen_boxes:
            duplicate_boxes_removed += 1
            continue
        seen_boxes.add(box)

        if box.sample_id not in known_image_ids:
            rejected_records.append(
                RejectedRecord(box.sample_id, "unknown_image_reference")
            )
            continue

        unique_boxes.append(box)
        boxes_by_sample[box.sample_id].append(box)

    rejected_image_ids: set[str] = set()
    for image in candidate_images:
        width, height = image_dimensions[image.sample_id]
        image_boxes = boxes_by_sample.get(image.sample_id, [])

        if image.label == 0 and image_boxes:
            rejected_records.append(
                RejectedRecord(
                    image.sample_id,
                    "unexpected_box_for_negative",
                )
            )
            rejected_image_ids.add(image.sample_id)
            continue

        invalid_reason = _find_invalid_box_reason(
            image_boxes,
            image_width=width,
            image_height=height,
        )
        if invalid_reason is not None:
            rejected_records.append(
                RejectedRecord(image.sample_id, invalid_reason)
            )
            rejected_image_ids.add(image.sample_id)

    valid_images = [
        image
        for image in candidate_images
        if image.sample_id not in rejected_image_ids
    ]
    valid_image_ids = {image.sample_id for image in valid_images}
    valid_boxes = [
        box for box in unique_boxes if box.sample_id in valid_image_ids
    ]

    return (
        valid_images,
        valid_boxes,
        rejected_records,
        duplicate_boxes_removed,
    )


def validate_rsna_records(
    labels: Any,
    classes: Any,
    image_directory: Path,
) -> tuple[
    list[ImageRecord],
    list[BoundingBoxRecord],
    list[RejectedRecord],
    int,
]:
    """Map RSNA tables, then validate their canonical records."""
    from .rsna import map_rsna_records

    images, boxes, rejections = map_rsna_records(
        labels,
        classes,
        image_directory,
    )
    return validate_records(images, boxes, rejections)


def _find_invalid_box_reason(
    boxes: list[BoundingBoxRecord],
    *,
    image_width: int,
    image_height: int,
) -> str | None:
    for box in boxes:
        values = (box.x, box.y, box.width, box.height)
        if not all(isfinite(value) for value in values):
            return "non_finite_box"
        if box.x < 0 or box.y < 0:
            return "invalid_box_coordinates"
        if box.width <= 0 or box.height <= 0:
            return "invalid_box_dimensions"
        if box.x + box.width > image_width:
            return "box_outside_image"
        if box.y + box.height > image_height:
            return "box_outside_image"
    return None


def _is_rejected(
    rejections: list[RejectedRecord],
    sample_id: str,
    reason: str,
) -> bool:
    return any(
        rejection.sample_id == sample_id and rejection.reason == reason
        for rejection in rejections
    )
