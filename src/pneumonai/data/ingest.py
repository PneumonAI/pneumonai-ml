"""Reproducible orchestration for canonical dataset ingestion."""

from collections import Counter
from pathlib import Path

import pandas as pd

from .rsna import map_rsna_records
from .schema import BoundingBoxRecord, ImageRecord, IngestionReport
from .validation import validate_records

IMAGE_COLUMNS = [
    "sample_id",
    "image_path",
    "label",
    "source_dataset",
    "raw_class",
]
BOUNDING_BOX_COLUMNS = [
    "sample_id",
    "x",
    "y",
    "width",
    "height",
]


def ingest_rsna(
    labels_path: Path,
    classes_path: Path,
    image_directory: Path,
    images_output_path: Path,
    boxes_output_path: Path,
) -> IngestionReport:
    """Ingest RSNA source data and write validated canonical metadata."""
    labels = pd.read_csv(labels_path)
    classes = pd.read_csv(classes_path)

    image_records, bounding_box_records, mapping_rejections = (
        map_rsna_records(
            labels,
            classes,
            image_directory,
        )
    )
    (
        valid_image_records,
        valid_bounding_box_records,
        rejected_records,
        duplicate_boxes_removed,
    ) = validate_records(
        image_records,
        bounding_box_records,
        mapping_rejections,
    )

    _write_records(
        valid_image_records,
        images_output_path,
        columns=IMAGE_COLUMNS,
    )
    _write_records(
        valid_bounding_box_records,
        boxes_output_path,
        columns=BOUNDING_BOX_COLUMNS,
    )

    rejection_reasons = Counter(
        record.reason for record in rejected_records
    )
    positive_images = sum(
        record.label == 1 for record in valid_image_records
    )
    negative_images = sum(
        record.label == 0 for record in valid_image_records
    )

    return IngestionReport(
        source_rows=len(labels),
        accepted_images=len(valid_image_records),
        positive_images=positive_images,
        negative_images=negative_images,
        bounding_boxes=len(valid_bounding_box_records),
        duplicate_boxes_removed=duplicate_boxes_removed,
        rejected_records=len(rejected_records),
        rejection_reasons=dict(rejection_reasons),
    )


def _write_records(
    records: list[ImageRecord] | list[BoundingBoxRecord],
    output_path: Path,
    *,
    columns: list[str],
) -> None:
    """Write canonical records with deterministic column ordering."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(
        (record.to_dict() for record in records),
        columns=columns,
    )
    dataframe.to_csv(output_path, index=False)
