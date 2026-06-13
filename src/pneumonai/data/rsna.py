"""RSNA-specific source CSV parsing and field mapping.

This module translates RSNA source fields into canonical records. It must not
contain generic pipeline orchestration or model preprocessing.
"""

from pathlib import Path

import pandas as pd

from .schema import BoundingBoxRecord, ImageRecord, RejectedRecord

BOX_COLUMNS = ["x", "y", "width", "height"]
EXPECTED_TARGETS = {
    "Lung Opacity": 1,
    "Normal": 0,
    "No Lung Opacity / Not Normal": 0,
}


def map_rsna_records(
    labels: pd.DataFrame,
    classes: pd.DataFrame,
    image_directory: Path,
) -> tuple[
    list[ImageRecord],
    list[BoundingBoxRecord],
    list[RejectedRecord],
]:
    """Parse RSNA CSV files into canonical records."""
    unique_classes = classes.drop_duplicates(
        subset=["patientId", "class"]
    )
    class_counts = unique_classes.groupby("patientId")["class"].nunique()
    conflicting_class_ids = set(class_counts[class_counts > 1].index)
    mergeable_classes = unique_classes[
        ~unique_classes["patientId"].isin(conflicting_class_ids)
    ]

    result_df = labels.merge(
        mergeable_classes,
        on="patientId",
        how="left",
        validate="many_to_one",
    )

    image_records: list[ImageRecord] = []
    bounding_box_records: list[BoundingBoxRecord] = []
    rejected_records = [
        RejectedRecord(str(patient_id), "conflicting_classes")
        for patient_id in sorted(conflicting_class_ids)
    ]

    for patient_id, patient_rows in result_df.groupby("patientId"):
        sample_id = str(patient_id)
        if patient_id in conflicting_class_ids:
            continue

        targets = patient_rows["Target"].dropna().unique()
        raw_classes = patient_rows["class"].dropna().unique()

        if len(raw_classes) == 0:
            rejected_records.append(
                RejectedRecord(sample_id, "missing_class")
            )
            continue
        if len(raw_classes) > 1:
            rejected_records.append(
                RejectedRecord(sample_id, "conflicting_classes")
            )
            continue
        if len(targets) == 0:
            rejected_records.append(
                RejectedRecord(sample_id, "missing_target")
            )
            continue
        if len(targets) > 1:
            rejected_records.append(
                RejectedRecord(sample_id, "conflicting_targets")
            )
            continue

        target = targets[0]
        raw_class = str(raw_classes[0])
        if target not in (0, 1):
            rejected_records.append(
                RejectedRecord(sample_id, "invalid_target")
            )
            continue
        if EXPECTED_TARGETS.get(raw_class) != int(target):
            rejected_records.append(
                RejectedRecord(sample_id, "class_target_mismatch")
            )
            continue

        box_values = patient_rows[BOX_COLUMNS]
        if int(target) == 0 and box_values.notna().any(axis=None):
            rejected_records.append(
                RejectedRecord(
                    sample_id,
                    "unexpected_box_for_negative",
                )
            )
            continue

        if int(target) == 1:
            if box_values.isna().all(axis=1).any():
                rejected_records.append(
                    RejectedRecord(sample_id, "missing_box_for_positive")
                )
                continue
            if box_values.isna().any(axis=1).any():
                rejected_records.append(
                    RejectedRecord(sample_id, "incomplete_box")
                )
                continue
            if (
                (box_values["width"] <= 0).any()
                or (box_values["height"] <= 0).any()
            ):
                rejected_records.append(
                    RejectedRecord(sample_id, "invalid_box_dimensions")
                )
                continue
            if (box_values["x"] < 0).any() or (box_values["y"] < 0).any():
                rejected_records.append(
                    RejectedRecord(sample_id, "invalid_box_coordinates")
                )
                continue

        image_records.append(
        ImageRecord(
        sample_id=sample_id,
        image_path=str(image_directory / f"{sample_id}.dcm"),
        label=int(target),
        source_dataset="rsna_pneumonia_2018",
        raw_class=raw_class,
        )
        )
        if int(target) == 1:
            for row in patient_rows.itertuples(index=False):
                bounding_box_records.append(
                    BoundingBoxRecord(
                        sample_id=sample_id,
                        x=float(row.x),
                        y=float(row.y),
                        width=float(row.width),
                        height=float(row.height),
                    )
                )

    return image_records, bounding_box_records, rejected_records
