"""Canonical image and bounding-box record definitions.

Implementation belongs to PNE-7 and must follow the schema recorded in PNE-6.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageRecord:
    """Canonical image record definition."""

    sample_id: str
    image_path: str
    label: int
    source_dataset: str
    raw_class: str

    def to_dict(self) -> dict[str, object]:
        """Convert the ImageRecord to a dictionary."""
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "label": self.label,
            "source_dataset": self.source_dataset,
            "raw_class": self.raw_class,
        }


@dataclass(frozen=True)
class BoundingBoxRecord:
    """Canonical bounding-box record definition."""

    sample_id: str
    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> dict[str, object]:
        """Convert the BoundingBoxRecord to a dictionary."""
        return {
            "sample_id": self.sample_id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class RejectedRecord:
    """Source record rejected during canonical mapping."""

    sample_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        """Convert the rejection to a dictionary."""
        return {
            "sample_id": self.sample_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class IngestionReport:
    """Canonical ingestion report definition."""

    source_rows: int
    accepted_images: int
    positive_images: int
    negative_images: int
    bounding_boxes: int
    duplicate_boxes_removed: int
    rejected_records: int
    rejection_reasons: dict[str, int]
