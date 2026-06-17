"""Versioned, framework-independent preprocessing specification."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PreprocessingSpec:
    """Validated preprocessing contract shared by training and inference."""

    version: str
    input_format: str
    input_dimensions: int
    invert_monochrome1: bool
    intensity_scaling: str
    intensity_output_min: float
    intensity_output_max: float
    width: int
    height: int
    channels: int
    resize: str
    preserve_aspect_ratio: bool
    tensor_layout: str
    tensor_dtype: str
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        """Reject unsupported or internally inconsistent specifications."""
        if not self.version.strip():
            raise ValueError("Preprocessing version must not be empty.")
        if self.input_format != "dicom":
            raise ValueError("MVP preprocessing supports only DICOM input.")
        if self.input_dimensions != 2:
            raise ValueError("MVP preprocessing requires a 2D input image.")
        if self.intensity_scaling != "min_max":
            raise ValueError("MVP preprocessing supports only min_max scaling.")
        if not all(
            isfinite(value)
            for value in (
                self.intensity_output_min,
                self.intensity_output_max,
            )
        ):
            raise ValueError("Intensity output bounds must be finite.")
        if self.intensity_output_min >= self.intensity_output_max:
            raise ValueError(
                "Intensity output minimum must be below the maximum."
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image width and height must be positive.")
        if self.channels != 3:
            raise ValueError("MVP preprocessing requires three channels.")
        if self.resize != "bilinear":
            raise ValueError("MVP preprocessing supports bilinear resize.")
        if self.preserve_aspect_ratio:
            raise ValueError(
                "MVP preprocessing uses direct resize without aspect ratio."
            )
        if self.tensor_layout != "CHW":
            raise ValueError("MVP tensor layout must be CHW.")
        if self.tensor_dtype != "float32":
            raise ValueError("MVP tensor dtype must be float32.")
        if len(self.mean) != self.channels:
            raise ValueError("Mean length must equal the channel count.")
        if len(self.std) != self.channels:
            raise ValueError("Standard deviation length must equal channels.")
        if not all(isfinite(value) for value in self.mean):
            raise ValueError("Normalization means must be finite.")
        if not all(isfinite(value) and value > 0 for value in self.std):
            raise ValueError(
                "Normalization standard deviations must be finite and positive."
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize the contract using the versioned configuration shape."""
        return {
            "version": self.version,
            "input": {
                "format": self.input_format,
                "dimensions": self.input_dimensions,
            },
            "intensity": {
                "invert_monochrome1": self.invert_monochrome1,
                "scaling": self.intensity_scaling,
                "output_min": self.intensity_output_min,
                "output_max": self.intensity_output_max,
            },
            "image": {
                "width": self.width,
                "height": self.height,
                "channels": self.channels,
                "resize": self.resize,
                "preserve_aspect_ratio": self.preserve_aspect_ratio,
            },
            "tensor": {
                "layout": self.tensor_layout,
                "dtype": self.tensor_dtype,
            },
            "normalization": {
                "mean": list(self.mean),
                "std": list(self.std),
            },
        }


def load_preprocessing_spec(path: Path) -> PreprocessingSpec:
    """Load and validate a preprocessing specification from YAML."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Preprocessing specification not found: {path}"
        )

    with path.open(encoding="utf-8") as specification_file:
        config = yaml.safe_load(specification_file)

    if not isinstance(config, dict):
        raise ValueError("Preprocessing specification must be a YAML mapping.")

    _require_keys(
        config,
        "version",
        "input",
        "intensity",
        "image",
        "tensor",
        "normalization",
    )
    input_config = _require_mapping(config, "input")
    intensity = _require_mapping(config, "intensity")
    image = _require_mapping(config, "image")
    tensor = _require_mapping(config, "tensor")
    normalization = _require_mapping(config, "normalization")

    _require_keys(input_config, "format", "dimensions")
    _require_keys(
        intensity,
        "invert_monochrome1",
        "scaling",
        "output_min",
        "output_max",
    )
    _require_keys(
        image,
        "width",
        "height",
        "channels",
        "resize",
        "preserve_aspect_ratio",
    )
    _require_keys(tensor, "layout", "dtype")
    _require_keys(normalization, "mean", "std")

    try:
        return PreprocessingSpec(
            version=str(config["version"]),
            input_format=str(input_config["format"]),
            input_dimensions=_as_int(
                input_config["dimensions"],
                "input.dimensions",
            ),
            invert_monochrome1=_as_bool(
                intensity["invert_monochrome1"],
                "intensity.invert_monochrome1",
            ),
            intensity_scaling=str(intensity["scaling"]),
            intensity_output_min=_as_float(
                intensity["output_min"],
                "intensity.output_min",
            ),
            intensity_output_max=_as_float(
                intensity["output_max"],
                "intensity.output_max",
            ),
            width=_as_int(image["width"], "image.width"),
            height=_as_int(image["height"], "image.height"),
            channels=_as_int(image["channels"], "image.channels"),
            resize=str(image["resize"]),
            preserve_aspect_ratio=_as_bool(
                image["preserve_aspect_ratio"],
                "image.preserve_aspect_ratio",
            ),
            tensor_layout=str(tensor["layout"]),
            tensor_dtype=str(tensor["dtype"]),
            mean=_as_float_tuple(normalization["mean"], "normalization.mean"),
            std=_as_float_tuple(normalization["std"], "normalization.std"),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError(
            "Preprocessing specification contains invalid values."
        ) from error


def _require_mapping(
    config: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    value = config[key]
    if not isinstance(value, dict):
        raise ValueError(
            f"Preprocessing section '{key}' must be a mapping."
        )
    return value


def _require_keys(mapping: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(
            "Preprocessing specification is missing required keys: "
            + ", ".join(missing)
        )


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer.")
    return value


def _as_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric.")
    return float(value)


def _as_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean.")
    return value


def _as_float_tuple(value: Any, field: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a YAML list.")
    return tuple(_as_float(item, field) for item in value)
