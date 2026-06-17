"""Tests for the versioned preprocessing specification."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from pneumonai.preprocessing.specification import (
    PreprocessingSpec,
    load_preprocessing_spec,
)


def _valid_config() -> dict[str, object]:
    return {
        "version": "1.0",
        "input": {"format": "dicom", "dimensions": 2},
        "intensity": {
            "invert_monochrome1": True,
            "scaling": "min_max",
            "output_min": 0.0,
            "output_max": 1.0,
        },
        "image": {
            "width": 224,
            "height": 224,
            "channels": 3,
            "resize": "bilinear",
            "preserve_aspect_ratio": False,
        },
        "tensor": {"layout": "CHW", "dtype": "float32"},
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }


def _write_config(tmp_path: Path, config: dict[str, object]) -> Path:
    path = tmp_path / "preprocessing.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_load_preprocessing_spec() -> None:
    specification = load_preprocessing_spec(
        Path("configs/preprocessing.yaml")
    )

    assert specification.version == "1.0"
    assert specification.width == 224
    assert specification.height == 224
    assert specification.mean == (0.485, 0.456, 0.406)
    assert specification.std == (0.229, 0.224, 0.225)


def test_specification_is_immutable() -> None:
    specification = load_preprocessing_spec(
        Path("configs/preprocessing.yaml")
    )

    with pytest.raises(FrozenInstanceError):
        specification.width = 256


def test_specification_serializes_to_configuration_shape() -> None:
    specification = load_preprocessing_spec(
        Path("configs/preprocessing.yaml")
    )

    serialized = specification.to_dict()

    assert serialized["image"] == {
        "width": 224,
        "height": 224,
        "channels": 3,
        "resize": "bilinear",
        "preserve_aspect_ratio": False,
    }
    assert serialized["normalization"] == {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


def test_load_rejects_missing_section(tmp_path: Path) -> None:
    config = _valid_config()
    del config["normalization"]

    with pytest.raises(ValueError, match="missing required keys"):
        load_preprocessing_spec(_write_config(tmp_path, config))


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("image", "width", 0, "width and height"),
        ("image", "channels", 1, "three channels"),
        ("image", "resize", "nearest", "bilinear"),
        ("image", "preserve_aspect_ratio", True, "direct resize"),
        ("tensor", "layout", "HWC", "CHW"),
        ("tensor", "dtype", "float64", "float32"),
    ],
)
def test_load_rejects_unsupported_contract_values(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    config = _valid_config()
    config[section][key] = value

    with pytest.raises(ValueError, match=message):
        load_preprocessing_spec(_write_config(tmp_path, config))


def test_load_rejects_wrong_normalization_length(tmp_path: Path) -> None:
    config = _valid_config()
    config["normalization"]["mean"] = [0.5]

    with pytest.raises(ValueError, match="Mean length"):
        load_preprocessing_spec(_write_config(tmp_path, config))


@pytest.mark.parametrize("invalid_std", [0.0, -1.0, float("inf")])
def test_load_rejects_invalid_standard_deviation(
    tmp_path: Path,
    invalid_std: float,
) -> None:
    config = _valid_config()
    config["normalization"]["std"][0] = invalid_std

    with pytest.raises(ValueError, match="finite and positive"):
        load_preprocessing_spec(_write_config(tmp_path, config))


def test_load_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_preprocessing_spec(tmp_path / "missing.yaml")


def test_direct_construction_validates_contract() -> None:
    with pytest.raises(ValueError, match="DICOM"):
        PreprocessingSpec(
            version="1.0",
            input_format="png",
            input_dimensions=2,
            invert_monochrome1=True,
            intensity_scaling="min_max",
            intensity_output_min=0.0,
            intensity_output_max=1.0,
            width=224,
            height=224,
            channels=3,
            resize="bilinear",
            preserve_aspect_ratio=False,
            tensor_layout="CHW",
            tensor_dtype="float32",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
