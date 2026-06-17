"""Tests for the DICOM preprocessing pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from pneumonai.preprocessing.specification import load_preprocessing_spec
from pneumonai.preprocessing.transform import preprocess

SPEC_PATH = Path("configs/preprocessing.yaml")


@pytest.fixture
def spec():
    return load_preprocessing_spec(SPEC_PATH)


def _make_dicom(pixels: np.ndarray, photometric: str = "MONOCHROME2") -> MagicMock:
    mock = MagicMock()
    mock.pixel_array = pixels
    mock.PhotometricInterpretation = photometric
    return mock


def _preprocess(pixels: np.ndarray, spec, photometric: str = "MONOCHROME2") -> torch.Tensor:
    with patch("pneumonai.preprocessing.transform.pydicom.dcmread", return_value=_make_dicom(pixels, photometric)):
        return preprocess(Path("fake.dcm"), spec)


def test_output_shape(spec) -> None:
    pixels = np.random.randint(0, 4096, (512, 512), dtype=np.uint16)
    tensor = _preprocess(pixels, spec)
    assert tensor.shape == (3, 224, 224)


def test_output_dtype(spec) -> None:
    pixels = np.random.randint(0, 4096, (512, 512), dtype=np.uint16)
    tensor = _preprocess(pixels, spec)
    assert tensor.dtype == torch.float32


def test_non_square_input_resizes_correctly(spec) -> None:
    pixels = np.random.randint(0, 4096, (1024, 512), dtype=np.uint16)
    tensor = _preprocess(pixels, spec)
    assert tensor.shape == (3, 224, 224)


def test_monochrome1_is_inverted(spec) -> None:
    pixels = np.zeros((64, 64), dtype=np.uint16)
    pixels[0, 0] = 1000
    tensor_m1 = _preprocess(pixels, spec, photometric="MONOCHROME1")
    tensor_m2 = _preprocess(pixels, spec, photometric="MONOCHROME2")
    assert not torch.allclose(tensor_m1, tensor_m2)


def test_uniform_image_does_not_produce_nan(spec) -> None:
    pixels = np.full((64, 64), 500, dtype=np.uint16)
    tensor = _preprocess(pixels, spec)
    assert torch.isfinite(tensor).all()


def test_normalization_produces_negative_values(spec) -> None:
    pixels = np.array([[0, 1000]], dtype=np.uint16)
    tensor = _preprocess(pixels, spec)
    assert tensor.min().item() < 0
