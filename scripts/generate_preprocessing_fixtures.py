"""Generate preprocessing fixtures for C++ parity verification."""

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from pneumonai.preprocessing.specification import load_preprocessing_spec
from pneumonai.preprocessing.transform import preprocess

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPOSITORY_ROOT / "data" / "fixtures"
SPEC_PATH = REPOSITORY_ROOT / "configs" / "preprocessing.yaml"


def _create_synthetic_dicom(
    path: Path,
    pixels: np.ndarray,
    photometric: str = "MONOCHROME2",
) -> None:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.PhotometricInterpretation = photometric
    ds.Rows, ds.Columns = pixels.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PixelData = pixels.tobytes()
    pydicom.dcmwrite(path, ds)


def _save_fixture(name: str, pixels: np.ndarray, photometric: str, spec) -> None:
    fixture_dir = FIXTURES_DIR / name
    fixture_dir.mkdir(parents=True, exist_ok=True)

    dicom_path = fixture_dir / "input.dcm"
    _create_synthetic_dicom(dicom_path, pixels, photometric)

    tensor = preprocess(dicom_path, spec)
    np.save(fixture_dir / "output.npy", tensor.numpy())


def main() -> None:
    spec = load_preprocessing_spec(SPEC_PATH)

    _save_fixture(
        name="normal",
        pixels=np.random.default_rng(42).integers(0, 4096, (64, 64), dtype=np.uint16),
        photometric="MONOCHROME2",
        spec=spec,
    )
    _save_fixture(
        name="monochrome1",
        pixels=np.random.default_rng(43).integers(0, 4096, (64, 64), dtype=np.uint16),
        photometric="MONOCHROME1",
        spec=spec,
    )
    _save_fixture(
        name="uniform",
        pixels=np.full((64, 64), 500, dtype=np.uint16),
        photometric="MONOCHROME2",
        spec=spec,
    )
    _save_fixture(
        name="nonsquare",
        pixels=np.random.default_rng(44).integers(0, 4096, (128, 64), dtype=np.uint16),
        photometric="MONOCHROME2",
        spec=spec,
    )

    print("Fixtures written to", FIXTURES_DIR)


if __name__ == "__main__":
    main()
