"""Export the selected ResNet checkpoint to TorchScript for C++ inference."""

import hashlib
from pathlib import Path

import numpy as np
import torch
import yaml

from pneumonai.training.model import build_model
from pneumonai.preprocessing.specification import load_preprocessing_spec

REPO = Path(__file__).resolve().parents[1]
RELEASE_CONFIG = REPO / "configs" / "model_release.yaml"
PREPROCESSING_CONFIG = REPO / "configs" / "preprocessing.yaml"
EXPORTS_DIR = REPO / "exports"


def main() -> None:
    EXPORTS_DIR.mkdir(exist_ok=True)

    with open(RELEASE_CONFIG) as f:
        release = yaml.safe_load(f)

    spec = load_preprocessing_spec(PREPROCESSING_CONFIG)

    model = build_model(release["arch"], release["num_classes"])
    checkpoint = REPO / release["checkpoint"]
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    dummy_input = torch.zeros(1, 3, spec.height, spec.width, dtype=torch.float32)
    with torch.no_grad():
        reference_output = model(dummy_input).squeeze(1)

    traced = torch.jit.trace(model, dummy_input)
    model_path = EXPORTS_DIR / "model.pt"
    traced.save(str(model_path))

    checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (EXPORTS_DIR / "checksum.sha256").write_text(f"{checksum}  model.pt\n")

    metadata = {
        "model_version": release["arch"],
        "preprocessing_version": spec.version,
        "num_classes": release["num_classes"],
        "input_shape": [1, spec.channels, spec.height, spec.width],
        "input_dtype": spec.tensor_dtype,
        "output": "raw logit — apply sigmoid for probability",
        "threshold": 0.5,
        "classes": {0: "normal", 1: "pneumonia"},
        "val_accuracy": release["val_accuracy"],
        "test_accuracy": release["test_accuracy"],
        "checksum_sha256": checksum,
    }
    with open(EXPORTS_DIR / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    np.save(EXPORTS_DIR / "reference_input.npy", dummy_input.numpy())
    np.save(EXPORTS_DIR / "reference_output.npy", reference_output.detach().numpy())

    print(f"Exported to {EXPORTS_DIR}")
    print(f"SHA256: {checksum}")
    print(f"Reference output (logit): {reference_output.item():.6f}")


if __name__ == "__main__":
    main()
