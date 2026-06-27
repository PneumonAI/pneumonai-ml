"""Export ResNet checkpoint to TorchScript for C++ inference + Grad-CAM.

The exported model returns (logits, feature_maps) so that C++ libtorch can
compute Grad-CAM via torch::autograd::grad without Python hooks. The graph
is kept alive — no torch.no_grad() wraps the trace or forward calls here.
"""

import hashlib
import re
from pathlib import Path

import numpy as np
import torch
import yaml

from pneumonai.training.model import build_model
from pneumonai.preprocessing.specification import load_preprocessing_spec
from pneumonai.explainability.gradcam import GradCAMResNet

REPO = Path(__file__).resolve().parents[1]
RELEASE_CONFIG = REPO / "configs" / "model_release.yaml"
PREPROCESSING_CONFIG = REPO / "configs" / "preprocessing.yaml"
EXPORTS_DIR = REPO / "exports"
MODEL_FILENAME = "pneumonia_resnet18_v1.pt"


def main() -> None:
    EXPORTS_DIR.mkdir(exist_ok=True)

    with open(RELEASE_CONFIG) as f:
        release = yaml.safe_load(f)

    spec = load_preprocessing_spec(PREPROCESSING_CONFIG)

    model = build_model(release["arch"], release["num_classes"])
    checkpoint = REPO / release["checkpoint"]
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    wrapped = GradCAMResNet(model)
    wrapped.eval()

    dummy_input = torch.zeros(1, 3, spec.height, spec.width, dtype=torch.float32)
    dummy_input.requires_grad_(True)

    traced = torch.jit.trace(wrapped, dummy_input)
    model_path = EXPORTS_DIR / MODEL_FILENAME
    traced.save(str(model_path))

    # Verify the graph is alive: feats must have grad_fn
    # If either assertion fails, the export is broken and C++ autograd::grad will fail.
    with torch.enable_grad():
        logits, feats = traced(dummy_input)
    assert feats.requires_grad, "feats.requires_grad is False — graph is dead, export broken"
    assert feats.grad_fn is not None, "feats.grad_fn is None — graph is dead, export broken"
    print(f"Graph alive: feats.grad_fn = {type(feats.grad_fn).__name__}")

    checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (EXPORTS_DIR / "checksum.sha256").write_text(f"{checksum}  {MODEL_FILENAME}\n")

    # Update checksum and artifact_filename in existing metadata.yaml without
    # overwriting the rest of the manually curated contract.
    metadata_path = EXPORTS_DIR / "metadata.yaml"
    if metadata_path.exists():
        content = metadata_path.read_text(encoding="utf-8")
        content = re.sub(
            r"^checksum_sha256:.*$", f"checksum_sha256: {checksum}",
            content, flags=re.MULTILINE,
        )
        content = re.sub(
            r"^artifact_filename:.*$", f"artifact_filename: {MODEL_FILENAME}",
            content, flags=re.MULTILINE,
        )
        metadata_path.write_text(content, encoding="utf-8")

    # Oracle fixtures for Artem's C++ parity test.
    # reference_feats lets him split parity into two steps:
    #   1. forward parity: his feats vs reference_feats
    #   2. math parity:    his heatmap vs the Python heatmap
    np.save(EXPORTS_DIR / "reference_input.npy", dummy_input.detach().numpy())
    np.save(EXPORTS_DIR / "reference_logits.npy", logits.detach().numpy())
    np.save(EXPORTS_DIR / "reference_feats.npy", feats.detach().numpy())

    print(f"Exported:        {model_path}")
    print(f"SHA256:          {checksum}")
    print(f"Reference logit: {logits.item():.6f}")
    print(f"Reference feats: shape={list(feats.shape)}, mean={feats.mean().item():.6f}")


if __name__ == "__main__":
    main()
