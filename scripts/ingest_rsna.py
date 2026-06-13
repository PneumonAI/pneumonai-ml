"""Command-line entry point for RSNA ingestion."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import yaml

from pneumonai.data.ingest import ingest_rsna

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "ingestion.yaml"


def main(argv: list[str] | None = None) -> None:
    """Load ingestion configuration and run the RSNA pipeline."""
    arguments = _parse_arguments(argv)
    config_path = arguments.config.resolve()
    config = _load_config(config_path)
    paths = _resolve_paths(config)

    report = ingest_rsna(
        labels_path=paths["labels"],
        classes_path=paths["classes"],
        image_directory=paths["train_images"],
        images_output_path=paths["images_output"],
        boxes_output_path=paths["boxes_output"],
    )

    print(json.dumps(asdict(report), indent=2, sort_keys=True))


def _parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map and validate the RSNA dataset, then write canonical metadata."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the ingestion YAML configuration.",
    )
    return parser.parse_args(argv)


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Ingestion configuration not found: {config_path}"
        )

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Ingestion configuration must be a YAML mapping.")

    _require_keys(config, "dataset", "inputs", "outputs")
    _require_keys(config["dataset"], "root")
    _require_keys(config["inputs"], "train_images", "labels", "classes")
    _require_keys(config["outputs"], "images", "bounding_boxes")
    return config


def _resolve_paths(config: dict[str, Any]) -> dict[str, Path]:
    dataset_root = _from_repository_root(config["dataset"]["root"])
    inputs = config["inputs"]
    outputs = config["outputs"]

    return {
        "train_images": dataset_root / inputs["train_images"],
        "labels": dataset_root / inputs["labels"],
        "classes": dataset_root / inputs["classes"],
        "images_output": _from_repository_root(outputs["images"]),
        "boxes_output": _from_repository_root(
            outputs["bounding_boxes"]
        ),
    }


def _from_repository_root(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPOSITORY_ROOT / path


def _require_keys(mapping: Any, *keys: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError("Ingestion configuration section must be a mapping.")

    missing_keys = [key for key in keys if key not in mapping]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(
            f"Ingestion configuration is missing required keys: {missing}"
        )


if __name__ == "__main__":
    main()
