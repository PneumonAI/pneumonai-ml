"""Command-line entry point for deterministic dataset splitting."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pneumonai.data.splits import create_splits

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "splits.yaml"


def main(argv: list[str] | None = None) -> None:
    """Load configuration, create splits, and write their manifests."""
    arguments = _parse_arguments(argv)
    config = _load_config(arguments.config.resolve())
    paths = _resolve_paths(config)

    images = pd.read_csv(paths["images"])
    ratios = config["ratios"]
    train, validation, test = create_splits(
        images,
        train_ratio=ratios["train"],
        validation_ratio=ratios["validation"],
        test_ratio=ratios["test"],
        seed=config["seed"],
    )

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    for name, split in splits.items():
        output_path = paths[name]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(output_path, index=False)

    print(json.dumps(_build_report(splits, config["seed"]), indent=2))


def _parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic, stratified train, validation, and test "
            "manifests."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the split YAML configuration.",
    )
    return parser.parse_args(argv)


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Split configuration not found: {config_path}"
        )

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Split configuration must be a YAML mapping.")

    _require_keys(config, "seed", "input", "ratios", "outputs")
    _require_keys(config["input"], "images")
    _require_keys(config["ratios"], "train", "validation", "test")
    _require_keys(config["outputs"], "train", "validation", "test")
    if not isinstance(config["seed"], int):
        raise ValueError("Split seed must be an integer.")
    return config


def _resolve_paths(config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["outputs"]
    return {
        "images": _from_repository_root(config["input"]["images"]),
        "train": _from_repository_root(outputs["train"]),
        "validation": _from_repository_root(outputs["validation"]),
        "test": _from_repository_root(outputs["test"]),
    }


def _from_repository_root(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPOSITORY_ROOT / path


def _build_report(
    splits: dict[str, pd.DataFrame],
    seed: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {"seed": seed, "splits": {}}
    for name, split in splits.items():
        class_counts = split["label"].value_counts().sort_index()
        report["splits"][name] = {
            "rows": len(split),
            "class_counts": {
                str(label): int(count)
                for label, count in class_counts.items()
            },
        }
    return report


def _require_keys(mapping: Any, *keys: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError("Split configuration section must be a mapping.")

    missing_keys = [key for key in keys if key not in mapping]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(
            f"Split configuration is missing required keys: {missing}"
        )


if __name__ == "__main__":
    main()
