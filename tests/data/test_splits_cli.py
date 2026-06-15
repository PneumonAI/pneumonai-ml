"""Tests for the dataset splitting command-line entry point."""

import json
from pathlib import Path

import pandas as pd

from scripts import create_splits as cli


def _config_text() -> str:
    return """
seed: 42
input:
  images: data/metadata/images.csv
ratios:
  train: 0.70
  validation: 0.15
  test: 0.15
outputs:
  train: data/splits/train.csv
  validation: data/splits/validation.csv
  test: data/splits/test.csv
""".strip()


def test_resolve_paths_uses_repository_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli, "REPOSITORY_ROOT", tmp_path)
    config = {
        "input": {"images": "data/metadata/images.csv"},
        "outputs": {
            "train": "data/splits/train.csv",
            "validation": "data/splits/validation.csv",
            "test": "data/splits/test.csv",
        },
    }

    paths = cli._resolve_paths(config)

    assert paths["images"] == tmp_path / "data/metadata/images.csv"
    assert paths["train"] == tmp_path / "data/splits/train.csv"


def test_main_writes_manifests_and_prints_report(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "splits.yaml"
    config_path.write_text(_config_text(), encoding="utf-8")
    images_path = tmp_path / "data" / "metadata" / "images.csv"
    images_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "sample_id": ["sample-1", "sample-2", "sample-3"],
            "label": [0, 1, 0],
        }
    ).to_csv(images_path, index=False)
    train = pd.DataFrame(
        {"sample_id": ["sample-1"], "label": [0]}
    )
    validation = pd.DataFrame(
        {"sample_id": ["sample-2"], "label": [1]}
    )
    test = pd.DataFrame(
        {"sample_id": ["sample-3"], "label": [0]}
    )
    received: dict[str, object] = {}

    def fake_create_splits(
        images: pd.DataFrame,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        received.update(
            {
                "rows": len(images),
                "ratios": (
                    train_ratio,
                    validation_ratio,
                    test_ratio,
                ),
                "seed": seed,
            }
        )
        return train, validation, test

    monkeypatch.setattr(cli, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(cli, "create_splits", fake_create_splits)

    cli.main(["--config", str(config_path)])

    assert received == {
        "rows": 3,
        "ratios": (0.70, 0.15, 0.15),
        "seed": 42,
    }
    assert pd.read_csv(
        tmp_path / "data/splits/train.csv"
    ).equals(train)
    assert pd.read_csv(
        tmp_path / "data/splits/validation.csv"
    ).equals(validation)
    assert pd.read_csv(
        tmp_path / "data/splits/test.csv"
    ).equals(test)

    report = json.loads(capsys.readouterr().out)
    assert report["seed"] == 42
    assert report["splits"]["train"] == {
        "rows": 1,
        "class_counts": {"0": 1},
    }


def test_load_config_rejects_missing_required_section(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "splits.yaml"
    config_path.write_text("seed: 42", encoding="utf-8")

    try:
        cli._load_config(config_path)
    except ValueError as error:
        assert "missing required keys" in str(error)
    else:
        raise AssertionError("Invalid configuration was accepted.")
