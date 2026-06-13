"""Tests for the RSNA ingestion command-line entry point."""

from pathlib import Path

from pneumonai.data.schema import IngestionReport
from scripts import ingest_rsna as cli


def test_resolve_paths_uses_dataset_root_for_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli, "REPOSITORY_ROOT", tmp_path)
    config = {
        "dataset": {"root": "data/raw/rsna"},
        "inputs": {
            "train_images": "train/images",
            "labels": "source_metadata/labels.csv",
            "classes": "source_metadata/classes.csv",
        },
        "outputs": {
            "images": "data/metadata/images.csv",
            "bounding_boxes": "data/metadata/bounding_boxes.csv",
        },
    }

    paths = cli._resolve_paths(config)

    assert paths["train_images"] == (
        tmp_path / "data/raw/rsna/train/images"
    )
    assert paths["labels"] == (
        tmp_path / "data/raw/rsna/source_metadata/labels.csv"
    )
    assert paths["images_output"] == (
        tmp_path / "data/metadata/images.csv"
    )


def test_main_calls_pipeline_and_prints_report(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "ingestion.yaml"
    config_path.write_text(
        """
dataset:
  root: data/raw/rsna
inputs:
  train_images: train/images
  labels: source_metadata/labels.csv
  classes: source_metadata/classes.csv
outputs:
  images: data/metadata/images.csv
  bounding_boxes: data/metadata/bounding_boxes.csv
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "REPOSITORY_ROOT", tmp_path)
    received: dict[str, Path] = {}

    def fake_ingest_rsna(**paths: Path) -> IngestionReport:
        received.update(paths)
        return IngestionReport(
            source_rows=3,
            accepted_images=2,
            positive_images=1,
            negative_images=1,
            bounding_boxes=1,
            duplicate_boxes_removed=0,
            rejected_records=1,
            rejection_reasons={"missing_class": 1},
        )

    monkeypatch.setattr(cli, "ingest_rsna", fake_ingest_rsna)

    cli.main(["--config", str(config_path)])

    output = capsys.readouterr().out
    assert received["labels_path"] == (
        tmp_path / "data/raw/rsna/source_metadata/labels.csv"
    )
    assert received["boxes_output_path"] == (
        tmp_path / "data/metadata/bounding_boxes.csv"
    )
    assert '"accepted_images": 2' in output
    assert '"missing_class": 1' in output
