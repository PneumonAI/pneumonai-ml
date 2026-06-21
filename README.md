# PneumonAI ML

Training, evaluation, explainability, and model export for PneumonAI MVP v1.

The MVP uses one chest X-ray dataset and one binary classification model. The
initial model will classify possible pneumonia-associated lung opacity and
produce Grad-CAM visualizations. This is a research project, not a diagnostic
medical device.

## MVP Responsibilities

- Prepare one public chest X-ray dataset.
- Train and evaluate one ResNet18 classifier.
- Generate Grad-CAM heatmaps and overlays.
- Export the selected model as TorchScript.
- Verify PyTorch and C++ libtorch inference parity.

## Repository Structure

```text
configs/       Reproducible workflow configuration
data/          Local source data and generated artifacts (gitignored)
exports/       TorchScript model artifact for C++ backend
notebooks/     Exploration, experiments, and evaluation reports
reports/       Evaluation metrics and threshold selection output
scripts/       Thin command-line entry points
src/           Reusable production Python package
tests/         Automated behavior and validation tests
```

Notebooks are used to understand data and communicate findings. Reusable
parsing, validation, ingestion, preprocessing, training, and evaluation logic
belongs under `src/`. Scripts should only parse command-line inputs and call
those reusable modules.

### Scripts

```text
scripts/
├── ingest_rsna.py                  Build canonical metadata from RSNA source files
├── create_splits.py                Generate deterministic train/validation/test splits
├── generate_preprocessing_fixtures.py  Save reference tensors for contract verification
├── train.py                        Train a model from a config file
├── evaluate.py                     Run loss and accuracy on the test set
└── export.py                       Export the release checkpoint as TorchScript
```

### Notebooks

```text
notebooks/
├── 01_rsna_data_exploration.ipynb  RSNA dataset structure, label distribution, DICOM sanity checks
├── 02_train_experiments.ipynb      ResNet18/34 training experiments on Kaggle GPU
└── 03_evaluation_report.ipynb      Threshold selection on val set, full metric report on test set
```

### Source Package

```text
src/pneumonai/
├── data/
│   ├── rsna.py         Maps RSNA tables to canonical records
│   ├── schema.py       Canonical data types and field definitions
│   ├── validation.py   DICOM readability and bounding box geometry checks
│   ├── ingest.py       Writes canonical CSV files and ingestion report
│   └── splits.py       Deterministic stratified train/validation/test split logic
├── preprocessing/
│   ├── specification.py  Loads and validates preprocessing config
│   └── transform.py      DICOM → float32 CHW tensor pipeline
└── training/
    ├── dataset.py      ChestXrayDataset — lazy DICOM loading from split CSVs
    ├── model.py        build_model — pretrained ResNet with replaced fc layer
    └── trainer.py      train_one_epoch and validate loop functions
```

### Configs

```text
configs/
├── ingestion.yaml       RSNA source paths and ingestion settings
├── splits.yaml          Split ratios and random seed
├── preprocessing.yaml   Versioned preprocessing contract (shared by Python and C++)
├── training.yaml        Training hyperparameters and paths
└── model_release.yaml   Selected model checkpoint, metrics, and threshold
```

### Exports

```text
exports/
├── model.pt              TorchScript model for C++ libtorch inference
├── metadata.yaml         Integration contract (input shape, dtype, threshold, classes)
├── checksum.sha256       SHA256 integrity verification
├── reference_input.npy   Dummy input tensor for parity testing
└── reference_output.npy  Reference logit output for parity testing
```

## Development Setup

The project requires Python 3.11 or newer. From WSL:

```bash
cd /mnt/c/PneumonAI/pneumonai-ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the automated data-pipeline tests with:

```bash
python -m pytest tests/data -q
```

## Preprocessing Contract

The versioned MVP preprocessing specification is stored in
`configs/preprocessing.yaml`. It defines the exact transformation that both
Python training and C++ inference must reproduce.

```text
DICOM pixel data
  -> require one two-dimensional grayscale image
  -> invert pixels when PhotometricInterpretation is MONOCHROME1
  -> min-max scale intensities to [0, 1]
  -> resize directly to 224 x 224 with bilinear interpolation
  -> repeat grayscale into three channels
  -> apply ImageNet channel mean and standard deviation
  -> return a float32 tensor in CHW layout
```

The resulting model input contract is:

```text
shape:   [3, 224, 224]
layout:  CHW
dtype:   float32
mean:    [0.485, 0.456, 0.406]
std:     [0.229, 0.224, 0.225]
version: 1.0
```

Direct resizing does not preserve the original aspect ratio. This is an
intentional MVP simplification that keeps Python and C++ behavior easy to
reproduce. Changing image dimensions, interpolation, intensity scaling, or
normalization creates a new preprocessing version and requires model
reevaluation.

Raw DICOM files are never modified. Preprocessing occurs in memory when an
image is loaded. Random rotation, flipping, cropping, and other training-only
augmentation are not part of this shared inference contract.

## Dataset

The selected MVP dataset is the
[RSNA Pneumonia Detection Challenge 2018](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pneumonia-detection-challenge-2018).

The model receives the complete decoded DICOM image. Bounding boxes identify
expert-marked regions of possible pneumonia-associated lung opacity. They are
kept as localization metadata for visualization and explainability evaluation;
they are not cropped model inputs.

### Local Data Layout

Dataset files are stored locally under `data/`:

```text
data/
|-- raw/
|   `-- rsna_pneumonia_2018/
|       |-- archives/
|       |   |-- stage_2_train_images.zip
|       |   |-- stage_2_test_images.zip
|       |   |-- stage_2_train_labels.csv.zip
|       |   `-- stage_2_detailed_class_info.csv.zip
|       |-- source_metadata/
|       |   |-- stage_2_train_labels.csv
|       |   `-- stage_2_detailed_class_info.csv
|       |-- train/
|       |   `-- images/
|       |       `-- <sample_id>.dcm
|       `-- test/
|           `-- images/
|               `-- <sample_id>.dcm
|-- metadata/
|   |-- images.csv
|   `-- bounding_boxes.csv
`-- splits/
    |-- train.csv
    |-- validation.csv
    `-- test.csv
```

`raw/` contains source files exactly as downloaded or extracted. It must not be
modified by training code.

`metadata/` contains canonical files generated by the ingestion pipeline:

```text
images.csv
sample_id,image_path,label,source_dataset,raw_class
```

```text
bounding_boxes.csv
sample_id,x,y,width,height
```

There is exactly one row per image in `images.csv`. An image can have zero or
more rows in `bounding_boxes.csv`.

`splits/` will contain deterministic train, validation, and test manifests.
Split generation belongs to a later dataset-foundation task.

### Ingestion Pipeline

The ingestion pipeline has four separate responsibilities:

```text
rsna.py        Maps RSNA tables to canonical records
validation.py  Verifies DICOM readability and box geometry
ingest.py      Writes canonical CSV files and builds the report
ingest_rsna.py Loads configuration and runs the pipeline
```

Run ingestion from the repository root:

```bash
source .venv/bin/activate
python scripts/ingest_rsna.py
```

The command uses `configs/ingestion.yaml` by default. A different
configuration can be supplied with:

```bash
python scripts/ingest_rsna.py --config path/to/ingestion.yaml
```

Configuration input paths are resolved from `dataset.root`. Output paths are
resolved from the repository root.

The command:

1. Reads the RSNA labels and detailed class tables.
2. Creates one canonical image record per accepted image.
3. Preserves zero or more bounding boxes for each accepted image.
4. Rejects inconsistent source rows rather than guessing.
5. Confirms that each accepted DICOM exists and its pixels can be decoded.
6. Confirms that bounding boxes fit inside the decoded image.
7. Writes `data/metadata/images.csv` and
   `data/metadata/bounding_boxes.csv`.
8. Prints a JSON ingestion report with accepted, rejected, class, box, and
   rejection-reason counts.

The pipeline does not modify source CSV or DICOM files. A rejected image and
all boxes associated with it are excluded from the generated metadata.

### Current Local Data

- Training DICOM images: 26,684
- Challenge test DICOM images: 3,000
- Training label rows: 30,227
- Detailed class rows: 30,227
- Unique labeled training image IDs: 26,684

The ingestion implementation is complete. Final canonical counts must be
recorded only after running the command against the complete local dataset and
reviewing its JSON report.

### Data Rules

- Do not commit datasets, DICOM images, generated manifests, model weights, or
  patient-level data to Git.
- Generate image paths from the configured dataset root; do not hard-code
  machine-specific paths in source code.
- Keep raw RSNA classes for provenance.
- Map RSNA `Target = 1` to canonical `label = 1`.
- Map RSNA `Target = 0` to canonical `label = 0`.
- Reject conflicting labels, invalid boxes, and missing or unreadable images
  instead of silently guessing.
- Do not attempt to identify or contact represented patients.

The source archives, extracted files, canonical metadata, and split manifests
are excluded through `.gitignore`.

## Project Management

Linear is the source of truth for roadmap planning, priorities, dependencies,
and task status. GitHub is used for code, branches, pull requests, reviews, and
releases.

See the [PneumonAI organization](https://github.com/PneumonAI) for the broader
project scope and resources.
