# Cardiovascular disease prediction with stacking

A reproducible research implementation of **Decision Tree + Gradient Boosting + AdaBoost → Logistic Regression**, based on *A Stacking-Based Machine Learning Approach for Cardiovascular Disease Prediction* and the original repository script.

**Research status:** the paper reports **98.05% test accuracy**. This repository has **not reproduced that result**: the original `heart.csv` is unavailable. Software tests use synthetic fixtures only. This is research software, with no demonstrated clinical validation.

## Results: reported versus computed

| Evidence | Accuracy | Confusion matrix, actual rows / predicted columns `[0, 1]` |
|---|---|---|
| Paper, p. 4, Figure 6 (205 test cases) | 98.05% | `[[98, 2], [2, 103]]` |
| Reproduction on original dataset | Not available | Not available |

The paper reports class-wise precision, recall and F1 of 0.98. These claims are transcribed in [reported results](reports/paper_reported.json); they are never inserted into computed run outputs. Passing tests establishes software behavior, not medical performance.

## Run

Use Python **3.14** for the tested environment. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation -e .
cardio-research --data data/heart.csv --output outputs/run-001
```

Obtain and verify the original data using [dataset/provenance instructions](data/README.md). The command exits with a clear error if data is missing or invalid. There is no download, substitute dataset, or generated research data. Use a new output directory for each run.

Each run performs the fixed stratified 80/20 split (`random_state=42`), 10-fold stratified CV on the training portion, full training-set fit, then held-out test evaluation. Base models carry their own preprocessing so each internal stacking fold fits encoding/scaling on its own training rows. No tuning uses the test set.

Outputs:

- `metrics.json`: computed classification report, accuracy, ROC-AUC, confusion matrix, CV fold scores, data audit, SHA-256, package versions, and model settings.
- `split_indices.json`: zero-based data row positions for reproducibility; keep with the exact CSV.
- `confusion_matrix.png` and `roc_curve.png`: figures derived from the supplied data, never paper values.

## Design and audit

See the [paper/code audit](docs/AUDIT.md), [methodology and limitations](docs/METHODOLOGY.md), and [architecture](docs/ARCHITECTURE.md). The original script is preserved unchanged in `legacy/` for traceability; the root script is now a CLI entry point.

```text
src/cardio_research/  validation, model, evaluation, CLI
tests/               synthetic-only software tests
docs/                audit, methods, architecture, review notes
data/                provenance instructions (CSV excluded)
reports/             paper-reported facts, not computed results
legacy/              original notebook-style script
outputs/             local run outputs (excluded)
```

## Verify software

```bash
pytest -q
ruff check .
ruff format --check src tests
python -m build --no-isolation
```

CI runs these checks and verifies missing-data failure without fetching patient data. [Validation notes](docs/VALIDATION.md) distinguish completed checks from unavailable research validation.

## Paper and citation

**A Stacking-Based Machine Learning Approach for Cardiovascular Disease Prediction**. Authors as displayed in the supplied PDF: Kothai G; Mukil Barath S S; Dharaneesh S S; Lihash L B; Dharshna Kumar K S.

The owner reports that the work was published. The supplied PDF does not establish a verified venue, publication year, DOI or publisher record; its IEEE copyright line is a template placeholder. No formal publication metadata is invented here. See [publication notes](docs/PUBLICATION.md) before citing. The PDF is not redistributed.

## Limitations and reuse

Exact dataset identity, patient grouping, dependency versions used for the paper, and some model defaults remain unknown. Duplicate rows or repeated patients may inflate row-split performance. There is no external, prospective, fairness or clinical validation. Do not use this repository for diagnosis or treatment decisions.

The code is released under the [MIT License](LICENSE). The license applies to the repository's original code and documentation; it does not grant rights to third-party datasets or the paper. Confirm dataset and paper redistribution rights separately. See [contribution guidance](CONTRIBUTING.md).
