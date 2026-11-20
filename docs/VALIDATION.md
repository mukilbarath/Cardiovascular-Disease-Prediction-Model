# Validation record

Local validation: 2026-11-20, Windows, Python 3.14.2. Dependencies are frozen in `requirements-lock.txt`; direct runtime/development dependencies are also listed separately. Linux GitHub Actions has been configured but has not yet run remotely.

- 13 software tests passed, including a full 10-fold evaluation plus five-fold internal stacking on synthetic fixtures, deterministic model predictions, data rejection, independent preprocessing, known metrics, split integrity and generated PNG files.
- Ruff lint and format checks passed.
- `pip check` reported no broken requirements.
- Wheel and source distribution built successfully with `python -m build --no-isolation`.
- Installed CLI with absent `data/heart.csv` returned exit code 2 and a useful dataset/provenance message.
- Original script retained byte-for-byte in `legacy/`.
- Synthetic confusion-matrix and ROC output figures were visually inspected. No synthetic metrics or figures are committed as research results.

One expected scikit-learn warning occurred during synthetic CV: a fold predicted no positive samples, so its precision was set to zero. This fixture validates execution, not predictive quality. No warnings are globally suppressed.

Real-data training, the paper's 98.05% result, original 10-fold scores, and clinical/external validation cannot be verified without the exact source dataset and study evidence. No synthetic test result is evidence for those claims.
