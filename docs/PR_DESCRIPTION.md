## Summary

Convert the notebook-style script into a reproducible research package while preserving Decision Tree + Gradient Boosting + AdaBoost → Logistic Regression, seed 42, and the stratified 80/20 split.

```text
validated original CSV
  -> fixed train/test split
  -> training-only 10-fold evaluation of the complete stack
  -> final training fit -> held-out metrics and figures
paper claims -> separate static report, never injected into run results
```

Adds fold-local preprocessing, pinned dependencies, data/provenance checks, model/run metadata, tests, CI, README, paper/code audit and methodology/architecture documentation. Preserves the original script in `legacy/` and replaces its root entry point with the installed CLI.

## Evidence

- **Before:** hard-coded `/content/heart.csv`, full-data category discovery, no separate 10-fold evaluation, no dependency record/tests/CI.
  **After:** configurable validated input, preprocessing fitted inside each base learner's folds, explicit training-only 10-fold CV plus separate five-fold stacking CV, recorded data/source hashes, split indices, versions, metrics and PNG figures.
- **Local checks:** 13 synthetic-only tests pass; Ruff lint/format, dependency check, source/wheel build and missing-data exit-code check pass. Figures inspected. One synthetic CV fold produced an undefined-precision warning (set to zero), documented in validation notes.
- **Research status:** exact `heart.csv` absent. 98.05% and `[[98,2],[2,103]]` remain paper-reported only. No real-data or clinical validation is claimed. Remote CI awaits branch publication.

## Merge Danger

**Door:** two-way

Changes can be reverted; the original script remains archived. Owner review is required before merge/release. The MIT license covers repository code and documentation only; no data, PDF, invented DOI/venue, or fitted model is included.

**Blast Radius:** research-workflow

The root script now requires package installation. Fold-local preprocessing, explicit outer CV settings, higher Logistic Regression iteration limit and modern pinned AdaBoost defaults are documented differences; exact historical predictions cannot be promised without the original dataset and environment. Duplicate/patient overlap remains a limitation of row-level evaluation.
