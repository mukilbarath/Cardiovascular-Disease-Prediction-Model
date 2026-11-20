"""Strict input validation without guessing data provenance or silently repairing records."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["cp", "restecg", "slope", "ca", "thal", "sex", "exang", "fbs"]
FEATURES = NUMERIC + CATEGORICAL


def load_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(
            f"Dataset not found: {path}. Supply the original heart.csv; see data/README.md. "
            "No substitute or synthetic research dataset is generated."
        )
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    required = set(FEATURES + ["target"])
    if len(header) != len(set(header)) or set(header) != required:
        raise ValueError(
            "CSV must contain exactly the 13 documented features and target, once each."
        )
    try:
        frame = pd.read_csv(path).apply(pd.to_numeric, errors="raise")
    except (ValueError, pd.errors.ParserError) as exc:
        raise ValueError("CSV must contain numeric values with the documented encoding.") from exc
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Dataset is empty or contains missing/infinite values; resolve at source.")
    if set(frame.target.unique()) != {0, 1}:
        raise ValueError("target must contain both classes, encoded as 0 and 1.")
    if not np.equal(frame[CATEGORICAL], np.floor(frame[CATEGORICAL])).all().all():
        raise ValueError("Categorical columns must use integer codes; verify the source codebook.")
    return frame[FEATURES + ["target"]].astype({"target": int})
