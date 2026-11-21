"""Run independent training-set evaluation, then evaluate the held-out test set once."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .data import FEATURES, load_data
from .evaluation import figures, metrics
from .model import SEED, build_model


def run(data_path: Path, output: Path):
    frame = load_data(data_path)
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a fresh run directory.")
    train, test = train_test_split(
        np.arange(len(frame)), test_size=0.2, random_state=SEED, stratify=frame.target
    )
    X, y = frame[FEATURES], frame.target
    if y.iloc[train].value_counts().min() < 10 or y.iloc[test].nunique() != 2:
        raise ValueError(
            "Each training class needs at least 10 rows and both test classes are required."
        )
    duplicate_rows = int(frame.duplicated().sum())
    hashes = pd.util.hash_pandas_object(X, index=False)
    overlap = len(set(hashes.iloc[train]) & set(hashes.iloc[test]))
    if duplicate_rows or overlap:
        warnings.warn(
            "Duplicate records/features detected. Row splits may overestimate generalization; "
            "see dataset_audit in metrics.json and docs/METHODOLOGY.md.",
            stacklevel=2,
        )
    model = build_model()
    scores = cross_validate(
        model,
        X.iloc[train],
        y.iloc[train],
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED),
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        error_score="raise",
        n_jobs=1,
    )
    model.fit(X.iloc[train], y.iloc[train])
    prediction = model.predict(X.iloc[test])
    probability = model.predict_proba(X.iloc[test])[:, 1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    result = {
        "status": "computed_from_supplied_dataset",
        "paper_result_reproduced": "not established; provenance requires independent verification",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "dataset_audit": {
            "rows": len(frame),
            "class_counts": y.value_counts().to_dict(),
            "duplicate_rows": duplicate_rows,
            "train_test_shared_feature_vectors": overlap,
        },
        "config": {
            "random_state": SEED,
            "test_size": 0.2,
            "outer_cv_folds": 10,
            "outer_cv_shuffle": True,
            "stacking_cv_folds": 5,
            "stacking_cv_shuffle": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": commit,
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "packages": {
                p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]
            },
        },
        "model_parameters": {k: repr(v) for k, v in model.get_params(deep=True).items()},
        "training_cv": {
            k.removeprefix("test_"): {
                "folds": v.tolist(),
                "mean": float(v.mean()),
                "std": float(v.std(ddof=0)),
            }
            for k, v in scores.items()
            if k.startswith("test_")
        },
        "held_out_test": metrics(y.iloc[test], prediction, probability),
    }
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "split_indices.json").write_text(
        json.dumps({"train": train.tolist(), "test": test.tolist()}, indent=2), encoding="utf-8"
    )
    figures(y.iloc[test], prediction, probability, output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/heart.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/run-001"))
    args = parser.parse_args(argv)
    try:
        run(args.data, args.output)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Completed supplied-dataset evaluation: {args.output / 'metrics.json'}")
    return 0
