import json

import numpy as np
import pytest
from sklearn.base import clone

from cardio_research.cli import main, run
from cardio_research.data import FEATURES, load_data
from cardio_research.evaluation import metrics
from cardio_research.model import build_model, preprocessing


def test_missing_dataset_fails_without_outputs(tmp_path, capsys):
    output = tmp_path / "output"
    assert main(["--data", str(tmp_path / "absent.csv"), "--output", str(output)]) == 2
    assert "Dataset not found" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("invalid", ["missing", "extra", "nan", "infinite", "target", "text"])
def test_invalid_data_rejected(synthetic_frame, tmp_path, invalid):
    frame = synthetic_frame.copy()
    if invalid == "missing":
        frame = frame.drop(columns="age")
    elif invalid == "extra":
        frame["patient_id"] = 1
    elif invalid == "nan":
        frame.loc[0, "age"] = np.nan
    elif invalid == "infinite":
        frame.loc[0, "age"] = np.inf
    elif invalid == "target":
        frame["target"] = 2
    else:
        frame["age"] = "unknown"
    path = tmp_path / "synthetic.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_data(path)


def test_duplicate_header_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("age,age,target\n1,2,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="once each"):
        load_data(path)


def test_preprocessing_does_not_learn_from_holdout(synthetic_frame):
    train = synthetic_frame.iloc[:80][FEATURES]
    held_out = synthetic_frame.iloc[80:][FEATURES].copy()
    held_out["age"] = 99999
    held_out["cp"] = 999
    transform = preprocessing().fit(train)
    mean_before = transform.named_transformers_["numeric"].mean_.copy()
    with pytest.warns(UserWarning, match="unknown categories"):
        transformed = transform.transform(held_out)
    np.testing.assert_array_equal(mean_before, transform.named_transformers_["numeric"].mean_)
    assert np.isfinite(transformed).all()
    assert 999 not in transform.named_transformers_["categorical"].categories_[0]


def test_model_repeatability_and_independent_preprocessors(synthetic_frame):
    X, y = synthetic_frame[FEATURES], synthetic_frame.target
    first = build_model().fit(X, y)
    second = clone(first).fit(X, y)
    np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X))
    assert len({id(p.named_steps["preprocess"]) for p in first.estimators_}) == 3


def test_known_metrics():
    result = metrics([0, 0, 1, 1], [0, 1, 0, 1], [0.1, 0.8, 0.4, 0.9])
    assert result["accuracy"] == 0.5
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["roc_auc"] == 0.75


def test_full_workflow_synthetic_only(synthetic_frame, tmp_path):
    path = tmp_path / "synthetic_software_fixture.csv"
    synthetic_frame.to_csv(path, index=False)
    output = tmp_path / "run"
    result = run(path, output)
    split = json.loads((output / "split_indices.json").read_text())
    assert len(split["train"]) == 80 and len(split["test"]) == 20
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["train"] + split["test"]) == set(range(100))
    assert len(result["training_cv"]["accuracy"]["folds"]) == 10
    assert sum(map(sum, result["held_out_test"]["confusion_matrix"])) == 20
    assert len(result["dataset_sha256"]) == 64
    for name in ["confusion_matrix.png", "roc_curve.png"]:
        assert (output / name).read_bytes().startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="already exists"):
        run(path, output)


def test_insufficient_training_class_fails(synthetic_frame, tmp_path):
    path = tmp_path / "small.csv"
    synthetic_frame.iloc[:16].to_csv(path, index=False)
    with pytest.raises(ValueError, match="at least 10"):
        run(path, tmp_path / "run")
