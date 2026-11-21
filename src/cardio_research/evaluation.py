"""Metrics and figures calculated only from the supplied run's predictions."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def metrics(y_true, prediction, probability):
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, prediction, labels=[0, 1], output_dict=True, zero_division=0
        ),
    }


def figures(y_true, prediction, probability, directory):
    ConfusionMatrixDisplay.from_predictions(y_true, prediction, labels=[0, 1], cmap="Blues")
    plt.title("Supplied dataset: held-out test confusion matrix")
    plt.tight_layout()
    plt.savefig(directory / "confusion_matrix.png", dpi=160)
    plt.close()
    RocCurveDisplay.from_predictions(y_true, probability)
    plt.title("Supplied dataset: held-out test ROC")
    plt.tight_layout()
    plt.savefig(directory / "roc_curve.png", dpi=160)
    plt.close()
