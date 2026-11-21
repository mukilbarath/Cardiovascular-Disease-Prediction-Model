# Architecture

```mermaid
flowchart TD
    CSV[Verified original CSV] --> V[Schema and data audit]
    V --> S[Stratified 80/20 split: seed 42]
    S --> T[80% training]
    S --> H[20% held-out test]
    T --> CV[10-fold evaluation of entire stack]
    T --> FIT[Fit final stack on all training rows]
    FIT --> E[Test evaluation]
    H --> E
    CV --> R[Computed metrics and provenance]
    E --> R
    E --> F[Confusion matrix and ROC figures]
```

Inside each stack: separate preprocessing pipelines feed Decision Tree, Gradient Boosting and AdaBoost. Five-fold out-of-fold positive probabilities train Logistic Regression. Test rows never train preprocessing or the meta-classifier.

`data.py` validates source input; `model.py` constructs the complete estimator; `evaluation.py` computes metrics/figures; `cli.py` owns splitting, CV, final fitting and run metadata. `reports/paper_reported.json` is static source evidence and is never read by training code. Tests generate local synthetic records only.
