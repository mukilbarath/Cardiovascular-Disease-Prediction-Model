# Paper versus original code audit

Sources: supplied five-page PDF `Cardiovascular Disease Prediction_final (1)-1.pdf` and original repository commit `b83565f3d65dc21bb3626f3e75fca3d87061ddd9`. The PDF is the research specification; the script supplies implementation details the paper omits.

| Item | Paper | Original code | Current implementation |
|---|---|---|---|
| Data | 1,025 rows, 13 predictors; 499/526 classes (pp. 2–4) | Hard-coded `/content/heart.csv` | Explicit path, strict schema, no fabricated data |
| Scaling | Five named continuous features (p. 3) | Fit on training set only | Fitted inside each base learner/fold |
| Encoding | Eight named categorical features, drop first (p. 3) | `get_dummies` on full data before split | Training-fold OneHotEncoder; unknown-category warning |
| Split | Stratified 80/20, seed 42 (p. 4) | Matches | Preserved with recorded row indices |
| Learners | DT, AdaBoost, GB → LR (pp. 1–4) | DT depth 5; AdaBoost 100 estimators; GB defaults | Architecture and stated script parameters preserved |
| Internal stacking CV | Out-of-fold predictions, fold count unstated (p. 3) | Implicit sklearn default (5) | Explicit 5-fold, stratified, unshuffled |
| Evaluation CV | 10-fold on training portion (p. 4) | Absent | Separate 10-fold stratified, shuffled with seed 42 |
| Test metrics | 98.05%; matrix 98/2/2/103 (p. 4, Fig. 6) | Prints accuracy/report/matrix | Actual computed metrics saved separately |
| Comparison chart | Fig. 7 compares other models | No comparison evaluation | No unsupported baseline values copied/generated |
| Environment | Python/scikit-learn, no versions | No requirements | Pinned tested environment, recorded run versions |
| Provenance | `heart.csv`, no precise source/version | No dataset | Explicit unresolved provenance |

## Deliberate clarifications and differences

- Fold-local preprocessing prevents held-out category discovery and prevents internal folds from sharing fitted transformations. This is a documented engineering correction, not a claim of bit-for-bit historical reproduction.
- Outer CV shuffling/stratification were not fully specified by the paper. This implementation uses `StratifiedKFold(10, shuffle=True, random_state=42)` and does not confuse it with internal stacking CV.
- Logistic Regression receives seed 42 and `max_iter=1000` for convergence headroom (original default iteration limit). Its default solver remains unchanged.
- Modern scikit-learn AdaBoost uses SAMME; the original unspecified environment may have used SAMME.R. The dependency lock defines the implemented variant. Matching the architecture cannot guarantee historical predictions.
- Numeric/category feature ordering is made explicit. Changed column order and dependency defaults can affect tree tie-breaking.
- EDA plots shown in the paper and baseline comparisons are not treated as reproduced evidence. The new figure outputs are confusion matrix and ROC for actual supplied-data predictions.

## Unresolved evidence

Original CSV, provenance/license, patient grouping, historical dependency versions, exact outer CV settings and fold scores, formal publication metadata, and any external validation are unavailable. None are inferred from the reported accuracy. The paper's template remnants and broad clinical claims are not copied into repository claims.
