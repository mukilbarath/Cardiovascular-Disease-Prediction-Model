# Methodology

1. Validate the supplied numeric CSV; no imputation, oversampling, feature selection or tuning.
2. Split raw row positions 80/20 with stratification and seed 42. Test data stays outside training CV.
3. Run 10-fold stratified training-only CV, shuffled with seed 42. Each fold trains a new entire stack. Report accuracy, positive-class precision/recall/F1 and ROC-AUC per fold, mean and population standard deviation; the standard deviation is not a confidence interval.
4. In each stack fit, three independent pipelines standardize five numeric fields and one-hot encode eight categorical fields (`drop='first'`). A depth-5 Decision Tree, 100-estimator AdaBoost and default Gradient Boosting generate positive-class probabilities through internal five-fold stratified, unshuffled CV. Each internal fold refits its preprocessing. Logistic Regression learns from those out-of-fold probabilities. Base pipelines are then refitted on all rows available to that stack.
5. Fit the final stack on the full 80% training portion and evaluate the untouched 20% test portion once. Use estimator `predict` for labels and positive-class probabilities for ROC-AUC. Confusion matrices use labels `[0, 1]`, rows actual and columns predicted. Classification reports include both classes, macro and weighted averages.

All applicable random states are 42; execution uses one job. Remaining estimator defaults are defined by pinned scikit-learn and exported in run metadata. Identical seeds do not guarantee identical floating-point outcomes across all platforms. The original software environment is unknown.

## Integrity and interpretation

Data hashes bind outputs to the exact CSV bytes, and split positions bind them to row order. Model parameters and versions support rerunning, but do not prove dataset provenance. Local outputs may include sensitive study metadata and should be reviewed before sharing. Models are not serialized or loaded from untrusted pickle files.

Exact duplicates and matching feature vectors across train/test are reported with a warning. No rows are automatically removed because that changes the study. Repeated patients can also cause leakage even without identical feature vectors; reliable patient identifiers are necessary for a grouped analysis. The primary protocol remains the requested row-stratified split, so duplicates/patient overlap limit interpretation of its metrics. Training CV also uses row splits and shares this limitation.

Neither the paper's accuracy nor successful software checks establish prospective performance, calibration, external generalization, fairness, safety, or clinical usefulness. No clinical deployment claim is made. A valid follow-up requires verified data provenance, patient-level separation where appropriate, independent external evaluation and suitable ethical review.

## Library references

- [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html): internal out-of-fold prediction and meta-classifier fitting.
- [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html): current algorithm/defaults; historical versions can differ.

See [audit](AUDIT.md) for paper page references and implementation deviations.
