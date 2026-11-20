# Dataset and provenance

The original `heart.csv` is **not supplied**. The paper describes 1,025 rows, 13 predictors plus a binary target, with 499 class-0 and 526 class-1 rows (pp. 2–4). A matching filename, size or class distribution alone does not establish dataset identity.

Obtain the exact research CSV from the authors or their original source. Record its source URL/provider, retrieval date, original version, SHA-256, permitted uses/license, codebook and any preprocessing history in a local provenance record. Confirm whether rows are unique patients, repeated observations or duplicates. Do not assume a Kaggle/UCI variant is identical. No download URL or data license is asserted here.

Place the verified CSV at `data/heart.csv`, or pass `--data PATH`. Data and outputs are excluded from Git. Do not commit patient records or row-level predictions. Schema validation is not provenance verification.

| Columns | Processing |
|---|---|
| `age, trestbps, chol, thalach, oldpeak` | Numeric, finite; StandardScaler |
| `cp, restecg, slope, ca, thal, sex, exang, fbs` | Integer-coded categories; one-hot, drop first |
| `target` | Binary 0 (absence), 1 (presence); both required |

The loader requires exactly these columns, rejects duplicate headers, missing/non-numeric/infinite values, and does not silently impute or drop rows. Category ranges/units must be verified against the original codebook; the paper does not establish a complete codebook. Unknown categories at evaluation time are encoded as all zeros with a warning. At least 10 rows per class in the training split are needed for outer CV; this is a software minimum, not adequate study size.

The run records duplicate-row counts and feature-vector overlap between training and test. It preserves rows for methodological traceability rather than silently deduplicating. Investigate duplicates and patient overlap before interpreting results; a deduplicated or grouped experiment is a distinct analysis requiring documentation.
