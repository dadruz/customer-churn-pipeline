# Model comparison

Stratified 5-fold cross-validation on the training split (5,634 rows); the test split (1,409 rows) was held out and scored once, after selection.

| Model | Encoding | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| `logistic_regression` **(selected)** | drop_first | 0.665 | 0.532 | 0.591 | 0.8455 |
| `random_forest` | all_levels | 0.678 | 0.496 | 0.572 | 0.8448 |
| `xgboost` | all_levels | 0.646 | 0.516 | 0.573 | 0.8419 |

Selected on mean CV ROC-AUC: **`logistic_regression`**.

## Held-out test set

| Metric | Value |
|---|---|
| accuracy | 0.7977 |
| precision | 0.6488 |
| recall | 0.5187 |
| f1 | 0.5765 |
| roc_auc | 0.8413 |
| avg_precision | 0.6350 |

Metrics above use the default 0.50 decision threshold. `src/evaluate.py` tunes that threshold for recall and is the source of the operating point actually used for scoring.
