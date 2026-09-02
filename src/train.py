"""Train and compare churn classifiers, then persist the winner.

Three candidates are scored with stratified 5-fold CV on the training split:

    logistic_regression  scaled LogisticRegression in a Pipeline (drop_first encoding)
    random_forest        RandomForestClassifier          (all-levels encoding)
    xgboost              XGBClassifier                   (all-levels encoding)

The model with the best mean CV ROC-AUC is refit on the full training split and
scored once against the held-out test set. That test set is touched exactly
once, at the end -- model selection happens entirely inside cross-validation.

    python3 src/train.py

Outputs:
    models/churn_model.joblib     fitted model + the metadata evaluate.py needs
    models/model_comparison.md    the CV table, for the README
"""

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import build_features, load_clean_customers

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
COMPARISON_PATH = MODEL_DIR / "model_comparison.md"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SCORING = ["precision", "recall", "f1", "roc_auc"]

# XGBoost's macOS wheel links against an LLVM OpenMP runtime (libomp.dylib) that
# is not part of the OS. Where it is missing -- `brew install libomp` is the fix --
# the import raises rather than the install failing, so it has to be caught here.
# HistGradientBoostingClassifier stands in: it is also a histogram-based gradient
# boosted tree ensemble and relies only on the OpenMP runtime scikit-learn already
# bundles. Restore libomp and this block silently prefers the real XGBoost again,
# with no other change to the file.
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    XGBOOST_ERROR = None
except Exception as exc:                                   # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingClassifier
    XGBOOST_AVAILABLE = False
    XGBOOST_ERROR = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)


def build_candidates():
    """Return {name: (estimator, encoding)} for the three candidate models.

    `encoding` selects which feature matrix the model is trained on:
      "drop_first"  one level dropped per categorical -- avoids the dummy-variable
                    trap, which matters for the linear model's conditioning.
      "all_levels"  every level kept -- tree ensembles are unaffected by the
                    redundancy and can split on any level directly.
    """
    boosted = (
        XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        )
        if XGBOOST_AVAILABLE else
        HistGradientBoostingClassifier(
            max_iter=400, max_depth=4, learning_rate=0.05,
            random_state=RANDOM_STATE,
        )
    )

    return {
        "logistic_regression": (
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]),
            "drop_first",
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=400, min_samples_leaf=5,
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
            "all_levels",
        ),
        "xgboost" if XGBOOST_AVAILABLE else "hist_gradient_boosting": (
            boosted, "all_levels",
        ),
    }


def cross_validate_candidates(candidates, matrices, y_train, train_idx):
    """Score every candidate with stratified K-fold CV on the training split."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, (estimator, encoding) in candidates.items():
        X_train = matrices[encoding].loc[train_idx]
        scores = cross_validate(estimator, X_train, y_train, cv=cv,
                                scoring=SCORING, n_jobs=-1)
        rows.append({
            "model": name,
            "encoding": encoding,
            **{metric: scores[f"test_{metric}"].mean() for metric in SCORING},
            "roc_auc_std": scores["test_roc_auc"].std(),
        })

    return pd.DataFrame(rows).set_index("model")


def evaluate_on_test(model, X_test, y_test):
    """Score a fitted model once against the held-out test set."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "avg_precision": average_precision_score(y_test, proba),
    }


def write_comparison_markdown(cv_table, best_name, test_metrics, n_train, n_test):
    """Persist the CV comparison as markdown for the README."""
    lines = [
        "# Model comparison",
        "",
        f"Stratified {CV_FOLDS}-fold cross-validation on the training split "
        f"({n_train:,} rows); the test split ({n_test:,} rows) was held out and "
        "scored once, after selection.",
        "",
        "| Model | Encoding | Precision | Recall | F1 | ROC-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in cv_table.iterrows():
        mark = " **(selected)**" if name == best_name else ""
        lines.append(
            f"| `{name}`{mark} | {row.encoding} | {row.precision:.3f} | "
            f"{row.recall:.3f} | {row.f1:.3f} | {row.roc_auc:.4f} |"
        )

    lines += [
        "",
        f"Selected on mean CV ROC-AUC: **`{best_name}`**.",
        "",
        "## Held-out test set",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for metric, value in test_metrics.items():
        lines.append(f"| {metric} | {value:.4f} |")

    lines += [
        "",
        "Metrics above use the default 0.50 decision threshold. "
        "`src/evaluate.py` tunes that threshold for recall and is the source of "
        "the operating point actually used for scoring.",
        "",
    ]
    if not XGBOOST_AVAILABLE:
        lines += [
            "> **Note:** XGBoost could not be loaded in the environment that "
            "produced this table, so scikit-learn's "
            "`HistGradientBoostingClassifier` was substituted as the boosted-tree "
            "candidate. Install the OpenMP runtime (`brew install libomp`) and "
            "re-run `src/train.py` to score the real XGBClassifier.",
            "",
        ]

    COMPARISON_PATH.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None, help="DuckDB file to read")
    args = parser.parse_args()

    df = load_clean_customers(args.db) if args.db else load_clean_customers()

    # Two encodings of the same rows, in the same order, so one split serves both.
    X_linear, y = build_features(df, drop_first=True)
    X_tree, _ = build_features(df, drop_first=False)
    matrices = {"drop_first": X_linear, "all_levels": X_tree}

    train_idx, test_idx = train_test_split(
        X_linear.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    print(f"rows              {len(df):,}")
    print(f"train / test      {len(train_idx):,} / {len(test_idx):,}"
          f"   (churn rate {y_train.mean():.4f} / {y_test.mean():.4f})")
    print(f"features          drop_first={X_linear.shape[1]}  all_levels={X_tree.shape[1]}")
    if not XGBOOST_AVAILABLE:
        print("\n!! XGBoost unavailable -- substituting HistGradientBoostingClassifier")
        print(f"   reason: {XGBOOST_ERROR}")

    candidates = build_candidates()

    print(f"\n=== Stratified {CV_FOLDS}-fold CV on the training split ===")
    cv_table = cross_validate_candidates(candidates, matrices, y_train, train_idx)
    display = cv_table[["encoding"] + SCORING].copy()
    for metric in SCORING:
        display[metric] = display[metric].round(4)
    print(display.to_string())

    best_name = cv_table.roc_auc.idxmax()
    best_estimator, best_encoding = candidates[best_name]
    best_row = cv_table.loc[best_name]
    print(f"\nbest by CV ROC-AUC: {best_name} "
          f"({best_row.roc_auc:.4f} +/- {best_row.roc_auc_std:.4f})")

    X = matrices[best_encoding]
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    best_estimator.fit(X_train, y_train)

    print("\n=== Held-out test set (threshold 0.50) ===")
    test_metrics = evaluate_on_test(best_estimator, X_test, y_test)
    for metric, value in test_metrics.items():
        print(f"  {metric:<16} {value:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    bundle = {
        "model": best_estimator,
        "model_name": best_name,
        "encoding": best_encoding,
        "drop_first": best_encoding == "drop_first",
        "feature_names": list(X.columns),
        "train_index": np.asarray(train_idx),
        "test_index": np.asarray(test_idx),
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "cv_roc_auc": float(best_row.roc_auc),
        "test_metrics": test_metrics,
        "xgboost_available": XGBOOST_AVAILABLE,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    joblib.dump(bundle, MODEL_PATH)

    write_comparison_markdown(cv_table, best_name, test_metrics,
                              len(train_idx), len(test_idx))

    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"\nsaved {MODEL_PATH.relative_to(ROOT)}  ({size_kb:,.1f} KB)")
    print(f"saved {COMPARISON_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
