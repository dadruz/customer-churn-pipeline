"""Evaluate the saved churn model and export scored customers.

Loads `models/churn_model.joblib`, reproduces the exact train/test split it was
fitted with, sweeps the decision threshold, recommends an operating point, and
writes a scored file for every customer.

    python3 src/evaluate.py
    python3 src/evaluate.py --threshold 0.35    # override the recommendation

Output:
    data/processed/scored_customers.csv
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from features import build_features, load_clean_customers

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
SCORED_PATH = ROOT / "data" / "processed" / "scored_customers.csv"

REPORT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6]

# The recommendation rule, stated up front so the choice is reproducible rather
# than eyeballed: take the threshold that catches the most churners, subject to
# at least half of the customers it flags being genuine churners.
#
# The asymmetry is deliberate. A false negative is a customer who leaves
# unflagged -- the full value of the account is lost. A false positive is a
# retention offer sent to someone who would have stayed anyway: the cost is the
# offer, which is a fraction of an account. Recall is therefore worth more than
# precision here, but precision cannot fall so far that the campaign is mostly
# waste, and 0.50 is the point where a majority of contacts are still on target.
# Tune PRECISION_FLOOR once the real offer cost and account value are known.
PRECISION_FLOOR = 0.50
SEARCH_GRID = np.round(np.arange(0.20, 0.71, 0.01), 2)


def load_bundle(path=MODEL_PATH):
    """Load the fitted model and the metadata train.py saved beside it."""
    if not path.exists():
        raise SystemExit(f"No model at {path} -- run `python3 src/train.py` first.")
    return joblib.load(path)


def threshold_metrics(y_true, proba, threshold):
    """Precision / recall / F1 and the confusion counts at one threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "flagged": int(pred.sum()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def recommend_threshold(y_true, proba):
    """Highest-recall threshold whose precision still clears PRECISION_FLOOR.

    Returns (threshold, table) where `table` is every candidate scored, so the
    caller can show the trade-off rather than just the verdict.
    """
    table = pd.DataFrame(threshold_metrics(y_true, proba, t) for t in SEARCH_GRID)
    eligible = table[table.precision >= PRECISION_FLOOR]

    if eligible.empty:                      # nothing clears the floor: fall back
        return float(table.loc[table.f1.idxmax(), "threshold"]), table

    # Ties on recall are broken toward the higher precision, i.e. the higher
    # threshold, since it flags fewer customers for the same catch rate.
    best = eligible.sort_values(["recall", "precision"], ascending=[False, False]).iloc[0]
    return float(best.threshold), table


def print_confusion(y_true, proba, threshold):
    """Print the confusion matrix at one threshold, with rates."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

    print(f"\n=== Confusion matrix @ {threshold:.2f} (test set, n={len(y_true):,}) ===")
    print("                  predicted stay   predicted churn")
    print(f"  actual stay        {tn:>10,}      {fp:>12,}")
    print(f"  actual churn       {fn:>10,}      {tp:>12,}")
    print(f"\n  caught {tp:,} of {tp + fn:,} churners ({tp / (tp + fn):.1%})")
    print(f"  {fp:,} false alarms out of {tn + fp:,} stayers ({fp / (tn + fp):.1%})")
    print(f"  {tp + fp:,} customers flagged; {tp / (tp + fp):.1%} of them churn")


def out_of_fold_probabilities(bundle, X, y):
    """Out-of-fold churn probabilities for the training split.

    Each training row is predicted by a fold model that never saw it, which
    makes these probabilities honest enough to tune the decision threshold on --
    without spending the test set to do it.
    """
    train_idx = pd.Index(bundle["train_index"])
    cv = StratifiedKFold(n_splits=bundle["cv_folds"], shuffle=True,
                         random_state=bundle["random_state"])
    proba = cross_val_predict(
        clone(bundle["model"]), X.loc[train_idx], y.loc[train_idx],
        cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]
    return pd.Series(proba, index=train_idx)


def score_all_customers(X, y, train_proba, test_proba, threshold):
    """Assemble the scored export, keeping every prediction out-of-sample.

    Train rows carry their out-of-fold probability, so each was predicted by a
    model that never saw it; test rows carry the saved model's probability, and
    it never saw them either. The two halves are therefore *equally honest but
    not identically produced* -- fold models are each fitted on 4/5 of the
    training data, the saved model on all of it -- so the probabilities are
    comparable in aggregate while differing slightly in provenance. The `split`
    column records which path produced each row; filter on it before using this
    file to measure anything.
    """
    proba = pd.concat([train_proba, test_proba]).sort_index()

    split = pd.Series("train", index=X.index)
    split.loc[test_proba.index] = "test"

    return pd.DataFrame({
        "churn_probability": proba.round(6),
        "predicted_churn": (proba >= threshold).astype(int),
        "actual_churn": y,
        "split": split,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the recommended threshold")
    args = parser.parse_args()

    bundle = load_bundle()
    df = load_clean_customers()
    X, y = build_features(df, drop_first=bundle["drop_first"])
    X = X[bundle["feature_names"]]          # guard against column drift

    test_idx = pd.Index(bundle["test_index"])
    train_idx = pd.Index(bundle["train_index"])
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    proba = bundle["model"].predict_proba(X_test)[:, 1]

    # Tune the operating point on out-of-fold training predictions. Choosing it
    # on the test set would let the reported precision/recall inherit that
    # choice, and the held-out numbers would read better than they deserve.
    train_proba = out_of_fold_probabilities(bundle, X, y)
    y_train = y.loc[train_idx]

    print(f"model             {bundle['model_name']}  ({bundle['encoding']} encoding)")
    print(f"trained with      sklearn {bundle['sklearn_version']}, "
          f"python {bundle['python_version']}")
    print(f"CV ROC-AUC        {bundle['cv_roc_auc']:.4f}")
    print(f"test ROC-AUC      {roc_auc_score(y_test, proba):.4f}  (n={len(y_test):,})")

    print("\n=== Threshold analysis (test set) ===")
    report = pd.DataFrame(
        threshold_metrics(y_test, proba, t) for t in REPORT_THRESHOLDS
    ).set_index("threshold")
    print(report[["precision", "recall", "f1", "flagged", "tp", "fp", "fn"]]
          .round(4).to_string())

    recommended, _ = recommend_threshold(y_train, train_proba.to_numpy())
    chosen = args.threshold if args.threshold is not None else recommended
    source = "override" if args.threshold is not None else "recommended"

    sel = threshold_metrics(y_train, train_proba.to_numpy(), chosen)
    rec = threshold_metrics(y_test, proba, chosen)
    print(f"\n{source}: threshold {chosen:.2f} — highest recall with precision "
          f">= {PRECISION_FLOOR:.2f}, chosen on the training folds")
    print(f"  on training folds  precision {sel['precision']:.4f}   "
          f"recall {sel['recall']:.4f}   f1 {sel['f1']:.4f}")
    print(f"  on held-out test   precision {rec['precision']:.4f}   "
          f"recall {rec['recall']:.4f}   f1 {rec['f1']:.4f}")

    print_confusion(y_test, proba, chosen)

    scored = score_all_customers(
        X, y, train_proba, pd.Series(proba, index=test_idx), chosen)
    scored.insert(0, "customerID", df["customerID"])

    SCORED_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(SCORED_PATH, index=False)

    print(f"\n=== Scored export ===")
    print(f"  rows                  {len(scored):,}")
    print(f"  flagged @ {chosen:.2f}          {int(scored.predicted_churn.sum()):,}")
    print(f"  actual churners       {int(scored.actual_churn.sum()):,}")
    print(f"  overall recall        "
          f"{recall_score(scored.actual_churn, scored.predicted_churn):.4f}")
    print(f"  overall precision     "
          f"{precision_score(scored.actual_churn, scored.predicted_churn):.4f}")
    print(f"  saved                 {SCORED_PATH.relative_to(ROOT)} "
          f"({SCORED_PATH.stat().st_size / 1024:,.1f} KB)")
    print()
    print(scored.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
