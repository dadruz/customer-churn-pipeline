"""Feature engineering for the Telco churn model.

Every function takes a DataFrame and returns a new one, so the steps compose and
none of them mutate the caller's frame. `build_features` chains them and hands
back a model-ready `(X, y)` pair.

The design decisions here are the ones settled in `notebooks/01_eda.ipynb`;
each is cross-referenced to the section that justifies it.

    python3 src/features.py
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "churn.duckdb"

TARGET = "churn_flag"

# The six optional services. `sql/02_clean.sql` has already collapsed the
# "No internet service" / "No phone service" placeholders to plain "No", so a
# simple == "Yes" test is exhaustive here.
ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-72"]

# Dropped from the model matrix, never from the source table.
#   customerID  - an identifier, not a feature.
#   Churn       - the target in text form; churn_flag is the modelled version.
#   TotalCharges- see the note in encode_features().
NON_FEATURES = ["customerID", "Churn", TARGET, "TotalCharges"]


def load_clean_customers(db_path=DEFAULT_DB):
    """Read the full `clean_customers` table from the DuckDB file.

    Args:
        db_path: Path to the DuckDB database built by `src/run_sql.py`.

    Returns:
        DataFrame with one row per customer.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute("SELECT * FROM clean_customers").df()
    finally:
        con.close()


def add_tenure_bucket(df):
    """Add `tenure_bucket`, a four-level ordered band of customer lifetime.

    Bands are 0-12, 13-24, 25-48 and 49-72 months. EDA section 3 shows churn
    falling monotonically across them (47.4% -> 28.7% -> 20.4% -> 9.5%), so the
    bins stay coarse enough to read while preserving that ordering. The raw
    `tenure` column is kept alongside: the bucket gives a tree cheap access to
    the lifecycle stage, the continuous column keeps the fine detail.

    Args:
        df: DataFrame containing a numeric `tenure` column.

    Returns:
        A copy of `df` with an ordered categorical `tenure_bucket` column.
    """
    out = df.copy()
    out["tenure_bucket"] = pd.cut(
        out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, ordered=True
    )
    return out


def add_service_count(df):
    """Add `service_count`, the number of optional add-on services held (0-6).

    EDA section 6 finds the two support services (OnlineSecurity, TechSupport)
    cut churn by ~16 points each while the two streaming services move it
    slightly the wrong way. The count compresses overall engagement into one
    numeric column; the individual indicators survive one-hot encoding, so the
    model can still separate support from entertainment.

    Args:
        df: DataFrame containing the six ADDON_SERVICES columns as Yes/No text.

    Returns:
        A copy of `df` with an integer `service_count` column.
    """
    out = df.copy()
    out["service_count"] = sum((out[col] == "Yes").astype(int)
                               for col in ADDON_SERVICES)
    return out


def add_charges_per_tenure(df):
    """Add the realised-vs-current billing rate features.

    `TotalCharges / tenure` is the customer's *average historical* monthly bill;
    `MonthlyCharges` is what they pay *now*. The ratio between them is the only
    genuinely new information in TotalCharges (EDA section 7): a value near 1.0
    means a stable plan, while a value away from 1.0 means the plan changed
    mid-life -- an upgrade or downgrade that a raw dollar total cannot express.

    tenure = 0 is handled explicitly rather than by dividing: those 11 customers
    have never been billed (TotalCharges is 0.0 by construction in
    sql/02_clean.sql), so their best available average rate is the monthly rate
    itself, giving a ratio of exactly 1.0 instead of a NaN or an infinity.

    Args:
        df: DataFrame with numeric `tenure`, `MonthlyCharges`, `TotalCharges`.

    Returns:
        A copy of `df` with `charges_per_tenure` and `charges_ratio` columns.
    """
    out = df.copy()

    billed = out["tenure"] > 0
    out["charges_per_tenure"] = np.where(
        billed,
        out["TotalCharges"] / out["tenure"].where(billed, 1),
        out["MonthlyCharges"],           # never billed -> fall back to the rate
    )

    # MonthlyCharges has a floor of $18.25 in this dataset, but guard anyway so
    # the function stays safe on a future extract.
    monthly = out["MonthlyCharges"].replace(0, np.nan)
    out["charges_ratio"] = (out["charges_per_tenure"] / monthly).fillna(1.0)

    return out


def encode_features(df, drop_first=True):
    """One-hot encode the categoricals and split off the target.

    TotalCharges is dropped here. EDA section 7 measured
    corr(TotalCharges, tenure x MonthlyCharges) = 0.9996 -- it is very nearly the
    arithmetic product of two columns already in the matrix, and correlates
    0.826 with tenure by itself. Keeping it adds a third collinear input that
    inflates coefficient variance in a linear model and splits importance across
    correlated columns in a tree model, masking tenure's real contribution. The
    part of it that is *not* redundant is preserved by add_charges_per_tenure()
    as a bounded ratio rather than a third copy of the same quantity.

    Args:
        df: DataFrame after the add_* steps have run.
        drop_first: Drop one level per categorical to avoid the dummy-variable
            trap. True suits linear models; set False for tree ensembles, which
            are unharmed by the redundancy and can split on any level directly.

    Returns:
        (X, y) where X is the all-numeric feature matrix and y is `churn_flag`.
    """
    y = df[TARGET].astype(int)
    features = df.drop(columns=[c for c in NON_FEATURES if c in df.columns])

    # Select by "not numeric" rather than by dtype name: pandas 3 splits the old
    # object dtype into object/str, and this stays correct across both versions.
    categorical = [c for c in features.columns
                   if not pd.api.types.is_numeric_dtype(features[c])]
    X = pd.get_dummies(features, columns=categorical,
                       drop_first=drop_first, dtype=int)

    return X, y


def build_features(df, drop_first=True):
    """Run the full feature pipeline: bucket -> service count -> charges -> encode.

    Args:
        df: Raw `clean_customers` DataFrame.
        drop_first: Passed through to encode_features().

    Returns:
        (X, y) ready for a scikit-learn or XGBoost estimator.
    """
    return encode_features(
        add_charges_per_tenure(add_service_count(add_tenure_bucket(df))),
        drop_first=drop_first,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB),
                        help="DuckDB file to read (default: data/churn.duckdb)")
    parser.add_argument("--keep-all-levels", action="store_true",
                        help="Skip drop_first (one column per category level)")
    args = parser.parse_args()

    df = load_clean_customers(args.db)
    X, y = build_features(df, drop_first=not args.keep_all_levels)

    print(f"source          clean_customers  {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"X shape         {X.shape}")
    print(f"y shape         {y.shape}   positive rate {y.mean():.4f}")
    print(f"dtypes          {sorted({str(t) for t in X.dtypes})}")
    print(f"any nulls       {bool(X.isna().any().any())}")

    print(f"\ncolumns ({X.shape[1]}):")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i:>2}. {col}")

    engineered = ["service_count", "charges_per_tenure", "charges_ratio"]
    print("\nengineered numeric features:")
    print(X[engineered].describe().round(3).to_string())


if __name__ == "__main__":
    main()
