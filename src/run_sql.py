"""Run the Phase 2 SQL pipeline against a local DuckDB file.

Executes every sql/*.sql script in filename order, then prints row counts,
the data quality gate and a numeric summary as a sanity check.

    python3 src/run_sql.py            # writes data/churn.duckdb
    python3 src/run_sql.py --db :memory:

Exits non-zero if any data quality check reports failures.
"""

import argparse
import os
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "sql"
DEFAULT_DB = ROOT / "data" / "churn.duckdb"

NUMERIC_COLUMNS = ("tenure", "MonthlyCharges", "TotalCharges")


def rule(title):
    print(f"\n{title}\n{'-' * len(title)}")


def run_scripts(con):
    scripts = sorted(SQL_DIR.glob("*.sql"))
    if not scripts:
        sys.exit(f"No .sql files found in {SQL_DIR}")

    rule("Executing SQL")
    for script in scripts:
        con.execute(script.read_text())
        print(f"  ok  {script.relative_to(ROOT)}")


def print_row_counts(con):
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()]

    rule("Row counts")
    for table in tables:
        count = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        print(f"  {table:<24} {count:>8,}")


def print_quality_checks(con):
    checks = con.execute(
        "SELECT check_name, failures FROM data_quality_checks ORDER BY check_name"
    ).fetchall()

    rule("Data quality checks")
    for name, failures in checks:
        status = "PASS" if failures == 0 else "FAIL"
        print(f"  [{status}] {name:<30} {failures:>6,}")

    return sum(failures for _, failures in checks)


def print_summary(con):
    total, churned, churn_rate = con.execute(
        """
        SELECT COUNT(*),
               SUM(churn_flag),
               ROUND(100.0 * AVG(churn_flag), 2)
        FROM clean_customers
        """
    ).fetchone()

    rule("Summary: clean_customers")
    print(f"  total rows               {total:>10,}")
    print(f"  churned (churn_flag=1)   {churned:>10,}")
    print(f"  retained (churn_flag=0)  {total - churned:>10,}")
    print(f"  churn rate               {churn_rate:>9}%")

    rule("Numeric columns")
    print(f"  {'column':<16}{'min':>12}{'max':>12}{'mean':>12}{'nulls':>8}")
    for column in NUMERIC_COLUMNS:
        lo, hi, mean, nulls = con.execute(
            f'''SELECT MIN("{column}"), MAX("{column}"),
                       ROUND(AVG("{column}"), 2),
                       COUNT(*) FILTER (WHERE "{column}" IS NULL)
                FROM clean_customers'''
        ).fetchone()
        print(f"  {column:<16}{lo:>12,}{hi:>12,}{mean:>12,}{nulls:>8,}")

    # The 11 blank TotalCharges values should all be brand-new customers.
    zero_total, zero_total_new = con.execute(
        """
        SELECT COUNT(*) FILTER (WHERE TotalCharges = 0.0),
               COUNT(*) FILTER (WHERE TotalCharges = 0.0 AND tenure = 0)
        FROM clean_customers
        """
    ).fetchone()
    rule("TotalCharges = 0.0 (was blank in raw)")
    print(f"  rows                     {zero_total:>10,}")
    print(f"  of which tenure = 0      {zero_total_new:>10,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB),
                        help="DuckDB file to build (default: data/churn.duckdb)")
    args = parser.parse_args()

    # SQL scripts reference the CSV by repo-relative path.
    os.chdir(ROOT)

    print(f"database: {args.db}")
    con = duckdb.connect(args.db)
    try:
        run_scripts(con)
        print_row_counts(con)
        failures = print_quality_checks(con)
        print_summary(con)
    finally:
        con.close()

    if failures:
        sys.exit(f"\n{failures} data quality failure(s) — see checks above.")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
