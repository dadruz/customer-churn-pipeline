"""Airflow DAG for the customer churn batch scoring pipeline."""

import os
import shutil
import sys
from pathlib import Path

import pendulum

# Airflow 3 moved BashOperator into the standard provider and the DAG class into
# the task SDK. Both import paths are tried so this file loads on 2.x and 3.x.
try:
    from airflow.sdk import DAG
except ImportError:                                        # Airflow 2.x
    from airflow import DAG

try:
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:                                        # Airflow 2.x
    from airflow.operators.bash import BashOperator

# This file is usually symlinked into $AIRFLOW_HOME/dags. resolve() follows the
# symlink back to the real checkout, so the project root is found either way.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Airflow generally runs in its own virtualenv, pinned separately from the data
# stack -- the tasks below need duckdb/pandas/scikit-learn, which Airflow does
# not. CHURN_PYTHON_BIN names the interpreter that has them; it defaults to
# whichever interpreter is parsing this DAG, which is correct for the simple
# case where one environment holds both.
PYTHON_BIN = os.environ.get("CHURN_PYTHON_BIN") or sys.executable

DAG_DOC = """
### Customer churn — batch scoring pipeline

Rebuilds the analytical tables from the raw CSV, verifies the engineered
features still assemble, and scores every customer with the **already-trained**
model committed at `models/churn_model.joblib`.

```
ingest_and_clean  ->  build_features  ->  score_customers
```

| Task | Script | Produces |
|---|---|---|
| `ingest_and_clean` | `src/run_sql.py` | `data/churn.duckdb` (`raw_customers`, `clean_customers`, `data_quality_checks`) |
| `build_features` | `src/features.py` | nothing on disk — a build check on the feature matrix |
| `score_customers` | `src/evaluate.py` | `data/processed/scored_customers.csv` |

#### Batch scoring, not scheduled retraining

The DAG deliberately **does not** call `src/train.py`. Training is a deliberate,
reviewed act: it changes which model is in production, and it needs a human to
compare candidates, check the metrics, and decide whether the new model is
actually better. Retraining on a timer instead silently swaps the model
underneath the business, and a bad data week becomes a bad model with nothing
between it and production.

So the split is: **the model is a versioned artifact** (trained by hand, reviewed,
committed), and **this DAG applies it**. New customer data arrives, gets cleaned,
gets scored by a known model whose evaluation numbers are on record. Retraining
happens out-of-band, and promoting a new model is a commit — visible in review,
revertible like any other change.

#### The data quality gate

`src/run_sql.py` builds a `data_quality_checks` table — duplicate IDs, nulls in
key columns, unmapped categorical levels, row count preserved from raw — and
exits **non-zero** if any check reports failures. Because that is a real process
exit code, `ingest_and_clean` fails, and the downstream tasks never run: bad data
cannot reach the model or the scored export. `retries=0` is deliberate here; a
failed quality check is not transient, and retrying only obscures it.
"""

BATCH_DOC = {
    "ingest_and_clean": (
        "Runs `src/run_sql.py`: loads the raw CSV into DuckDB as all-VARCHAR, "
        "builds the typed `clean_customers` table, and runs the data quality "
        "gate. **Exits non-zero on any failed check**, which fails this task and "
        "halts the DAG before anything downstream can consume bad data."
    ),
    "build_features": (
        "Runs `src/features.py`: assembles the model-ready matrix (tenure "
        "buckets, add-on service count, charges ratio, one-hot encoding) and "
        "prints its shape. Writes nothing — it is a build check that the feature "
        "pipeline still runs against the freshly rebuilt tables."
    ),
    "score_customers": (
        "Runs `src/evaluate.py`: loads `models/churn_model.joblib`, applies the "
        "tuned decision threshold, and writes "
        "`data/processed/scored_customers.csv`."
    ),
}


def task(dag, task_id, script):
    """A pipeline step that runs one project script as a subprocess.

    BashOperator rather than PythonOperator, on purpose. Each script is a CLI
    entry point that signals failure with its **exit code** -- `run_sql.py`
    exits non-zero when the data quality gate fails. BashOperator maps a
    non-zero exit straight onto task failure, so that contract is preserved for
    free. A PythonOperator would have to import each module and catch the
    SystemExit its `main()` raises, turning a clean process boundary into
    exception plumbing, and it would run the task inside the scheduler's
    interpreter -- which is exactly the environment that may not have duckdb or
    scikit-learn installed.
    """
    return BashOperator(
        dag=dag,
        task_id=task_id,
        bash_command=f"cd {PROJECT_ROOT!s} && {PYTHON_BIN!s} {script}",
        doc_md=BATCH_DOC[task_id],
    )


with DAG(
    dag_id="churn_batch_pipeline",
    description="Rebuild churn tables from raw CSV and score every customer",
    schedule=None,          # manual trigger only -- no backfill semantics here
    catchup=False,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    default_args={
        # A failed data quality check is a real defect, not a transient error.
        # Retrying would re-run the same deterministic SQL against the same
        # file and fail again, while making the failure look flaky in the UI.
        "retries": 0,
        "depends_on_past": False,
    },
    tags=["churn", "batch-scoring", "duckdb"],
    doc_md=DAG_DOC,
    max_active_runs=1,      # the tasks read and write one shared DuckDB file
) as dag:
    ingest_and_clean = task(dag, "ingest_and_clean", "src/run_sql.py")
    build_features = task(dag, "build_features", "src/features.py")
    score_customers = task(dag, "score_customers", "src/evaluate.py")

    ingest_and_clean >> build_features >> score_customers
