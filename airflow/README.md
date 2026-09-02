# Airflow orchestration

`dags/churn_pipeline_dag.py` defines **`churn_batch_pipeline`**, a three-task DAG
that rebuilds the analytical tables from the raw CSV and scores every customer
with the committed model.

```
ingest_and_clean  ->  build_features  ->  score_customers
   src/run_sql.py      src/features.py     src/evaluate.py
```

It is `schedule=None` (manual trigger only) with `catchup=False`, so it never
backfills.

## Running it locally

### 1. Install Airflow in its own virtualenv

Airflow pins its dependency tree tightly and will fight with the project's
pandas / scikit-learn / xgboost versions if installed alongside them. Keep it
separate — the DAG is written to shell out to the project interpreter anyway
(see step 3).

```bash
python3 -m venv ~/.venvs/airflow
AIRFLOW_VERSION=2.10.5
PYTHON_VERSION=3.11
~/.venvs/airflow/bin/pip install "apache-airflow==${AIRFLOW_VERSION}" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
```

**The `--constraint` flag is not optional.** Installing bare `apache-airflow`
resolves the newest `cryptography` and `libcst`, which on x86_64 macOS ship as
source-only and need a Rust toolchain to build; the install fails with
`failed-wheel-build-for-install`. The constraints file pins versions that have
prebuilt wheels. Verified here with Airflow 2.10.5 on Python 3.11.

### 2. Point `AIRFLOW_HOME` somewhere and register the DAG

Airflow only loads DAGs from `$AIRFLOW_HOME/dags`, so symlink this one in rather
than copying it — a symlink keeps the checkout as the single source of truth, and
the DAG resolves its own path back through the link to find the project root.

```bash
export AIRFLOW_HOME=~/airflow
mkdir -p "$AIRFLOW_HOME/dags"
ln -sf "$(pwd)/airflow/dags/churn_pipeline_dag.py" "$AIRFLOW_HOME/dags/"
```

### 3. Tell the DAG which interpreter runs the tasks

The tasks need `duckdb`, `pandas`, `scikit-learn` and `joblib`; the Airflow venv
has none of them. `CHURN_PYTHON_BIN` names the interpreter that does. It defaults
to whichever interpreter parses the DAG, which is right only if one environment
holds both.

```bash
export CHURN_PYTHON_BIN=$(which python3)
```

### 4. Start Airflow and trigger the run

```bash
export AIRFLOW__CORE__LOAD_EXAMPLES=False    # keep the UI to just this DAG
~/.venvs/airflow/bin/airflow standalone
```

`standalone` initialises the metadata database, creates an `admin` user (the
password is printed to the console and written to
`$AIRFLOW_HOME/simple_auth_manager_passwords.json.generated`), and runs the
scheduler plus webserver on <http://localhost:8080>.

In a second shell:

```bash
export AIRFLOW_HOME=~/airflow
~/.venvs/airflow/bin/airflow dags unpause churn_batch_pipeline
~/.venvs/airflow/bin/airflow dags trigger churn_batch_pipeline

# watch the task states
~/.venvs/airflow/bin/airflow dags list-runs -d churn_batch_pipeline
~/.venvs/airflow/bin/airflow tasks states-for-dag-run churn_batch_pipeline <run_id>
```

To execute the whole DAG synchronously without a scheduler or webserver — the
quickest way to check a change — use:

```bash
~/.venvs/airflow/bin/airflow dags test churn_batch_pipeline
```

### Gotcha: this directory shadows the `airflow` package

Running `python3 -c "import airflow"` from the repo root imports **this
directory**, not the installed library, because the working directory precedes
site-packages on `sys.path`. The symptom is a confusing
`module 'airflow' has no attribute '__version__'` from an `import` that appeared
to succeed. Run `airflow` commands from anywhere else, or invoke the venv's
`airflow` binary by full path as above.

## Design

### Batch scoring, not scheduled retraining

The DAG applies a model; it does not build one. `src/train.py` is deliberately
absent from the graph.

Training is a decision, not a chore. It changes which model is in production, and
it wants a human to compare the candidates, read the metrics, and judge whether
the new model is genuinely better than the one it replaces. Retraining on a timer
removes that judgement: the model silently changes underneath the business, a bad
week of upstream data quietly becomes a bad model, and the first sign of trouble
is a drifting business metric weeks later. Worse, nothing in the system records
*which* model produced *which* scores.

So the model is a **versioned artifact**. `models/churn_model.joblib` is trained
by hand, reviewed against `models/model_comparison.md`, and committed — which
means promoting a new model is a commit, visible in review and revertible like
any other change. This DAG then does the repeatable half: new data arrives, gets
cleaned, and gets scored by a known model whose evaluation numbers are on record.
Retraining happens out-of-band, on a human's initiative — typically when
monitoring shows the scores drifting, or when enough new labelled churn has
accumulated to be worth the retrain.

### The data quality gate

`src/run_sql.py` builds a `data_quality_checks` table alongside
`clean_customers`: duplicate `customerID`s, nulls in the key columns, categorical
levels that failed to map, leftover `"No internet service"` placeholders, and a
row count reconciled against `raw_customers`. Each check counts offending rows,
so a healthy load is all zeroes, and the script exits **non-zero** if any check
reports failures.

That exit code is the whole mechanism. `ingest_and_clean` is a `BashOperator`, so
a non-zero exit is a failed task, and because the tasks are chained
`ingest_and_clean >> build_features >> score_customers`, the downstream tasks are
never scheduled. Bad data therefore cannot reach the model or overwrite
`scored_customers.csv` — the previous good export stays in place, which is the
behaviour you want when the alternative is silently publishing wrong scores.

`retries=0` is set deliberately. A failed quality check is a defect in the data,
not a transient fault; re-running the same deterministic SQL over the same file
produces the same failure, and retrying only makes a real problem look flaky in
the UI. Raise it above zero only for tasks whose failures are genuinely
transient, such as a network fetch.
