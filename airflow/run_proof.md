# DAG run proof — `churn_batch_pipeline`

Local end-to-end run on Apache Airflow 2.10.5 (Python 3.11, macOS).
Triggered manually with `airflow dags trigger`; scheduler and webserver
started by `airflow standalone`.

![Successful run](dag_success.png)

```
$ airflow dags list-runs -d churn_batch_pipeline
dag_id               | run_id                       | state   | execution_date            | start_date                       | end_date                        
=====================+==============================+=========+===========================+==================================+=================================
churn_batch_pipeline | manual_proof_20260902T155354 | success | 2026-09-02T19:53:55+00:00 | 2026-09-02T19:53:56.470645+00:00 | 2026-09-02T19:54:10.819416+00:00
                                                                                                                                                               
```

```
$ airflow tasks states-for-dag-run churn_batch_pipeline manual_proof_20260902T155354
dag_id               | execution_date            | task_id          | state   | start_date                       | end_date                        
=====================+===========================+==================+=========+==================================+=================================
churn_batch_pipeline | 2026-09-02T19:53:55+00:00 | ingest_and_clean | success | 2026-09-02T19:53:57.877095+00:00 | 2026-09-02T19:53:58.592984+00:00
churn_batch_pipeline | 2026-09-02T19:53:55+00:00 | build_features   | success | 2026-09-02T19:54:00.480397+00:00 | 2026-09-02T19:54:01.659395+00:00
churn_batch_pipeline | 2026-09-02T19:53:55+00:00 | score_customers  | success | 2026-09-02T19:54:03.482176+00:00 | 2026-09-02T19:54:10.436669+00:00
                                                                                                                                                   
```

All three tasks reached `success`; the run completed in ~14 seconds.

## Artifacts written by the run

The tasks really executed the project scripts rather than merely reporting
success — both outputs were rewritten inside the task windows above, and
the task logs carry the scripts' own output:

```
data/churn.duckdb                     rewritten 15:53:58  (ingest_and_clean)
data/processed/scored_customers.csv   rewritten 15:54:09  (score_customers)
```

```
ingest_and_clean  INFO - All checks passed.
build_features    INFO - X shape         (7043, 28)
score_customers   INFO - recommended: threshold 0.24 — highest recall with precision >= 0.50
score_customers   INFO -   caught 303 of 374 churners (81.0%)
score_customers   INFO -   saved  data/processed/scored_customers.csv (204.3 KB)
```

---

## Failure path — intentional test with a corrupted copy

![Failure test](dag_failure.png)

> **This is a deliberate negative test, not a real incident.** It was run against
> a throwaway **copy** of the repository in a scratch directory, whose CSV was
> corrupted on purpose. The real `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
> was never modified — its SHA-256 was recorded before and after the test and is
> unchanged:
>
> ```
> 88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a
> ```

The success path above proves the DAG runs. This section proves the other half of
the contract: that a failed data quality check **stops the pipeline** instead of
letting bad data reach the model.

### The corruption

Two defects were introduced into the sandbox copy of the CSV, both chosen to
survive the `CAST` steps so the failure lands on the quality gate itself rather
than on a SQL type error:

| Defect | Check it trips |
|---|---|
| An existing customer row duplicated verbatim (`7590-VHVEG`) | `duplicate_customerID` |
| A second row duplicated with `SeniorCitizen` set to `'2'` (`5575-GNVDE`) | `duplicate_customerID` + `unmapped_SeniorCitizen` |

### Task states

```
$ airflow dags list-runs -d churn_batch_pipeline
dag_id               | run_id                          | state   | execution_date            | start_date                       | end_date                        
=====================+=================================+=========+===========================+==================================+=================================
churn_batch_pipeline | manual_failtest_20260902T160523 | failed  | 2026-09-02T20:05:25+00:00 | 2026-09-02T20:05:26.534306+00:00 | 2026-09-02T20:05:30.330899+00:00
```

```
$ airflow tasks states-for-dag-run churn_batch_pipeline manual_failtest_20260902T160523
dag_id               | execution_date            | task_id          | state           | start_date                       | end_date                        
=====================+===========================+==================+=================+==================================+=================================
churn_batch_pipeline | 2026-09-02T20:05:25+00:00 | ingest_and_clean | failed          | 2026-09-02T20:05:28.153134+00:00 | 2026-09-02T20:05:28.865493+00:00
churn_batch_pipeline | 2026-09-02T20:05:25+00:00 | build_features   | upstream_failed | 2026-09-02T20:05:28.934124+00:00 | 2026-09-02T20:05:28.934124+00:00
churn_batch_pipeline | 2026-09-02T20:05:25+00:00 | score_customers  | upstream_failed | 2026-09-02T20:05:29.292325+00:00 | 2026-09-02T20:05:29.292325+00:00
                                                                                                                                                           
```

`ingest_and_clean` **failed**; `build_features` and `score_customers` were both
marked **`upstream_failed`** and never executed. The DAG run ended in ~4 seconds.

### The gate firing

From the `ingest_and_clean` task log:

```
  [FAIL] duplicate_customerID                2
  [PASS] null_MonthlyCharges                 0
  [PASS] null_TotalCharges                   0
  [PASS] null_churn_flag                     0
  [PASS] null_customerID                     0
  [PASS] null_tenure                         0
  [PASS] residual_placeholder_levels         0
  [PASS] rows_lost_vs_raw                    0
  [FAIL] unmapped_SeniorCitizen              1
3 data quality failure(s) — see checks above.
Task exited with return code 1
```

`run_sql.py` counted 3 offending rows across two checks and exited non-zero;
`BashOperator` turned that exit code into a task failure; the dependency chain
did the rest.

### What was protected

Because the downstream tasks never ran, the previous good outputs were left in
place rather than being overwritten with scores derived from corrupt input —
which is the behaviour you want when the alternative is silently publishing
wrong numbers. Verified on the real checkout after the test:

```
clean_customers rows:  7,043   (not the corrupted 7,045)
total check failures:  0
data/churn.duckdb                     last written 15:53:58  (the successful run)
data/processed/scored_customers.csv   last written 15:54:09  (the successful run)
```
