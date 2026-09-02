-- Phase 2, step 1: raw ingestion.
-- Every column lands as VARCHAR so the file is preserved exactly as delivered:
-- TotalCharges contains blank (" ") values that would otherwise force DuckDB's
-- sniffer into a bad guess, and SeniorCitizen's 0/1 is really a categorical.
-- All typing decisions happen downstream in 02_clean.sql.

CREATE OR REPLACE TABLE raw_customers AS
SELECT *
FROM read_csv(
    'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv',
    header = true,
    all_varchar = true
);
