-- Phase 2, step 2: typed, standardized model table.
--
-- Transformations applied to raw_customers:
--   * TotalCharges  -> DOUBLE; the 11 blank/whitespace values belong to
--                      tenure = 0 customers who have not been billed yet, so
--                      they become 0.0 rather than NULL.
--   * tenure        -> INTEGER, MonthlyCharges -> DOUBLE.
--   * SeniorCitizen -> 'Yes'/'No' text, matching every other categorical
--                      instead of being the lone 0/1 column.
--   * Churn         -> churn_flag INTEGER (Yes=1, No=0); the original text
--                      column is kept alongside it.
--   * Add-on columns -> the placeholder levels "No internet service" and
--                      "No phone service" collapse to plain "No". They encode
--                      the same fact as InternetService/PhoneService and would
--                      otherwise create redundant dummy variables downstream.

CREATE OR REPLACE TABLE clean_customers AS
SELECT
    customerID,
    gender,
    CASE SeniorCitizen
        WHEN '1' THEN 'Yes'
        WHEN '0' THEN 'No'
    END                                                    AS SeniorCitizen,
    Partner,
    Dependents,
    CAST(tenure AS INTEGER)                                AS tenure,
    PhoneService,
    CASE WHEN MultipleLines    = 'No phone service'    THEN 'No' ELSE MultipleLines    END AS MultipleLines,
    InternetService,
    CASE WHEN OnlineSecurity   = 'No internet service' THEN 'No' ELSE OnlineSecurity   END AS OnlineSecurity,
    CASE WHEN OnlineBackup     = 'No internet service' THEN 'No' ELSE OnlineBackup     END AS OnlineBackup,
    CASE WHEN DeviceProtection = 'No internet service' THEN 'No' ELSE DeviceProtection END AS DeviceProtection,
    CASE WHEN TechSupport      = 'No internet service' THEN 'No' ELSE TechSupport      END AS TechSupport,
    CASE WHEN StreamingTV      = 'No internet service' THEN 'No' ELSE StreamingTV      END AS StreamingTV,
    CASE WHEN StreamingMovies  = 'No internet service' THEN 'No' ELSE StreamingMovies  END AS StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    CAST(MonthlyCharges AS DOUBLE)                         AS MonthlyCharges,
    CASE
        WHEN TRIM(TotalCharges) = '' THEN 0.0
        ELSE CAST(TotalCharges AS DOUBLE)
    END                                                    AS TotalCharges,
    Churn,
    CASE Churn
        WHEN 'Yes' THEN 1
        WHEN 'No'  THEN 0
    END                                                    AS churn_flag
FROM raw_customers;

-- Data quality gate. Every check counts offending rows, so a healthy load is
-- all zeroes; run_sql.py prints this table and exits non-zero if any fail.
CREATE OR REPLACE TABLE data_quality_checks AS
SELECT 'duplicate_customerID'      AS check_name, COUNT(*) - COUNT(DISTINCT customerID)     AS failures FROM clean_customers
UNION ALL
SELECT 'null_customerID',          COUNT(*) FILTER (WHERE customerID     IS NULL) FROM clean_customers
UNION ALL
SELECT 'null_tenure',              COUNT(*) FILTER (WHERE tenure         IS NULL) FROM clean_customers
UNION ALL
SELECT 'null_MonthlyCharges',      COUNT(*) FILTER (WHERE MonthlyCharges IS NULL) FROM clean_customers
UNION ALL
SELECT 'null_TotalCharges',        COUNT(*) FILTER (WHERE TotalCharges   IS NULL) FROM clean_customers
UNION ALL
SELECT 'null_churn_flag',          COUNT(*) FILTER (WHERE churn_flag     IS NULL) FROM clean_customers
UNION ALL
SELECT 'unmapped_SeniorCitizen',   COUNT(*) FILTER (WHERE SeniorCitizen  IS NULL) FROM clean_customers
UNION ALL
SELECT 'rows_lost_vs_raw',
       (SELECT COUNT(*) FROM raw_customers) - (SELECT COUNT(*) FROM clean_customers)
UNION ALL
SELECT 'residual_placeholder_levels',
       COUNT(*) FILTER (
           WHERE 'No internet service' IN (OnlineSecurity, OnlineBackup, DeviceProtection,
                                           TechSupport, StreamingTV, StreamingMovies)
              OR MultipleLines = 'No phone service'
       )
FROM clean_customers
ORDER BY check_name;
