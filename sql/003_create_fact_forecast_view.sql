CREATE OR REPLACE VIEW mart.v_yearly_fact_forecast_total AS
WITH fact AS (
  SELECT
    dd.year,
    SUM(ff.tourists_cnt) AS tourists_cnt_fact
  FROM dwh.fact_flow ff
  JOIN dwh.dim_date dd ON dd.date_id = ff.date_id
  GROUP BY dd.year
),
fc AS (
  SELECT
    year,
    yhat,
    yhat_lower,
    yhat_upper,
    model_name
  FROM dwh.fact_forecast_monthly
  WHERE destination_id IS NULL
)
SELECT
  COALESCE(fact.year, fc.year) AS year,
  fact.tourists_cnt_fact,
  fc.yhat AS tourists_cnt_forecast,
  fc.yhat_lower,
  fc.yhat_upper,
  fc.model_name
FROM fact
FULL JOIN fc ON fc.year = fact.year
ORDER BY year;
