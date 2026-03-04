CREATE OR REPLACE VIEW mart.v_yearly_fact_forecast_by_country AS
WITH fact AS (
  SELECT
    ff.destination_id,
    d.country,
    d.country_code,
    dd.year,
    SUM(ff.tourists_cnt) AS tourists_cnt_fact
  FROM dwh.fact_flow ff
  JOIN dwh.dim_date dd ON dd.date_id = ff.date_id
  JOIN dwh.dim_destination d ON d.destination_id = ff.destination_id
  GROUP BY ff.destination_id, d.country, d.country_code, dd.year
),
fc AS (
  SELECT
    destination_id,
    year,
    yhat,
    yhat_lower,
    yhat_upper,
    model_name
  FROM dwh.fact_forecast_monthly
  WHERE destination_id IS NOT NULL
)
SELECT
  COALESCE(fact.destination_id, fc.destination_id) AS destination_id,
  COALESCE(fact.country, d.country) AS country,
  COALESCE(fact.country_code, d.country_code) AS country_code,
  COALESCE(fact.year, fc.year) AS year,
  fact.tourists_cnt_fact,
  fc.yhat AS tourists_cnt_forecast,
  fc.yhat_lower,
  fc.yhat_upper,
  fc.model_name
FROM fact
FULL JOIN fc
  ON fc.destination_id = fact.destination_id
 AND fc.year = fact.year
LEFT JOIN dwh.dim_destination d
  ON d.destination_id = COALESCE(fact.destination_id, fc.destination_id)
ORDER BY destination_id, year;
