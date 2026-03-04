-- 008_create_performance_indexes.sql
-- Safe to run multiple times.

CREATE INDEX IF NOT EXISTS ix_dim_destination_country
  ON dwh.dim_destination(country);

CREATE INDEX IF NOT EXISTS ix_dim_date_year
  ON dwh.dim_date(year);

CREATE INDEX IF NOT EXISTS ix_fact_flow_dest_date
  ON dwh.fact_flow(destination_id, date_id);

CREATE INDEX IF NOT EXISTS ix_forecast_model_horizon_generated
  ON dwh.fact_forecast_monthly(model_name, horizon_months, generated_at DESC);

CREATE INDEX IF NOT EXISTS ix_forecast_dest_model_horizon_year
  ON dwh.fact_forecast_monthly(destination_id, model_name, horizon_months, year);
