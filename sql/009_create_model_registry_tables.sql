-- 009_create_model_registry_tables.sql
-- Stores model versions and quality history for comparison in the UI.

CREATE TABLE IF NOT EXISTS dwh.model_training_runs (
  run_id BIGSERIAL PRIMARY KEY,
  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  trained_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  device TEXT,
  model_path TEXT,
  epochs INT,
  lr DOUBLE PRECISION,
  weight_decay DOUBLE PRECISION,
  patience INT,
  seed INT,
  params_json JSONB,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS dwh.model_metrics_history (
  metric_id BIGSERIAL PRIMARY KEY,
  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  eval_type TEXT NOT NULL,
  horizon_years INT,
  n_points INT,
  mae DOUBLE PRECISION,
  rmse DOUBLE PRECISION,
  mape DOUBLE PRECISION,
  smape DOUBLE PRECISION,
  wape DOUBLE PRECISION,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_model_training_runs_name_ver
  ON dwh.model_training_runs(model_name, model_version, trained_at DESC);

CREATE INDEX IF NOT EXISTS ix_model_metrics_history_name_ver_eval
  ON dwh.model_metrics_history(model_name, model_version, eval_type, horizon_years);
