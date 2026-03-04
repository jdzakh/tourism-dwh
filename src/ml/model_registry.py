from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
from sqlalchemy import text


def ensure_registry_tables(engine) -> None:
    ddl = """
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
      eval_type TEXT NOT NULL, -- in_sample_total | horizon_backtest_total
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
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def register_training_run(
    engine,
    model_name: str,
    model_version: str,
    device: str,
    model_path: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
    notes: Optional[str] = None,
    params: Optional[Dict] = None,
) -> None:
    ensure_registry_tables(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO dwh.model_training_runs
                (model_name, model_version, trained_at, device, model_path, epochs, lr, weight_decay, patience, seed, params_json, notes)
                VALUES
                (:model_name, :model_version, :trained_at, :device, :model_path, :epochs, :lr, :weight_decay, :patience, :seed, CAST(:params_json AS jsonb), :notes)
                """
            ),
            {
                "model_name": model_name,
                "model_version": model_version,
                "trained_at": datetime.utcnow(),
                "device": device,
                "model_path": model_path,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "patience": patience,
                "seed": seed,
                "params_json": json.dumps(params or {}),
                "notes": notes,
            },
        )


def save_metric_rows(
    engine,
    model_name: str,
    model_version: str,
    eval_type: str,
    metrics_df: pd.DataFrame,
) -> None:
    if metrics_df.empty:
        return
    ensure_registry_tables(engine)
    to_insert = metrics_df.copy()
    to_insert["model_name"] = model_name
    to_insert["model_version"] = model_version
    to_insert["eval_type"] = eval_type
    to_insert["created_at"] = datetime.utcnow()
    cols = [
        "model_name",
        "model_version",
        "eval_type",
        "horizon_years",
        "n_points",
        "mae",
        "rmse",
        "mape",
        "smape",
        "wape",
        "created_at",
    ]
    to_insert = to_insert[cols]
    to_insert.to_sql("model_metrics_history", engine, schema="dwh", if_exists="append", index=False)
