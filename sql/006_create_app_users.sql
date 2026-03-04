CREATE SCHEMA IF NOT EXISTS app;

CREATE TABLE IF NOT EXISTS app.users (
  user_id       BIGSERIAL PRIMARY KEY,
  name          TEXT,
  email         TEXT NOT NULL UNIQUE,
  role          TEXT NOT NULL,
  password_hash TEXT NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
