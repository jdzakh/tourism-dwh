-- 001_create_dwh.sql
CREATE SCHEMA IF NOT EXISTS dwh;
CREATE SCHEMA IF NOT EXISTS mart;

-- ===== DIMENSIONS =====

CREATE TABLE IF NOT EXISTS dwh.dim_date (
  date_id        INT PRIMARY KEY,          -- YYYYMMDD
  date_value     DATE NOT NULL,
  year           INT NOT NULL,
  month          INT NOT NULL,
  month_name     TEXT NOT NULL,
  quarter        INT NOT NULL,
  week_of_year   INT NOT NULL,
  is_weekend     BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS dwh.dim_destination (
  destination_id BIGSERIAL PRIMARY KEY,
  country        TEXT NOT NULL,
  country_code   TEXT,  
  region         TEXT,
  city           TEXT,
  UNIQUE (country, region, city)
);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'dwh' 
    AND table_name = 'dim_destination' 
    AND column_name = 'country_code'
  ) THEN
    ALTER TABLE dwh.dim_destination ADD COLUMN country_code TEXT;
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS dwh.dim_product (
  product_id     BIGSERIAL PRIMARY KEY,
  product_type   TEXT NOT NULL,             -- например: package, excursion, flight+hotel
  category       TEXT,                      -- leisure/business/etc
  duration_days  INT,
  UNIQUE (product_type, category, duration_days)
);

CREATE TABLE IF NOT EXISTS dwh.dim_channel (
  channel_id     BIGSERIAL PRIMARY KEY,
  channel_name   TEXT NOT NULL UNIQUE       -- online, office, partner, etc
);

-- ===== FACT =====

CREATE TABLE IF NOT EXISTS dwh.fact_flow (
  flow_id        BIGSERIAL PRIMARY KEY,
  date_id        INT NOT NULL REFERENCES dwh.dim_date(date_id),
  destination_id BIGINT NOT NULL REFERENCES dwh.dim_destination(destination_id),
  product_id     BIGINT REFERENCES dwh.dim_product(product_id),
  channel_id     BIGINT REFERENCES dwh.dim_channel(channel_id),

  tourists_cnt   BIGINT NOT NULL CHECK (tourists_cnt >= 0),
  bookings_cnt   BIGINT CHECK (bookings_cnt IS NULL OR bookings_cnt >= 0),
  revenue        NUMERIC(20,2) CHECK (revenue IS NULL OR revenue >= 0)
);

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'dwh' 
    AND table_name = 'fact_flow' 
    AND column_name = 'revenue'
    AND numeric_precision < 20
  ) THEN
    ALTER TABLE dwh.fact_flow ALTER COLUMN revenue TYPE NUMERIC(20,2);
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_fact_flow_date ON dwh.fact_flow(date_id);
CREATE INDEX IF NOT EXISTS ix_fact_flow_dest ON dwh.fact_flow(destination_id);
CREATE INDEX IF NOT EXISTS ix_fact_flow_product ON dwh.fact_flow(product_id);
CREATE INDEX IF NOT EXISTS ix_fact_flow_channel ON dwh.fact_flow(channel_id);
CREATE INDEX IF NOT EXISTS ix_fact_flow_dest_date ON dwh.fact_flow(destination_id, date_id);

-- ===== FORECAST TABLE (куда будем писать прогнозы) =====

CREATE TABLE IF NOT EXISTS dwh.fact_forecast_monthly (
  forecast_id      BIGSERIAL PRIMARY KEY,
  generated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

  year             INT NOT NULL,
  month            INT NOT NULL,
  destination_id   BIGINT REFERENCES dwh.dim_destination(destination_id), -- NULL = общий поток

  yhat             DOUBLE PRECISION NOT NULL,
  yhat_lower       DOUBLE PRECISION,
  yhat_upper       DOUBLE PRECISION,
  model_name       TEXT NOT NULL,
  horizon_months   INT NOT NULL,

  UNIQUE (generated_at, year, month, destination_id, model_name)
);

CREATE INDEX IF NOT EXISTS ix_forecast_month ON dwh.fact_forecast_monthly(year, month);
CREATE INDEX IF NOT EXISTS ix_forecast_dest ON dwh.fact_forecast_monthly(destination_id);
CREATE INDEX IF NOT EXISTS ix_forecast_model_horizon_generated
  ON dwh.fact_forecast_monthly(model_name, horizon_months, generated_at DESC);
CREATE INDEX IF NOT EXISTS ix_forecast_dest_model_horizon_year
  ON dwh.fact_forecast_monthly(destination_id, model_name, horizon_months, year);

-- ===== MARTS (витрины для Power BI) =====

-- 1) общий поток по месяцам
CREATE OR REPLACE VIEW mart.v_monthly_flow_total AS
SELECT
  dd.year,
  dd.month,
  (dd.year::text || '-' || lpad(dd.month::text, 2, '0')) AS year_month,
  SUM(ff.tourists_cnt) AS tourists_cnt,
  SUM(COALESCE(ff.bookings_cnt, 0)) AS bookings_cnt,
  SUM(COALESCE(ff.revenue, 0)) AS revenue
FROM dwh.fact_flow ff
JOIN dwh.dim_date dd ON dd.date_id = ff.date_id
GROUP BY dd.year, dd.month
ORDER BY dd.year, dd.month;

-- 2) поток по направлениям по месяцам
CREATE OR REPLACE VIEW mart.v_monthly_flow_by_destination AS
SELECT
  dd.year,
  dd.month,
  (dd.year::text || '-' || lpad(dd.month::text, 2, '0')) AS year_month,
  dest.destination_id,
  dest.country,
  dest.region,
  dest.city,
  SUM(ff.tourists_cnt) AS tourists_cnt,
  SUM(COALESCE(ff.bookings_cnt, 0)) AS bookings_cnt,
  SUM(COALESCE(ff.revenue, 0)) AS revenue
FROM dwh.fact_flow ff
JOIN dwh.dim_date dd ON dd.date_id = ff.date_id
JOIN dwh.dim_destination dest ON dest.destination_id = ff.destination_id
GROUP BY dd.year, dd.month, dest.destination_id, dest.country, dest.region, dest.city
ORDER BY dd.year, dd.month, tourists_cnt DESC;

-- 3) сезонность (удобно для анализа)
CREATE OR REPLACE VIEW mart.v_seasonality AS
SELECT
  dd.month,
  dd.month_name,
  SUM(ff.tourists_cnt) AS tourists_cnt
FROM dwh.fact_flow ff
JOIN dwh.dim_date dd ON dd.date_id = ff.date_id
GROUP BY dd.month, dd.month_name
ORDER BY dd.month;
