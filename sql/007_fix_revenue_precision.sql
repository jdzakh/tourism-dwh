-- Миграция: увеличение точности поля revenue в таблице fact_flow
-- NUMERIC(14,2) -> NUMERIC(20,2) для поддержки больших значений выручки

ALTER TABLE dwh.fact_flow ALTER COLUMN revenue TYPE NUMERIC(20,2);

