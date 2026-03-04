#!/usr/bin/env sh
set -e

cat sql/002_create_yearly_marts.sql \
    sql/003_create_fact_forecast_view.sql \
    sql/004_create_country_fact_forecast_view.sql \
    sql/005_create_top_countries_view.sql \
    sql/008_create_performance_indexes.sql \
  | docker exec -i tourism_postgres psql -U tourism -d tourism_dwh
