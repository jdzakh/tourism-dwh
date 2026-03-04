CREATE SCHEMA IF NOT EXISTS mart;

-- Общий турпоток по годам
CREATE OR REPLACE VIEW mart.v_yearly_flow_total AS
SELECT
  dd.year,
  SUM(ff.tourists_cnt) AS tourists_cnt,
  SUM(COALESCE(ff.revenue, 0)) AS revenue
FROM dwh.fact_flow ff
JOIN dwh.dim_date dd ON dd.date_id = ff.date_id
GROUP BY dd.year
ORDER BY dd.year;

-- По направлениям по годам
CREATE OR REPLACE VIEW mart.v_yearly_flow_by_destination AS
SELECT
  dd.year,
  dest.destination_id,
  dest.country,
  SUM(ff.tourists_cnt) AS tourists_cnt,
  SUM(COALESCE(ff.revenue, 0)) AS revenue
FROM dwh.fact_flow ff
JOIN dwh.dim_date dd ON dd.date_id = ff.date_id
JOIN dwh.dim_destination dest ON dest.destination_id = ff.destination_id
GROUP BY dd.year, dest.destination_id, dest.country
ORDER BY dd.year, tourists_cnt DESC;

-- ТОП направлений по суммарному потоку (весь период)
CREATE OR REPLACE VIEW mart.v_top_destinations AS
SELECT
  dest.destination_id,
  dest.country,
  SUM(ff.tourists_cnt) AS tourists_cnt_total,
  SUM(COALESCE(ff.revenue,0)) AS revenue_total
FROM dwh.fact_flow ff
JOIN dwh.dim_destination dest ON dest.destination_id = ff.destination_id
GROUP BY dest.destination_id, dest.country
ORDER BY tourists_cnt_total DESC;
