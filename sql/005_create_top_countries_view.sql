CREATE OR REPLACE VIEW mart.v_top_countries AS
SELECT
  dest.destination_id,
  dest.country,
  dest.country_code,
  SUM(ff.tourists_cnt) AS tourists_cnt_total,
  SUM(COALESCE(ff.revenue,0)) AS revenue_total
FROM dwh.fact_flow ff
JOIN dwh.dim_destination dest ON dest.destination_id = ff.destination_id
WHERE dest.country_code IS NOT NULL
  AND length(dest.country_code) = 3
  -- выкидываем распространённые агрегаты WB (примеры: EMU, NAC, OED, WLD и т.п.)
  AND dest.country_code NOT IN ('WLD','EMU','NAC','OED','HIC','LIC','MIC','LMC','UMC','ECS','EAS','AFE','AFW','ARB','CSS','CEA','CEB','EAP','ECA','ECS','EMU','EUU','FCS','HPC','IBD','IBT','IDA','IDB','IDX','INX','LAC','LCN','LDC','MEA','MNA','NAC','OED','OSS','PRE','PSS','PST','SAS','SSA','SSF','SST','TEA','TEC','TLA','TMN','TSA','TSS','UMC','WLD')
  AND dest.country !~* '(income|dividend|OECD|Europe|Asia|Africa|Middle East|Latin|Caribbean|Arab|Union|states|IDA|IBRD|Fragile|Small|emerging|developing|least|total|area)'
GROUP BY dest.destination_id, dest.country, dest.country_code
ORDER BY tourists_cnt_total DESC;
