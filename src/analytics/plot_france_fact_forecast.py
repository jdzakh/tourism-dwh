import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.db import get_engine

engine = get_engine()

df = pd.read_sql("""
SELECT year,
       tourists_cnt_fact,
       tourists_cnt_forecast
FROM mart.v_yearly_fact_forecast_by_country
WHERE country = 'France'
ORDER BY year
""", engine)

plt.figure(figsize=(10,5))
plt.plot(df["year"], df["tourists_cnt_fact"], label="Факт")
plt.plot(df["year"], df["tourists_cnt_forecast"], label="Прогноз", linestyle="--")

plt.title("Туристический поток: Франция")
plt.xlabel("Год")
plt.ylabel("Число туристов")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("reports/france_fact_forecast.png", dpi=150)
plt.show()
