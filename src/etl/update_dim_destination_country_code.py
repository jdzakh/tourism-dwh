import os

import pandas as pd
from sqlalchemy import text
from src.utils.db import get_engine

WB_CSV_PATH = "data/processed/world_bank_tourism.csv"
LEGACY_CSV_PATH = "data/raw/world_tourism_economy_data.csv"

def main():
    csv_path = WB_CSV_PATH if os.path.exists(WB_CSV_PATH) else LEGACY_CSV_PATH
    df = pd.read_csv(csv_path, usecols=["country", "country_code"])
    df = df.dropna().drop_duplicates()

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("UPDATE dwh.dim_destination SET country_code = NULL;"))
        # обновляем по названию страны
        for _, r in df.iterrows():
            conn.execute(
                text("UPDATE dwh.dim_destination SET country_code=:cc WHERE country=:c"),
                {"cc": r["country_code"], "c": r["country"]}
            )

    print("Updated country_code in dim_destination")

if __name__ == "__main__":
    main()
