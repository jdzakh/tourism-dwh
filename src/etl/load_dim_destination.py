import os

import pandas as pd
from sqlalchemy import inspect, text
from src.utils.db import get_engine
from src.utils.logging_utils import configure_logging, get_logger

WB_CSV_PATH = "data/processed/world_bank_tourism.csv"
LEGACY_CSV_PATH = "data/raw/world_tourism_economy_data.csv"
logger = get_logger("etl.load_dim_destination")

def main():
    configure_logging()
    csv_path = WB_CSV_PATH if os.path.exists(WB_CSV_PATH) else LEGACY_CSV_PATH
    df = pd.read_csv(csv_path)

    cols = ["country"]
    if "country_code" in df.columns:
        cols.append("country_code")

    dest = df[cols].dropna(subset=["country"]).drop_duplicates()
    dest["region"] = None
    dest["city"] = None

    engine = get_engine()
    db_cols = [
        col["name"]
        for col in inspect(engine).get_columns("dim_destination", schema="dwh")
    ]
    dest = dest[[c for c in ["country", "region", "city", "country_code"] if c in db_cols]]

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE dwh.dim_destination CASCADE;"))

    dest.to_sql(
        "dim_destination",
        engine,
        schema="dwh",
        if_exists="append",
        index=False
    )

    logger.info("Inserted destinations: %s", len(dest))

if __name__ == "__main__":
    main()
