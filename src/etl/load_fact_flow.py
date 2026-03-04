import os

import pandas as pd
from sqlalchemy import text
from src.utils.db import get_engine
from src.utils.logging_utils import configure_logging, get_logger

WB_CSV_PATH = "data/processed/world_bank_tourism.csv"
LEGACY_CSV_PATH = "data/raw/world_tourism_economy_data.csv"
logger = get_logger("etl.load_fact_flow")

def main():
    configure_logging()
    csv_path = WB_CSV_PATH if os.path.exists(WB_CSV_PATH) else LEGACY_CSV_PATH
    df = pd.read_csv(csv_path)

    # Берём только строки, где есть год, страна и arrivals (турпоток)
    df = df.dropna(subset=["country", "year", "tourism_arrivals"])

    # date_id = YYYY0101 (так как данные годовые)
    df["date_id"] = (df["year"].astype(int).astype(str) + "0101").astype(int)

    engine = get_engine()

    # Достанем mapping: country -> destination_id
    dest_map = pd.read_sql(
        "SELECT destination_id, country FROM dwh.dim_destination",
        engine
    )
    df = df.merge(dest_map, on="country", how="left")

    # Если вдруг есть страны, которых нет в dim_destination
    missing = df[df["destination_id"].isna()]["country"].unique().tolist()
    if missing:
        raise ValueError(f"Missing destination_id for countries: {missing[:10]} ... total={len(missing)}")

    tourists_cnt = pd.to_numeric(df["tourism_arrivals"], errors="coerce").round()
    revenue = pd.to_numeric(df.get("tourism_receipts"), errors="coerce")

    fact = pd.DataFrame({
        "date_id": df["date_id"].astype(int),
        "destination_id": df["destination_id"].astype(int),
        "product_id": None,
        "channel_id": None,
        "tourists_cnt": tourists_cnt.astype("Int64"),
        "bookings_cnt": None,
        "revenue": revenue,
    })

    # Чистим NaN в revenue
    fact["revenue"] = fact["revenue"].where(pd.notna(fact["revenue"]), None)

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE dwh.fact_flow;"))

    fact.to_sql(
        "fact_flow",
        engine,
        schema="dwh",
        if_exists="append",
        index=False,
        chunksize=5000
    )

    logger.info("Inserted fact_flow rows: %s", len(fact))

if __name__ == "__main__":
    main()
