from datetime import date, timedelta
import pandas as pd
from sqlalchemy import text
from src.utils.db import get_engine
from src.utils.logging_utils import configure_logging, get_logger

logger = get_logger("etl.seed_dim_date")

def build_dim_date(start: date, end: date) -> pd.DataFrame:
    rows = []
    d = start
    while d <= end:
        rows.append({
            "date_id": int(d.strftime("%Y%m%d")),
            "date_value": d,
            "year": d.year,
            "month": d.month,
            "month_name": d.strftime("%B"),
            "quarter": (d.month - 1) // 3 + 1,
            "week_of_year": int(d.strftime("%V")),
            "is_weekend": d.weekday() >= 5,
        })
        d += timedelta(days=1)
    return pd.DataFrame(rows)

def main():
    configure_logging()
    # расширяем диапазон под датасет (есть 1999)
    df = build_dim_date(date(1990, 1, 1), date(2030, 12, 31))
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE dwh.dim_date CASCADE;"))
    df.to_sql("dim_date", engine, schema="dwh", if_exists="append", index=False, chunksize=10000)
    logger.info("Inserted dim_date rows: %s", len(df))

if __name__ == "__main__":
    main()
