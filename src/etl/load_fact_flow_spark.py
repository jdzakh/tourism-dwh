import os
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, concat, lpad, round as sround
from pyspark.sql.types import IntegerType, LongType, DecimalType
from sqlalchemy import text
from src.utils.db import get_engine

load_dotenv()

WB_CSV_PATH = "data/processed/world_bank_tourism.csv"
LEGACY_CSV_PATH = "data/raw/world_tourism_economy_data.csv"
POSTGRES_DRIVER_VERSION = "42.7.3"  # можно оставить так

def build_spark() -> SparkSession:
    # Драйвер Postgres подтянется автоматически (нужен интернет)
    return (
        SparkSession.builder
        .appName("tourism-dwh-etl")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.jars.packages", f"org.postgresql:postgresql:{POSTGRES_DRIVER_VERSION}")
        .getOrCreate()
    )

def main():
    pg_host = os.getenv("PG_HOST", "localhost")
    pg_port = os.getenv("PG_PORT", "5433")
    pg_db = os.getenv("PG_DB", "tourism_dwh")
    pg_user = os.getenv("PG_USER", "tourism")
    pg_pwd = os.getenv("PG_PASSWORD", "tourism")

    jdbc_url = f"jdbc:postgresql://{pg_host}:{pg_port}/{pg_db}"
    jdbc_props = {
        "user": pg_user,
        "password": pg_pwd,
        "driver": "org.postgresql.Driver",
    }

    spark = build_spark()

    csv_path = WB_CSV_PATH if os.path.exists(WB_CSV_PATH) else LEGACY_CSV_PATH

    # 1) Читаем CSV
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(csv_path)
    )

    # 2) Оставляем только нужное
    df = df.select(
        col("country"),
        col("year"),
        col("tourism_arrivals"),
        col("tourism_receipts"),
    )

    # 3) Фильтруем пустое (нам нужен поток)
    df = df.filter(col("country").isNotNull() & col("year").isNotNull() & col("tourism_arrivals").isNotNull())

    # 4) date_id = YYYY0101 (годовые данные)
    df = df.withColumn("year_int", col("year").cast(IntegerType()))
    df = df.withColumn(
        "date_id",
        concat(col("year_int").cast("string"), lit("0101")).cast(IntegerType())
    )

    # 5) Подтягиваем dim_destination из Postgres
    dim_dest = (
        spark.read
        .jdbc(url=jdbc_url, table="dwh.dim_destination", properties=jdbc_props)
        .select(col("destination_id").cast(LongType()), col("country").alias("country_dim"))
    )

    # 6) Join по country
    df = df.join(dim_dest, df["country"] == dim_dest["country_dim"], how="left")

    # 7) Проверка: нет ли стран без destination_id
    missing = df.filter(col("destination_id").isNull()).select("country").distinct().limit(20).collect()
    if missing:
        spark.stop()
        raise RuntimeError(f"Missing destination_id for countries (sample): {[r['country'] for r in missing]}")

    # 8) Формируем fact_flow под схему
    fact = (
        df.select(
            col("date_id").cast(IntegerType()),
            col("destination_id").cast(LongType()),
            lit(None).cast(LongType()).alias("product_id"),
            lit(None).cast(LongType()).alias("channel_id"),
            sround(col("tourism_arrivals")).cast(LongType()).alias("tourists_cnt"),
            lit(None).cast(LongType()).alias("bookings_cnt"),
            col("tourism_receipts").cast(DecimalType(20, 2)).alias("revenue"),
        )
    )

    # 9) TRUNCATE fact_flow перед загрузкой (через SQLAlchemy)
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE dwh.fact_flow;"))

    # 10) Пишем в Postgres
    (
        fact.write
        .mode("append")
        .jdbc(url=jdbc_url, table="dwh.fact_flow", properties=jdbc_props)
    )

    # маленькая проверка
    print("Loaded rows into dwh.fact_flow (spark):", fact.count())

    spark.stop()

if __name__ == "__main__":
    main()
