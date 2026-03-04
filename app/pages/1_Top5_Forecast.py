import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
px.defaults.template = "plotly_dark"


def get_engine():
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    db = os.getenv("PG_DB", "tourism_dwh")
    user = os.getenv("PG_USER", "tourism")
    pwd = os.getenv("PG_PASSWORD", "tourism")
    if host == "localhost":
        port = os.getenv("PG_PORT", "5433")
    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10})


@st.cache_data(ttl=300)
def list_models(_engine):
    return pd.read_sql(
        """
        SELECT DISTINCT model_name
        FROM dwh.fact_forecast_monthly
        WHERE destination_id IS NOT NULL
        ORDER BY model_name
        """,
        _engine,
    )["model_name"].tolist()


@st.cache_data(ttl=300)
def latest_generated_at(_engine, model_name, horizon_years):
    return pd.read_sql(
        text(
            """
            SELECT MAX(generated_at) AS generated_at
            FROM dwh.fact_forecast_monthly
            WHERE destination_id IS NOT NULL
              AND model_name = :m
              AND horizon_months = :h
            """
        ),
        _engine,
        params={"m": model_name, "h": horizon_years * 12},
    ).iloc[0]["generated_at"]


@st.cache_data(ttl=300)
def load_top5_fact_forecast(_engine, model_name, generated_at, horizon_years):
    q = text(
        """
        WITH top AS (
          SELECT destination_id, country, country_code
          FROM mart.v_top_countries
          LIMIT 5
        ),
        fact AS (
          SELECT ff.destination_id, dd.year, SUM(ff.tourists_cnt) AS tourists_cnt_fact
          FROM dwh.fact_flow ff
          JOIN dwh.dim_date dd ON dd.date_id = ff.date_id
          JOIN top t ON t.destination_id = ff.destination_id
          GROUP BY ff.destination_id, dd.year
        ),
        fc AS (
          SELECT f.destination_id, f.year, f.yhat, f.yhat_lower, f.yhat_upper
          FROM dwh.fact_forecast_monthly f
          JOIN top t ON t.destination_id = f.destination_id
          WHERE f.model_name = :m
            AND f.generated_at = :g
            AND f.horizon_months = :h
        ),
        combined AS (
          SELECT
            COALESCE(fact.destination_id, fc.destination_id) AS destination_id,
            COALESCE(fact.year, fc.year) AS year,
            fact.tourists_cnt_fact,
            fc.yhat,
            fc.yhat_lower,
            fc.yhat_upper
          FROM fact
          FULL JOIN fc ON fc.destination_id = fact.destination_id AND fc.year = fact.year
        )
        SELECT c.year, c.tourists_cnt_fact, c.yhat, c.yhat_lower, c.yhat_upper, t.country
        FROM combined c
        JOIN top t ON t.destination_id = c.destination_id
        ORDER BY t.country, c.year
        """
    )
    return pd.read_sql(q, _engine, params={"m": model_name, "g": generated_at, "h": horizon_years * 12})


def compute_error_metrics(df):
    paired = df[["tourists_cnt_fact", "yhat"]].dropna()
    if paired.empty:
        return None, None, None, 0
    err = paired["tourists_cnt_fact"] - paired["yhat"]
    mae = float(err.abs().mean())
    rmse = float((err.pow(2).mean()) ** 0.5)
    wape_den = float(paired["tourists_cnt_fact"].abs().sum())
    wape = float(err.abs().sum() / wape_den * 100) if wape_den > 0 else None
    return mae, rmse, wape, int(len(paired))


def main():
    st.set_page_config(page_title="Прогноз TOP-5", layout="wide")
    st.title("Прогноз по ключевым направлениям")
    st.caption("TOP-5 стран: сравнение факта и прогноза, а также метрики качества модели.")

    engine = get_engine()
    horizon_years = st.slider("Горизонт прогноза (лет)", 5, 10, 5)
    model_name = "NN_MLP_GLOBAL"

    if model_name not in list_models(engine):
        st.warning("Прогнозы нейросети отсутствуют. Сначала запустите обучение и генерацию прогнозов.")
        return

    generated_at = latest_generated_at(engine, model_name, horizon_years)
    if generated_at is None:
        st.warning("Нет данных для выбранного горизонта.")
        return

    df = load_top5_fact_forecast(engine, model_name, generated_at, horizon_years)
    if df.empty:
        st.warning("Нет данных TOP-5 для выбранного горизонта.")
        return

    mae, rmse, wape, n = compute_error_metrics(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Модель", model_name)
    c2.metric("Сформировано", str(generated_at)[:19])
    c3.metric("WAPE, %", f"{wape:.2f}" if wape is not None else "н/д")
    c4.metric("Точек сравнения", n)
    if mae is not None and rmse is not None:
        st.caption(f"MAE: {int(mae):,}".replace(",", " ") + " | RMSE: " + f"{int(rmse):,}".replace(",", " "))

    melted = df.melt(
        id_vars=["country", "year"],
        value_vars=["tourists_cnt_fact", "yhat"],
        var_name="Ряд",
        value_name="Туристов",
    ).dropna(subset=["Туристов"])
    melted["Ряд"] = melted["Ряд"].replace({"tourists_cnt_fact": "Факт", "yhat": "Прогноз"})

    fig = px.line(
        melted,
        x="year",
        y="Туристов",
        color="Ряд",
        line_dash="Ряд",
        facet_col="country",
        facet_col_wrap=2,
        markers=True,
        labels={"year": "Год"},
        title="TOP-5 стран: факт и прогноз",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig, use_container_width=True)

    table = df.rename(
        columns={
            "country": "Страна",
            "year": "Год",
            "tourists_cnt_fact": "Факт",
            "yhat": "Прогноз",
            "yhat_lower": "Нижняя граница",
            "yhat_upper": "Верхняя граница",
        }
    )
    for col in ["Факт", "Прогноз", "Нижняя граница", "Верхняя граница"]:
        table[col] = table[col].apply(lambda x: f"{int(x):,}".replace(",", " ") if pd.notna(x) else "—")
    st.dataframe(table[["Страна", "Год", "Факт", "Прогноз", "Нижняя граница", "Верхняя граница"]], use_container_width=True)


if __name__ == "__main__":
    main()
