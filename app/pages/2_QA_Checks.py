import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


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
def load_qa(_engine):
    with _engine.begin() as conn:
        fact_cnt = conn.execute(text("SELECT COUNT(*) FROM dwh.fact_flow")).scalar()
        dest_cnt = conn.execute(text("SELECT COUNT(*) FROM dwh.dim_destination")).scalar()
        years_cnt = conn.execute(text("SELECT COUNT(DISTINCT dd.year) FROM dwh.fact_flow ff JOIN dwh.dim_date dd ON dd.date_id=ff.date_id")).scalar()
        null_dest = conn.execute(text("SELECT COUNT(*) FROM dwh.fact_flow WHERE destination_id IS NULL")).scalar()
        null_date = conn.execute(text("SELECT COUNT(*) FROM dwh.fact_flow WHERE date_id IS NULL")).scalar()
        min_year = conn.execute(text("SELECT MIN(dd.year) FROM dwh.fact_flow ff JOIN dwh.dim_date dd ON dd.date_id=ff.date_id")).scalar()
        max_year = conn.execute(text("SELECT MAX(dd.year) FROM dwh.fact_flow ff JOIN dwh.dim_date dd ON dd.date_id=ff.date_id")).scalar()
    return {
        "fact_cnt": fact_cnt,
        "dest_cnt": dest_cnt,
        "years_cnt": years_cnt,
        "null_dest": null_dest,
        "null_date": null_date,
        "min_year": min_year,
        "max_year": max_year,
    }


def main():
    st.set_page_config(page_title="Контроль качества", layout="wide")
    st.title("Контроль качества данных")
    st.caption("Проверка полноты и консистентности данных в хранилище.")

    qa = load_qa(get_engine())
    c1, c2, c3 = st.columns(3)
    c1.metric("Строк в факте", f"{qa['fact_cnt']:,}".replace(",", " "))
    c2.metric("Направлений", f"{qa['dest_cnt']:,}".replace(",", " "))
    c3.metric("Покрытие лет", f"{qa['years_cnt']:,}".replace(",", " "))

    c4, c5, c6 = st.columns(3)
    c4.metric("Пустой destination_id", f"{qa['null_dest']:,}".replace(",", " "))
    c5.metric("Пустой date_id", f"{qa['null_date']:,}".replace(",", " "))
    c6.metric("Диапазон лет", f"{qa['min_year']} - {qa['max_year']}")

    checks = [
        ("Факт загружен", qa["fact_cnt"] > 0, qa["fact_cnt"]),
        ("Справочник направлений заполнен", qa["dest_cnt"] > 0, qa["dest_cnt"]),
        ("Нет NULL destination_id", qa["null_dest"] == 0, qa["null_dest"]),
        ("Нет NULL date_id", qa["null_date"] == 0, qa["null_date"]),
    ]
    report = pd.DataFrame(
        {"Проверка": [r[0] for r in checks], "Статус": ["OK" if r[1] else "Проверить" for r in checks], "Значение": [r[2] for r in checks]}
    )
    st.dataframe(report, use_container_width=True, hide_index=True)

    if any(not r[1] for r in checks):
        st.error("Есть проблемы с качеством данных. Требуется повторная загрузка.")
    else:
        st.success("Проверки пройдены. Данные готовы для аналитики.")


if __name__ == "__main__":
    main()
