import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF, XPos, YPos
from sqlalchemy import create_engine, text

load_dotenv()
px.defaults.template = "plotly_dark"


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@400;600&display=swap');
        :root {
          --ink: #eef2f6;
          --muted: #9aa6b2;
          --accent: #f6c36a;
          --card: #0f172ad9;
          --border: #1f2937;
        }
        .stApp {
          background: radial-gradient(1200px 700px at 80% -10%, #111827 0%, #0b0f14 55%, #0b0f14 100%),
                      radial-gradient(900px 600px at 0% 0%, #1f2937 0%, #0b0f14 60%);
          color: var(--ink);
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { padding-top: 2rem; }
        h1, h2, h3 { font-family: "Playfair Display", "Georgia", serif; }
        body, p, div, span, label, input, button, textarea {
          font-family: "Source Sans 3", "Segoe UI", sans-serif;
        }
        .stMetric {
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 12px;
        }
        .hero {
          background: linear-gradient(135deg, #111827 0%, #0f172a 50%, #0b0f14 100%);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 20px 24px;
          margin-bottom: 14px;
        }
        .account-icon {
          position: fixed;
          top: 64px;
          right: 18px;
          width: 42px;
          height: 42px;
          border-radius: 50%;
          background: #111827;
          border: 1px solid var(--border);
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--accent);
          text-decoration: none;
          font-size: 20px;
          z-index: 9999;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
def load_top_countries(_engine, n=15):
    return pd.read_sql(
        text(
            """
            SELECT country, country_code, tourists_cnt_total
            FROM mart.v_top_countries
            LIMIT :n
            """
        ),
        _engine,
        params={"n": n},
    )


@st.cache_data(ttl=300)
def load_country_map_data(_engine):
    return pd.read_sql(
        """
        SELECT country, country_code, tourists_cnt_total
        FROM mart.v_top_countries
        WHERE country_code IS NOT NULL AND length(country_code)=3
        """,
        _engine,
    )


@st.cache_data(ttl=300)
def load_total_fact_forecast(_engine, horizon_years, model_name):
    q = text(
        """
        WITH fact AS (
          SELECT dd.year, SUM(ff.tourists_cnt) AS tourists_cnt_fact
          FROM dwh.fact_flow ff
          JOIN dwh.dim_date dd ON dd.date_id = ff.date_id
          GROUP BY dd.year
        ),
        fc AS (
          SELECT year, yhat, yhat_lower, yhat_upper
          FROM dwh.fact_forecast_monthly
          WHERE destination_id IS NULL
            AND model_name = :m
            AND horizon_months = :h
            AND generated_at = (
              SELECT MAX(generated_at)
              FROM dwh.fact_forecast_monthly
              WHERE destination_id IS NULL AND model_name = :m AND horizon_months = :h
            )
        )
        SELECT
          COALESCE(fact.year, fc.year) AS year,
          fact.tourists_cnt_fact,
          fc.yhat AS tourists_cnt_forecast,
          fc.yhat_lower,
          fc.yhat_upper
        FROM fact
        FULL JOIN fc ON fc.year = fact.year
        ORDER BY year
        """
    )
    return pd.read_sql(q, _engine, params={"m": model_name, "h": horizon_years * 12})


@st.cache_data(ttl=300)
def load_country_series(_engine, country, horizon_years, model_name):
    q = text(
        """
        WITH fact AS (
          SELECT ff.destination_id, d.country, dd.year, SUM(ff.tourists_cnt) AS tourists_cnt_fact
          FROM dwh.fact_flow ff
          JOIN dwh.dim_date dd ON dd.date_id = ff.date_id
          JOIN dwh.dim_destination d ON d.destination_id = ff.destination_id
          WHERE d.country = :c
          GROUP BY ff.destination_id, d.country, dd.year
        ),
        fc AS (
          SELECT destination_id, year, yhat, yhat_lower, yhat_upper
          FROM dwh.fact_forecast_monthly
          WHERE destination_id IS NOT NULL
            AND model_name = :m
            AND horizon_months = :h
            AND generated_at = (
              SELECT MAX(generated_at)
              FROM dwh.fact_forecast_monthly
              WHERE destination_id IS NOT NULL AND model_name = :m AND horizon_months = :h
            )
        )
        SELECT
          COALESCE(fact.year, fc.year) AS year,
          fact.tourists_cnt_fact,
          fc.yhat AS tourists_cnt_forecast,
          fc.yhat_lower,
          fc.yhat_upper
        FROM fact
        FULL JOIN fc ON fc.destination_id = fact.destination_id AND fc.year = fact.year
        LEFT JOIN dwh.dim_destination d ON d.destination_id = COALESCE(fact.destination_id, fc.destination_id)
        WHERE d.country = :c
        ORDER BY year
        """
    )
    return pd.read_sql(q, _engine, params={"c": country, "m": model_name, "h": horizon_years * 12})


@st.cache_data(ttl=300)
def list_countries(_engine):
    return pd.read_sql(
        """
        SELECT DISTINCT country
        FROM mart.v_yearly_fact_forecast_by_country
        WHERE country IS NOT NULL
        ORDER BY country
        """,
        _engine,
    )["country"].tolist()


def _fmt_int(value):
    if value is None or pd.isna(value):
        return "н/д"
    return f"{int(value):,}".replace(",", " ")


def compute_error_metrics(df, fact_col="tourists_cnt_fact", forecast_col="tourists_cnt_forecast"):
    paired = df[[fact_col, forecast_col]].dropna()
    if paired.empty:
        return None, None, None, None, 0
    err = paired[fact_col] - paired[forecast_col]
    mae = float(err.abs().mean())
    rmse = float((err.pow(2).mean()) ** 0.5)
    mape = float((err.abs() / paired[fact_col].replace(0, pd.NA)).dropna().mean() * 100)
    wape_den = float(paired[fact_col].abs().sum())
    wape = float(err.abs().sum() / wape_den * 100) if wape_den > 0 else None
    return mae, rmse, mape, wape, int(len(paired))


def build_pdf_report(country, total, country_series):
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=14)
    pdf.add_page()

    font_path = os.path.join(os.path.dirname(__file__), "assets", "fonts", "ArialUnicode.ttf")
    pdf.add_font("ArialUnicode", "", font_path)
    pdf.add_font("ArialUnicode", "B", font_path)
    pdf.set_font("ArialUnicode", "B", 16)
    pdf.cell(0, 10, "Отчет по туристическим потокам", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("ArialUnicode", "", 10)
    pdf.cell(0, 6, f"Сформирован: {datetime.now().strftime('%d.%m.%Y %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    last_fact = total.dropna(subset=["tourists_cnt_fact"]).tail(1)
    last_fc = total.dropna(subset=["tourists_cnt_forecast"]).head(1)
    if not last_fact.empty:
        pdf.cell(0, 6, f"Последний год факта: {int(last_fact['year'].iloc[0])}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if not last_fc.empty:
        pdf.cell(0, 6, f"Начало прогноза: {int(last_fc['year'].iloc[0])}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(2)
    pdf.set_font("ArialUnicode", "B", 12)
    pdf.cell(0, 8, f"Страна: {country}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("ArialUnicode", "", 9)
    for _, row in country_series.tail(8).iterrows():
        fact = _fmt_int(row.get("tourists_cnt_fact"))
        fc = _fmt_int(row.get("tourists_cnt_forecast"))
        pdf.cell(0, 6, f"{int(row['year'])}: факт={fact}, прогноз={fc}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    out = pdf.output()
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return out.encode("latin-1")


def build_country_snapshot(engine, countries, horizon_years, model_name):
    rows = []
    for country in countries:
        cs = load_country_series(engine, country, horizon_years, model_name)
        last_fact = cs.dropna(subset=["tourists_cnt_fact"]).tail(1)
        next_fc = cs.dropna(subset=["tourists_cnt_forecast"]).head(1)
        if last_fact.empty or next_fc.empty:
            continue
        fact_v = float(last_fact["tourists_cnt_fact"].iloc[0])
        fc_v = float(next_fc["tourists_cnt_forecast"].iloc[0])
        growth = ((fc_v - fact_v) / fact_v * 100) if fact_v > 0 else None
        rows.append(
            {
                "Страна": country,
                "Последний факт": fact_v,
                "Ближайший прогноз": fc_v,
                "Изменение, %": growth,
            }
        )
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="Туризм DWH", layout="wide")
    inject_css()
    st.title("Аналитика туристических потоков")
    st.markdown('<a class="account-icon" href="/Account" title="Личный кабинет">👤</a>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <div style="font-size:1.15rem;font-weight:700;">Платформа аналитики спроса в туризме</div>
          <div style="margin-top:8px;color:#9aa6b2;">Хранилище данных + ETL + прогнозирование + BI-визуализация.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "country_sidebar" not in st.session_state:
        st.session_state.country_sidebar = None
    if "selected_country" not in st.session_state:
        st.session_state.selected_country = None
    if "country_sidebar_prev" not in st.session_state:
        st.session_state.country_sidebar_prev = None

    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        st.error(f"Ошибка подключения к базе данных: {e}")
        st.stop()

    st.sidebar.header("Параметры")
    mode = st.sidebar.radio("Режим интерфейса", ["Базовый", "Эксперт"], index=0)
    horizon_years = st.sidebar.slider("Горизонт прогноза (лет)", 5, 10, 5)
    model_name = "NN_MLP_GLOBAL"
    top_n = st.sidebar.slider("Топ стран", 5, 30, 10) if mode == "Эксперт" else 10

    countries = list_countries(engine)
    if not countries:
        st.error("Список стран пуст. Проверьте витрину `mart.v_yearly_fact_forecast_by_country`.")
        st.stop()

    default_country = "France" if "France" in countries else countries[0]
    if st.session_state.country_sidebar not in countries:
        st.session_state.country_sidebar = default_country
    st.sidebar.selectbox("Страна", countries, key="country_sidebar")
    # Sync selected country when sidebar selection changes.
    if st.session_state.country_sidebar_prev != st.session_state.country_sidebar:
        st.session_state.selected_country = st.session_state.country_sidebar
        st.session_state.country_sidebar_prev = st.session_state.country_sidebar
    elif st.session_state.selected_country not in countries:
        st.session_state.selected_country = st.session_state.country_sidebar

    total = load_total_fact_forecast(engine, horizon_years, model_name)
    top = load_top_countries(engine, n=top_n)
    cs = load_country_series(engine, st.session_state.selected_country, horizon_years, model_name)
    map_df = load_country_map_data(engine)

    tabs = st.tabs(["Обзор", "Страна", "Экспорт"] if mode == "Эксперт" else ["Обзор", "Страна"])

    with tabs[0]:
        mae, rmse, mape, wape, n_points = compute_error_metrics(total)
        last_fact = total.dropna(subset=["tourists_cnt_fact"]).tail(1)
        last_fc = total.dropna(subset=["tourists_cnt_forecast"]).head(1)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Последний год факта", int(last_fact["year"].iloc[0]) if not last_fact.empty else "н/д")
        c2.metric("Туристов (факт)", _fmt_int(last_fact["tourists_cnt_fact"].iloc[0]) if not last_fact.empty else "н/д")
        c3.metric("Начало прогноза", int(last_fc["year"].iloc[0]) if not last_fc.empty else "н/д")
        c4.metric("WAPE, %", f"{wape:.2f}" if wape is not None else "н/д")

        if mode == "Эксперт":
            c5, c6, c7 = st.columns(3)
            c5.metric("MAE", _fmt_int(mae) if mae is not None else "н/д")
            c6.metric("RMSE", _fmt_int(rmse) if rmse is not None else "н/д")
            c7.metric("MAPE, %", f"{mape:.2f}" if mape is not None else "н/д")
            st.caption(f"Точек сравнения факт/прогноз: {n_points}")

        plot_total = total.copy()
        plot_total["Год"] = pd.to_numeric(plot_total["year"], errors="coerce")
        plot_total["Факт"] = pd.to_numeric(plot_total["tourists_cnt_fact"], errors="coerce")
        plot_total["Прогноз"] = pd.to_numeric(plot_total["tourists_cnt_forecast"], errors="coerce")
        fig_total = px.line(
            plot_total,
            x="Год",
            y=["Факт", "Прогноз"],
            labels={"value": "Туристов", "variable": "Ряд"},
            title="Общий поток: факт и прогноз",
        )
        st.plotly_chart(fig_total, use_container_width=True)

        st.subheader("Карта туристических потоков")
        fig_map = px.choropleth(
            map_df,
            locations="country_code",
            color="tourists_cnt_total",
            hover_name="country",
            color_continuous_scale="YlOrBr",
            labels={"tourists_cnt_total": "Туристов"},
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

        drill_country = st.selectbox(
            "Быстрый переход к стране",
            map_df.sort_values("country")["country"].tolist(),
            index=map_df.sort_values("country")["country"].tolist().index(st.session_state.selected_country)
            if st.session_state.selected_country in map_df["country"].values
            else 0,
        )
        if drill_country != st.session_state.selected_country:
            st.session_state.selected_country = drill_country
            st.info("Страна для вкладки «Страна» обновлена.")

    with tabs[1]:
        country = st.session_state.selected_country
        cs = load_country_series(engine, country, horizon_years, model_name)
        mae_c, rmse_c, mape_c, wape_c, n_c = compute_error_metrics(cs)

        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Страна", country)
        x2.metric("WAPE, %", f"{wape_c:.2f}" if wape_c is not None else "н/д")
        x3.metric("MAE", _fmt_int(mae_c) if mae_c is not None else "н/д")
        x4.metric("RMSE", _fmt_int(rmse_c) if rmse_c is not None else "н/д")
        st.caption(f"Точек сравнения: {n_c}")

        plot_country = cs.copy()
        plot_country["Год"] = pd.to_numeric(plot_country["year"], errors="coerce")
        plot_country["Факт"] = pd.to_numeric(plot_country["tourists_cnt_fact"], errors="coerce")
        plot_country["Прогноз"] = pd.to_numeric(plot_country["tourists_cnt_forecast"], errors="coerce")
        fig_country = px.line(
            plot_country,
            x="Год",
            y=["Факт", "Прогноз"],
            labels={"value": "Туристов", "variable": "Ряд"},
            title=f"{country}: факт и прогноз",
        )
        st.plotly_chart(fig_country, use_container_width=True)

        st.markdown("### Сравнение стран")
        st.caption("Новый функционал: сравнение выбранных стран по последнему факту и ближайшему прогнозу.")
        compare_list = st.multiselect(
            "Выберите 2-5 стран",
            options=countries,
            default=[country] if country in countries else [],
            max_selections=5,
        )
        if len(compare_list) >= 2:
            snap = build_country_snapshot(engine, compare_list, horizon_years, model_name)
            if snap.empty:
                st.warning("Для выбранных стран недостаточно данных для сравнения.")
            else:
                fig_snap = px.bar(
                    snap.melt(
                        id_vars=["Страна"],
                        value_vars=["Последний факт", "Ближайший прогноз"],
                        var_name="Показатель",
                        value_name="Значение",
                    ),
                    x="Страна",
                    y="Значение",
                    color="Показатель",
                    barmode="group",
                    title="Сравнение стран: последний факт и ближайший прогноз",
                )
                st.plotly_chart(fig_snap, use_container_width=True)
                snap_display = snap.copy()
                snap_display["Последний факт"] = snap_display["Последний факт"].apply(_fmt_int)
                snap_display["Ближайший прогноз"] = snap_display["Ближайший прогноз"].apply(_fmt_int)
                snap_display["Изменение, %"] = snap_display["Изменение, %"].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "н/д"
                )
                st.dataframe(snap_display, use_container_width=True, hide_index=True)
        else:
            st.info("Чтобы включить сравнение, выберите минимум две страны.")

        if mode == "Эксперт":
            table = cs[["year", "tourists_cnt_fact", "tourists_cnt_forecast", "yhat_lower", "yhat_upper"]].rename(
                columns={
                    "year": "Год",
                    "tourists_cnt_fact": "Факт",
                    "tourists_cnt_forecast": "Прогноз",
                    "yhat_lower": "Нижняя граница",
                    "yhat_upper": "Верхняя граница",
                }
            )
            table["Год"] = table["Год"].astype("Int64").astype(str)
            for col in ["Факт", "Прогноз", "Нижняя граница", "Верхняя граница"]:
                table[col] = table[col].apply(lambda x: _fmt_int(x) if pd.notna(x) else "—")
            st.dataframe(table.tail(12), use_container_width=True, hide_index=True)

    if mode == "Эксперт":
        with tabs[2]:
            st.subheader("Экспорт отчета")
            st.caption("Скачайте PDF с ключевыми цифрами по рынку и выбранной стране.")
            report_bytes = build_pdf_report(st.session_state.selected_country, total, cs)
            st.download_button(
                "Скачать отчет (PDF)",
                data=report_bytes,
                file_name=f"tourism_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )


if __name__ == "__main__":
    main()
