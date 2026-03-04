import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from src.ml.model_registry import save_metric_rows

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
def list_countries(_engine):
    return pd.read_sql(
        """
        SELECT DISTINCT country
        FROM dwh.dim_destination
        WHERE country IS NOT NULL
        ORDER BY country
        """,
        _engine,
    )["country"].tolist()


@st.cache_data(ttl=300)
def load_total_paired(_engine):
    return pd.read_sql(
        text(
            """
            WITH fact AS (
              SELECT dd.year, SUM(ff.tourists_cnt) AS fact
              FROM dwh.fact_flow ff
              JOIN dwh.dim_date dd ON dd.date_id = ff.date_id
              GROUP BY dd.year
            ),
            latest AS (
              SELECT model_name, horizon_months, MAX(generated_at) AS generated_at
              FROM dwh.fact_forecast_monthly
              WHERE destination_id IS NULL
              GROUP BY model_name, horizon_months
            )
            SELECT f.model_name, f.horizon_months/12 AS horizon_years, f.year, fact.fact, f.yhat
            FROM dwh.fact_forecast_monthly f
            JOIN latest l ON l.model_name=f.model_name AND l.horizon_months=f.horizon_months AND l.generated_at=f.generated_at
            JOIN fact ON fact.year=f.year
            WHERE f.destination_id IS NULL
            ORDER BY f.model_name, f.horizon_months, f.year
            """
        ),
        _engine,
    )


@st.cache_data(ttl=300)
def load_country_paired(_engine, country):
    return pd.read_sql(
        text(
            """
            WITH fact AS (
              SELECT d.destination_id, dd.year, SUM(ff.tourists_cnt) AS fact
              FROM dwh.fact_flow ff
              JOIN dwh.dim_date dd ON dd.date_id = ff.date_id
              JOIN dwh.dim_destination d ON d.destination_id = ff.destination_id
              WHERE d.country = :c
              GROUP BY d.destination_id, dd.year
            ),
            latest AS (
              SELECT model_name, horizon_months, MAX(generated_at) AS generated_at
              FROM dwh.fact_forecast_monthly
              WHERE destination_id IS NOT NULL
              GROUP BY model_name, horizon_months
            )
            SELECT f.model_name, f.horizon_months/12 AS horizon_years, f.year, fact.fact, f.yhat
            FROM dwh.fact_forecast_monthly f
            JOIN latest l ON l.model_name=f.model_name AND l.horizon_months=f.horizon_months AND l.generated_at=f.generated_at
            JOIN fact ON fact.destination_id=f.destination_id AND fact.year=f.year
            WHERE f.destination_id IS NOT NULL
            ORDER BY f.model_name, f.horizon_months, f.year
            """
        ),
        _engine,
        params={"c": country},
    )


@st.cache_data(ttl=300)
def load_version_metrics(_engine, model_name="NN_MLP_GLOBAL", eval_type="in_sample_total"):
    try:
        return pd.read_sql(
            text(
                """
                SELECT model_version, horizon_years, n_points, mae, rmse, mape, smape, wape, created_at
                FROM dwh.model_metrics_history
                WHERE model_name=:m AND eval_type=:e
                ORDER BY created_at DESC, horizon_years
                """
            ),
            _engine,
            params={"m": model_name, "e": eval_type},
        )
    except Exception:
        return pd.DataFrame()


def _metrics_from_group(df):
    x = df.dropna(subset=["fact", "yhat"]).copy()
    if x.empty:
        return None
    err = x["fact"] - x["yhat"]
    mae = float(err.abs().mean())
    rmse = float(np.sqrt((err.pow(2)).mean()))
    mape_den = x["fact"].replace(0, np.nan)
    mape = float((err.abs() / mape_den).dropna().mean() * 100) if mape_den.notna().any() else np.nan
    smape_den = (x["fact"].abs() + x["yhat"].abs()).replace(0, np.nan)
    smape = float((200.0 * err.abs() / smape_den).dropna().mean()) if smape_den.notna().any() else np.nan
    sum_fact = float(x["fact"].abs().sum())
    wape = float(err.abs().sum() / sum_fact * 100) if sum_fact > 0 else np.nan
    return {"n_points": int(len(x)), "mae": mae, "rmse": rmse, "mape": mape, "smape": smape, "wape": wape}


def aggregate_metrics(df):
    rows = []
    for (model_name, horizon_years), grp in df.groupby(["model_name", "horizon_years"]):
        m = _metrics_from_group(grp)
        if m is None:
            continue
        rows.append({"Источник": model_name, "Тип": "Модель", "Горизонт, лет": int(horizon_years), **m})
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def nn_horizon_backtest_total(_engine, horizon_min=5, horizon_max=10, test_windows=10, window=3, epochs=300, lr=0.003):
    try:
        import torch
        from torch import nn
    except Exception:
        return None, "PyTorch не установлен."

    fact = pd.read_sql(
        """
        SELECT dd.year, SUM(ff.tourists_cnt) AS fact
        FROM dwh.fact_flow ff
        JOIN dwh.dim_date dd ON dd.date_id = ff.date_id
        GROUP BY dd.year
        ORDER BY dd.year
        """,
        _engine,
    ).dropna()

    max_h = int(horizon_max)
    min_h = int(horizon_min)
    if len(fact) < (window + max_h + 4):
        return None, "Недостаточно данных для backtest."

    s = fact.set_index("year")["fact"].astype(float)
    idx = pd.Index(range(int(s.index.min()), int(s.index.max()) + 1), name="year")
    s = s.reindex(idx).interpolate(limit_direction="both").clip(lower=0)
    years = s.index.tolist()
    values = np.log1p(s.values)

    class MLP(nn.Module):
        def __init__(self, win):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(win, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def train_one(train_vals):
        xs, ys = [], []
        for i in range(window, len(train_vals)):
            xs.append(train_vals[i - window:i])
            ys.append(train_vals[i])
        if not xs:
            return None, None
        x_t = torch.tensor(np.stack(xs), dtype=torch.float32)
        y_t = torch.tensor(np.array(ys), dtype=torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP(window).to(device)
        x_t, y_t = x_t.to(device), y_t.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn = nn.SmoothL1Loss()
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(x_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()
        return model.eval(), device

    def forecast_multi(model, device, history_log, steps):
        hist = list(history_log)
        preds = []
        with torch.no_grad():
            for _ in range(steps):
                inp = torch.tensor(hist[-window:], dtype=torch.float32).to(device)
                pred_log = float(model(inp).cpu().item())
                hist.append(pred_log)
                preds.append(pred_log)
        return preds

    rows = []
    split_end_max = len(values) - max_h
    split_end_start = max(window + 2, split_end_max - test_windows + 1)
    for split_end in range(split_end_start, split_end_max + 1):
        model, device = train_one(values[:split_end])
        if model is None:
            continue
        preds = forecast_multi(model, device, values[:split_end], max_h)
        for h in range(min_h, max_h + 1):
            target_idx = split_end + h - 1
            rows.append(
                {
                    "horizon_years": h,
                    "fact": max(0.0, float(np.expm1(np.clip(values[target_idx], -50, 50)))),
                    "yhat": max(0.0, float(np.expm1(np.clip(preds[h - 1], -50, 50)))),
                }
            )

    raw = pd.DataFrame(rows)
    if raw.empty:
        return None, "Backtest не дал результатов."

    metric_rows = []
    for h, grp in raw.groupby("horizon_years"):
        m = _metrics_from_group(grp)
        if m:
            metric_rows.append({"horizon_years": int(h), **m})
    metrics = pd.DataFrame(metric_rows).sort_values("horizon_years")
    if metrics.empty:
        return None, "Нет метрик backtest."
    return {"device": "cuda" if torch.cuda.is_available() else "cpu", "metrics": metrics}, None


def main():
    st.set_page_config(page_title="Сравнение моделей", layout="wide")
    st.title("Сравнение моделей прогноза")
    st.caption("Один график и одна таблица. Дублирующиеся ряды скрываются автоматически.")

    engine = get_engine()
    scope = st.radio("Область сравнения", ["Общий поток", "По стране"], horizontal=True)
    period = st.radio("Период оценки", ["Весь период", "До 2020", "С 2020"], horizontal=True)

    if scope == "По стране":
        countries = list_countries(engine)
        country = st.selectbox("Страна", countries, index=countries.index("France") if "France" in countries else 0)
        raw = load_country_paired(engine, country)
    else:
        raw = load_total_paired(engine)

    if raw.empty:
        st.warning("Недостаточно данных для сравнения.")
        return

    if period == "До 2020":
        raw = raw[raw["year"] <= 2019]
    elif period == "С 2020":
        raw = raw[raw["year"] >= 2020]

    model_df = aggregate_metrics(raw)
    if model_df.empty:
        st.warning("Нет пар факт/прогноз после фильтрации.")
        return

    version_hist = load_version_metrics(engine, model_name="NN_MLP_GLOBAL", eval_type="in_sample_total")
    if not version_hist.empty:
        version_df = (
            version_hist.sort_values(["model_version", "created_at"], ascending=[True, False])
            .drop_duplicates(subset=["model_version", "horizon_years"], keep="first")
            .rename(columns={"model_version": "Источник", "horizon_years": "Горизонт, лет"})
        )
        version_df["Тип"] = "Версия"
        version_df = version_df[["Тип", "Источник", "Горизонт, лет", "n_points", "mae", "rmse", "mape", "smape", "wape"]]
    else:
        version_df = pd.DataFrame(columns=["Тип", "Источник", "Горизонт, лет", "n_points", "mae", "rmse", "mape", "smape", "wape"])

    combined = pd.concat(
        [
            model_df[["Тип", "Источник", "Горизонт, лет", "n_points", "mae", "rmse", "mape", "smape", "wape"]],
            version_df,
        ],
        ignore_index=True,
    )
    combined["Горизонт, лет"] = pd.to_numeric(combined["Горизонт, лет"], errors="coerce").astype("Int64")
    combined = combined.dropna(subset=["Горизонт, лет"]).copy()
    combined["series_label"] = combined["Тип"] + ":" + combined["Источник"]
    combined["Горизонт"] = combined["Горизонт, лет"].astype(int).astype(str) + "y"

    # Скрываем полностью совпадающие ряды по всем метрикам.
    metric_cols = ["mae", "rmse", "mape", "smape", "wape"]
    sig_rows = []
    for label, grp in combined.sort_values(["series_label", "Горизонт, лет"]).groupby("series_label"):
        sig = tuple(np.round(grp[metric_cols].to_numpy().flatten(), 6))
        sig_rows.append({"series_label": label, "signature": sig})
    sig_df = pd.DataFrame(sig_rows)
    keep_labels = sig_df.drop_duplicates(subset=["signature"], keep="first")["series_label"].tolist()
    hidden_labels = sorted(set(combined["series_label"].unique()) - set(keep_labels))
    shown = combined[combined["series_label"].isin(keep_labels)].copy()

    metric = st.selectbox("Метрика для графика", ["wape", "smape", "rmse", "mae"], index=0)
    fig = px.bar(
        shown.sort_values(["Горизонт, лет", "series_label"]),
        x="Горизонт",
        y=metric,
        color="series_label",
        barmode="group",
        labels={"Горизонт": "Горизонт", metric: metric.upper(), "series_label": "Модель/версия"},
        title=f"Сравнение по метрике {metric.upper()}",
    )
    st.plotly_chart(fig, use_container_width=True)
    if hidden_labels:
        st.info("Скрыты идентичные ряды: " + ", ".join(hidden_labels))

    # Единая таблица.
    table = shown.rename(
        columns={
            "n_points": "Точек",
            "mae": "MAE",
            "rmse": "RMSE",
            "mape": "MAPE, %",
            "smape": "sMAPE, %",
            "wape": "WAPE, %",
        }
    )[["Тип", "Источник", "Горизонт, лет", "Точек", "MAE", "RMSE", "MAPE, %", "sMAPE, %", "WAPE, %"]]
    for col in ["MAE", "RMSE"]:
        table[col] = table[col].apply(lambda x: f"{x:,.0f}".replace(",", " ") if pd.notna(x) else "н/д")
    for col in ["MAPE, %", "sMAPE, %", "WAPE, %"]:
        table[col] = table[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "н/д")
    st.dataframe(table.sort_values(["Тип", "Источник", "Горизонт, лет"]), use_container_width=True, hide_index=True)

    with st.expander("Горизонтальный backtest NN (t+h, общий поток)"):
        h_min, h_max = st.select_slider("Горизонты", options=[5, 6, 7, 8, 9, 10], value=(5, 10))
        test_windows = st.slider("Скользящие окна", 4, 14, 10)
        epochs = st.slider("Эпохи обучения на сплит", 100, 700, 300, step=50)
        lr = st.slider("Скорость обучения", 0.001, 0.01, 0.003, step=0.001)
        versions = sorted(version_hist["model_version"].unique().tolist(), reverse=True) if not version_hist.empty else ["unspecified"]
        selected_version = st.selectbox("Версия для сохранения метрик backtest", versions, index=0)
        if st.button("Запустить backtest"):
            bt, err = nn_horizon_backtest_total(engine, h_min, h_max, test_windows, 3, epochs, lr)
            if err:
                st.error(err)
            else:
                mdf = bt["metrics"].copy().sort_values("horizon_years")
                b1, b2, b3 = st.columns(3)
                b1.metric("Устройство", bt["device"])
                b2.metric("Лучший горизонт по WAPE", int(mdf.sort_values("wape").iloc[0]["horizon_years"]))
                b3.metric("Лучший WAPE, %", f"{mdf['wape'].min():.2f}")

                fig_bt = px.line(mdf, x="horizon_years", y=["wape", "smape"], markers=True, title="Метрики backtest по горизонтам")
                st.plotly_chart(fig_bt, use_container_width=True)

                out = mdf.rename(
                    columns={
                        "horizon_years": "Горизонт, лет",
                        "n_points": "Точек",
                        "mae": "MAE",
                        "rmse": "RMSE",
                        "mape": "MAPE, %",
                        "smape": "sMAPE, %",
                        "wape": "WAPE, %",
                    }
                )
                for col in ["MAE", "RMSE"]:
                    out[col] = out[col].apply(lambda x: f"{x:,.0f}".replace(",", " "))
                for col in ["MAPE, %", "sMAPE, %", "WAPE, %"]:
                    out[col] = out[col].apply(lambda x: f"{x:.2f}")
                st.dataframe(out, use_container_width=True, hide_index=True)

                save_metric_rows(
                    engine=engine,
                    model_name="NN_MLP_GLOBAL",
                    model_version=selected_version,
                    eval_type="horizon_backtest_total",
                    metrics_df=mdf[["horizon_years", "n_points", "mae", "rmse", "mape", "smape", "wape"]],
                )
                st.success(f"Метрики backtest сохранены для версии: {selected_version}")


if __name__ == "__main__":
    main()
