import numpy as np
import pandas as pd
from sqlalchemy import text
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.utils.db import get_engine

HORIZON_YEARS = 5
MODEL_NAME = "SARIMAX(1,1,1)"


def build_continuous_series(ts: pd.DataFrame) -> pd.Series:
    ts = ts.sort_values("year")
    year_min = int(ts["year"].min())
    year_max = int(ts["year"].max())
    idx = pd.Index(range(year_min, year_max + 1), name="year")
    s = ts.set_index("year")["tourists_cnt"].reindex(idx)
    s = s.interpolate(limit_direction="both")
    s = s.clip(lower=0)
    return s


def safe_expm1(values) -> pd.Series:
    return np.expm1(np.clip(values, -50, 50))

def filter_pandemic_outliers(ts: pd.DataFrame) -> pd.DataFrame:
    ts = ts.sort_values("year")
    pre = ts[ts["year"] <= 2019].copy()
    if pre.empty:
        return ts
    baseline = pre["tourists_cnt"].tail(5).median()
    if pd.isna(baseline) or baseline <= 0:
        return ts
    mask = (ts["year"] >= 2020) & (ts["tourists_cnt"] < 0.6 * baseline)
    return ts[~mask]


def main():
    engine = get_engine()

    # Берём годовой ряд
    ts = pd.read_sql(
        "SELECT year, tourists_cnt FROM mart.v_yearly_flow_total ORDER BY year",
        engine
    )

    ts = ts.dropna()
    ts["year"] = ts["year"].astype(int)
    ts["tourists_cnt"] = ts["tourists_cnt"].astype(float)
    ts = filter_pandemic_outliers(ts)

    s = build_continuous_series(ts)
    y = pd.Series(
        s.values,
        index=pd.PeriodIndex(s.index, freq="Y"),
    )

    # SARIMAX для тренда (годовая частота, сезонности нет)
    y_log = np.log1p(y)
    model = SARIMAX(
        y_log,
        order=(1, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    years_hist = s.index.tolist()
    last_year = int(y.index.max().year)
    future_years = list(range(last_year + 1, last_year + 1 + HORIZON_YEARS))

    pred_in = res.get_prediction(start=0, end=len(y_log) - 1)
    mean_in = safe_expm1(pred_in.predicted_mean).clip(lower=0)
    conf_in = pred_in.conf_int(alpha=0.2).applymap(safe_expm1).clip(lower=0)

    pred_out = res.get_forecast(steps=HORIZON_YEARS)
    mean_out = safe_expm1(pred_out.predicted_mean).clip(lower=0)
    conf_out = pred_out.conf_int(alpha=0.2).applymap(safe_expm1).clip(lower=0)

    years_all = years_hist + future_years
    yhat_all = list(mean_in.values) + list(mean_out.values)
    lo_all = list(conf_in.iloc[:, 0].values) + list(conf_out.iloc[:, 0].values)
    hi_all = list(conf_in.iloc[:, 1].values) + list(conf_out.iloc[:, 1].values)

    out = pd.DataFrame({
        "year": years_all,
        "month": [1] * len(years_all),
        "destination_id": [None] * len(years_all),
        "yhat": yhat_all,
        "yhat_lower": lo_all,
        "yhat_upper": hi_all,
        "model_name": [MODEL_NAME] * len(years_all),
        "horizon_months": [HORIZON_YEARS * 12] * len(years_all),
    })

    with engine.begin() as conn:
        # удалим предыдущие прогнозы этой модели для total
        conn.execute(text("""
            DELETE FROM dwh.fact_forecast_monthly
            WHERE destination_id IS NULL AND model_name = :m
        """), {"m": MODEL_NAME})

    out.to_sql("fact_forecast_monthly", engine, schema="dwh", if_exists="append", index=False)
    print(out)

if __name__ == "__main__":
    main()
