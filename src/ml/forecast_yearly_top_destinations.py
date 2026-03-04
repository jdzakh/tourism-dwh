import numpy as np
import pandas as pd
from sqlalchemy import text
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.utils.db import get_engine

HORIZON_YEARS = 5
MODEL_NAME = "SARIMAX(1,1,1)_TOP5_COUNTRIES"


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

def forecast_series(y: pd.Series):
    # y: year -> value
    y_log = np.log1p(y)
    model = SARIMAX(y_log, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred_in = res.get_prediction(start=0, end=len(y_log) - 1)
    mean_in = safe_expm1(pred_in.predicted_mean).clip(lower=0)
    conf_in = pred_in.conf_int(alpha=0.2).applymap(safe_expm1).clip(lower=0)

    pred_out = res.get_forecast(steps=HORIZON_YEARS)
    mean_out = safe_expm1(pred_out.predicted_mean).clip(lower=0)
    conf_out = pred_out.conf_int(alpha=0.2).applymap(safe_expm1).clip(lower=0)

    return (
        mean_in.values,
        conf_in.iloc[:, 0].values,
        conf_in.iloc[:, 1].values,
        mean_out.values,
        conf_out.iloc[:, 0].values,
        conf_out.iloc[:, 1].values,
    )


def clamp_nonnegative(arr):
    return [max(0.0, float(x)) if x is not None else None for x in arr]


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

    top = pd.read_sql(
        "SELECT destination_id, country FROM mart.v_top_countries LIMIT 5",
        engine
    )

    # удаляем предыдущие прогнозы этой модели (только для destination_id not null)
    with engine.begin() as conn:
        conn.execute(text("""
            DELETE FROM dwh.fact_forecast_monthly
            WHERE destination_id IS NOT NULL AND model_name = :m
        """), {"m": MODEL_NAME})

    all_out = []

    for _, row in top.iterrows():
        dest_id = int(row["destination_id"])
        country = row["country"]

        ts = pd.read_sql(
            text(
                """
                SELECT dd.year, SUM(ff.tourists_cnt) AS tourists_cnt
                FROM dwh.fact_flow ff
                JOIN dwh.dim_date dd ON dd.date_id = ff.date_id
                WHERE ff.destination_id = :dest
                GROUP BY dd.year
                ORDER BY dd.year
                """
            ),
            engine,
            params={"dest": dest_id},
        ).dropna()

        if ts.shape[0] < 10:
            print(f"Skip {country}: too few points ({ts.shape[0]})")
            continue

        ts["year"] = ts["year"].astype(int)
        ts["tourists_cnt"] = ts["tourists_cnt"].astype(float)
        ts = filter_pandemic_outliers(ts)
        s = build_continuous_series(ts)

        # важно: сделаем индекс "годовой" как PeriodIndex -> меньше warning
        y = pd.Series(s.values, index=pd.PeriodIndex(s.index, freq="Y"))

        mean_in, lo_in, hi_in, mean_out, lo_out, hi_out = forecast_series(y)
        last_year = int(s.index.max())
        future_years = list(range(last_year + 1, last_year + 1 + HORIZON_YEARS))
        years_all = s.index.tolist() + future_years

        mean_all = clamp_nonnegative(list(mean_in) + list(mean_out))
        lo_all = clamp_nonnegative(list(lo_in) + list(lo_out))
        hi_all = clamp_nonnegative(list(hi_in) + list(hi_out))

        out = pd.DataFrame({
            "year": years_all,
            "month": [1] * len(years_all),
            "destination_id": [dest_id] * len(years_all),
            "yhat": mean_all,
            "yhat_lower": lo_all,
            "yhat_upper": hi_all,
            "model_name": [MODEL_NAME] * len(years_all),
            "horizon_months": [HORIZON_YEARS * 12] * len(years_all),
        })

        all_out.append(out)
        print(f"Forecasted: {country}")

    if all_out:
        res = pd.concat(all_out, ignore_index=True)
        res.to_sql("fact_forecast_monthly", engine, schema="dwh", if_exists="append", index=False)
        print("Inserted forecast rows:", len(res))
    else:
        print("No forecasts inserted.")

if __name__ == "__main__":
    main()
