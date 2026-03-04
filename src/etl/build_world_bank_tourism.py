import os

import pandas as pd

ARRIVALS_CSV = "data/raw/world_bank/arrivals/API_ST.INT.ARVL_DS2_EN_csv_v2_21669.csv"
RECEIPTS_CSV = "data/raw/world_bank/receipts/API_ST.INT.RCPT.CD_DS2_EN_csv_v2_38807.csv"
OUT_PATH = "data/processed/world_bank_tourism.csv"


def load_indicator(path: str, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=4, encoding="utf-8-sig")
    df = df.rename(columns={"Country Name": "country", "Country Code": "country_code"})
    year_cols = [c for c in df.columns if c.isdigit()]
    df = df[["country", "country_code"] + year_cols]
    df = df.melt(
        id_vars=["country", "country_code"],
        value_vars=year_cols,
        var_name="year",
        value_name=value_col,
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df


def main():
    arrivals = load_indicator(ARRIVALS_CSV, "tourism_arrivals")
    receipts = load_indicator(RECEIPTS_CSV, "tourism_receipts")

    merged = arrivals.merge(
        receipts, on=["country", "country_code", "year"], how="outer"
    )
    merged = merged.dropna(subset=["country", "country_code", "year"])
    merged = merged.dropna(
        subset=["tourism_arrivals", "tourism_receipts"], how="all"
    )

    merged["year"] = merged["year"].astype(int)
    merged = merged.sort_values(["country", "year"])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH} rows={len(merged)}")


if __name__ == "__main__":
    main()
