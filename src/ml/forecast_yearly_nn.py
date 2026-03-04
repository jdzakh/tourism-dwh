import argparse
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sqlalchemy import text
from torch import nn

from src.ml.model_registry import register_training_run, save_metric_rows
from src.utils.db import get_engine

MODEL_NAME = "NN_MLP_GLOBAL"
WINDOW = 3
MODEL_PATH = "models/nn_mlp_global.pt"
MODEL_VERSIONS_DIR = "models/versions"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_continuous_series(ts: pd.DataFrame) -> pd.Series:
    ts = ts.sort_values("year")
    year_min = int(ts["year"].min())
    year_max = int(ts["year"].max())
    idx = pd.Index(range(year_min, year_max + 1), name="year")
    s = ts.set_index("year")["tourists_cnt"].reindex(idx)
    s = s.interpolate(limit_direction="both")
    s = s.clip(lower=0)
    return s


def make_windows(series: np.ndarray, window: int):
    xs, ys = [], []
    for i in range(window, len(series)):
        xs.append(series[i - window:i])
        ys.append(series[i])
    if not xs:
        return None, None
    return np.stack(xs), np.array(ys)


class MLP(nn.Module):
    def __init__(self, window: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_global_model(
    all_windows: list,
    all_targets: list,
    device: torch.device,
    epochs: int = 400,
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.15,
    patience: int = 35,
):
    x_np = np.concatenate(all_windows)
    y_np = np.concatenate(all_targets)

    n = len(x_np)
    idx = np.arange(n)
    np.random.shuffle(idx)
    val_n = max(1, int(n * val_ratio))
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]
    if len(tr_idx) == 0:
        tr_idx = idx
        val_idx = idx

    x_tr = torch.tensor(x_np[tr_idx], dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_np[tr_idx], dtype=torch.float32).to(device)
    x_val = torch.tensor(x_np[val_idx], dtype=torch.float32).to(device)
    y_val = torch.tensor(y_np[val_idx], dtype=torch.float32).to(device)

    model = MLP(window=x_tr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_tr)
        loss = loss_fn(pred, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = float(loss_fn(val_pred, y_val).item())

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

        if (epoch + 1) % 50 == 0:
            print(f"epoch={epoch+1} train={float(loss.item()):.5f} val={val_loss:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def load_model_if_exists(device: torch.device):
    if not os.path.exists(MODEL_PATH):
        return None
    payload = torch.load(MODEL_PATH, map_location=device)
    window = payload.get("window", WINDOW)
    if window != WINDOW:
        return None
    model = MLP(window=WINDOW).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def predict_in_sample(model: MLP, series_log: np.ndarray, device: torch.device):
    preds_in = []
    model.eval()
    with torch.no_grad():
        for i in range(WINDOW, len(series_log)):
            window = torch.tensor(series_log[i - WINDOW:i], dtype=torch.float32).to(device)
            preds_in.append(float(model(window).item()))
    return preds_in


def predict_future(model: MLP, series_log: np.ndarray, horizon: int, device: torch.device):
    preds_out = []
    history = list(series_log)
    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            window = torch.tensor(history[-WINDOW:], dtype=torch.float32).to(device)
            pred = float(model(window).item())
            history.append(pred)
            preds_out.append(pred)
    return preds_out


def to_level(values_log):
    arr = np.expm1(np.clip(values_log, -50, 50))
    return np.maximum(arr, 0)


def calc_metrics_frame(df: pd.DataFrame, fact_col: str, pred_col: str, horizon_years: int) -> pd.DataFrame:
    x = df[[fact_col, pred_col]].dropna().copy()
    if x.empty:
        return pd.DataFrame(columns=["horizon_years", "n_points", "mae", "rmse", "mape", "smape", "wape"])
    err = x[fact_col] - x[pred_col]
    mape_den = x[fact_col].replace(0, np.nan)
    smape_den = (x[fact_col].abs() + x[pred_col].abs()).replace(0, np.nan)
    wape_den = float(x[fact_col].abs().sum())
    row = {
        "horizon_years": int(horizon_years),
        "n_points": int(len(x)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err.pow(2)).mean())),
        "mape": float((err.abs() / mape_den).dropna().mean() * 100) if mape_den.notna().any() else np.nan,
        "smape": float((200.0 * err.abs() / smape_den).dropna().mean()) if smape_den.notna().any() else np.nan,
        "wape": float(err.abs().sum() / wape_den * 100) if wape_den > 0 else np.nan,
    }
    return pd.DataFrame([row])


def apply_recovery_scenario(years_hist, actual_in, yhat_out):
    if not years_hist or len(actual_in) < 4 or len(yhat_out) == 0:
        return yhat_out
    pre = [v for y, v in zip(years_hist, actual_in) if y <= 2019]
    if len(pre) < 3:
        return yhat_out
    baseline = float(np.median(pre[-3:]))
    last_fact = float(actual_in[-1])
    if baseline <= 0:
        return yhat_out
    if last_fact >= 0.6 * baseline:
        return yhat_out
    recovery_years = min(3, len(yhat_out))
    adjusted = []
    for i, pred in enumerate(yhat_out):
        if i < recovery_years:
            target = last_fact + (baseline - last_fact) * ((i + 1) / recovery_years)
        else:
            target = baseline
        adjusted.append(max(pred, target))
    return np.array(adjusted)


def main(
    horizon_years: int = 5,
    retrain: bool = False,
    epochs: int = 400,
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    patience: int = 35,
    seed: int = 42,
    device_arg: str = "auto",
    model_version: str | None = None,
    notes: str | None = None,
):
    set_seed(seed)
    device = choose_device(device_arg)
    print(f"Using device: {device}")

    engine = get_engine()

    fact = pd.read_sql(
        """
        SELECT d.destination_id, d.country, dd.year, SUM(ff.tourists_cnt) AS tourists_cnt
        FROM dwh.fact_flow ff
        JOIN dwh.dim_date dd ON dd.date_id = ff.date_id
        JOIN dwh.dim_destination d ON d.destination_id = ff.destination_id
        GROUP BY d.destination_id, d.country, dd.year
        ORDER BY d.destination_id, dd.year
        """,
        engine,
    )

    all_windows, all_targets = [], []
    series_by_dest = {}

    for dest_id, group in fact.groupby("destination_id"):
        ts = group[["year", "tourists_cnt"]].dropna()
        if ts.shape[0] < WINDOW + 2:
            continue
        s = build_continuous_series(ts)
        s_log = np.log1p(s.values)
        x, y = make_windows(s_log, WINDOW)
        if x is None:
            continue
        all_windows.append(x)
        all_targets.append(y)
        series_by_dest[dest_id] = (s.index.tolist(), s_log)

    if not all_windows:
        raise RuntimeError("Not enough data to train model.")

    if model_version is None:
        model_version = datetime.utcnow().strftime("v%Y%m%d_%H%M%S")

    model = None if retrain else load_model_if_exists(device=device)
    trained_now = False
    version_path = os.path.join(MODEL_VERSIONS_DIR, f"nn_mlp_global_{model_version}.pt")
    if model is None:
        model = train_global_model(
            all_windows=all_windows,
            all_targets=all_targets,
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
        )
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        os.makedirs(MODEL_VERSIONS_DIR, exist_ok=True)
        payload = {
            "state_dict": model.cpu().state_dict(),
            "window": WINDOW,
            "model_name": MODEL_NAME,
            "model_version": model_version,
            "trained_at": datetime.utcnow().isoformat(),
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "patience": patience,
            "seed": seed,
            "device": str(device),
        }
        torch.save(
            payload,
            version_path,
        )
        # Keep latest pointer for quick loading.
        torch.save(
            {
                **payload,
                "model_version": "latest",
            },
            MODEL_PATH,
        )
        model = model.to(device)
        trained_now = True

    rows = []
    generated_at = datetime.utcnow()

    for dest_id, (years_hist, series_log) in series_by_dest.items():
        preds_in = predict_in_sample(model, series_log, device=device)
        preds_out = predict_future(model, series_log, horizon_years, device=device)
        years_pred_in = years_hist
        years_pred_out = list(range(years_hist[-1] + 1, years_hist[-1] + 1 + horizon_years))

        actual_in = to_level(np.array(series_log))
        yhat_in = np.concatenate([actual_in[:WINDOW], to_level(np.array(preds_in))])
        yhat_out = to_level(np.array(preds_out))
        yhat_out = apply_recovery_scenario(years_hist, actual_in, yhat_out)
        yhat_all = np.concatenate([yhat_in, yhat_out])
        years_all = years_pred_in + years_pred_out

        for year, yhat in zip(years_all, yhat_all):
            rows.append(
                {
                    "generated_at": generated_at,
                    "year": int(year),
                    "month": 1,
                    "destination_id": int(dest_id),
                    "yhat": float(yhat),
                    "yhat_lower": float(yhat * 0.9),
                    "yhat_upper": float(yhat * 1.1),
                    "model_name": MODEL_NAME,
                    "horizon_months": horizon_years * 12,
                }
            )

    total = fact.groupby("year", as_index=False)["tourists_cnt"].sum().dropna()
    if total.shape[0] >= WINDOW + 2:
        s_total = build_continuous_series(total)
        s_log = np.log1p(s_total.values)
        preds_in = predict_in_sample(model, s_log, device=device)
        preds_out = predict_future(model, s_log, horizon_years, device=device)
        years_pred_in = s_total.index.tolist()
        years_pred_out = list(range(s_total.index.tolist()[-1] + 1, s_total.index.tolist()[-1] + 1 + horizon_years))

        actual_in = to_level(np.array(s_log))
        yhat_in = np.concatenate([actual_in[:WINDOW], to_level(np.array(preds_in))])
        yhat_out = to_level(np.array(preds_out))
        yhat_out = apply_recovery_scenario(s_total.index.tolist(), actual_in, yhat_out)
        yhat_all = np.concatenate([yhat_in, yhat_out])
        years_all = years_pred_in + years_pred_out

        for year, yhat in zip(years_all, yhat_all):
            rows.append(
                {
                    "generated_at": generated_at,
                    "year": int(year),
                    "month": 1,
                    "destination_id": None,
                    "yhat": float(yhat),
                    "yhat_lower": float(yhat * 0.9),
                    "yhat_upper": float(yhat * 1.1),
                    "model_name": MODEL_NAME,
                    "horizon_months": horizon_years * 12,
                }
            )

    out = pd.DataFrame(rows)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM dwh.fact_forecast_monthly
                WHERE model_name = :m AND horizon_months = :h
                """
            ),
            {"m": MODEL_NAME, "h": horizon_years * 12},
        )

    out.to_sql("fact_forecast_monthly", engine, schema="dwh", if_exists="append", index=False)
    print(f"Inserted NN forecasts: {len(out)} rows, horizon={horizon_years} years")

    # Persist run metadata only when a new model was trained.
    if trained_now:
        register_training_run(
            engine=engine,
            model_name=MODEL_NAME,
            model_version=model_version,
            device=str(device),
            model_path=version_path,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            seed=seed,
            notes=notes,
            params={"window": WINDOW},
        )

    # Save in-sample total metrics for version comparison.
    total_pairs = out[out["destination_id"].isna()][["year", "yhat"]].rename(columns={"yhat": "pred"})
    fact_total = (
        fact.groupby("year", as_index=False)["tourists_cnt"]
        .sum()
        .rename(columns={"tourists_cnt": "fact"})
    )
    paired_total = fact_total.merge(total_pairs, on="year", how="inner")
    metric_df = calc_metrics_frame(paired_total, fact_col="fact", pred_col="pred", horizon_years=horizon_years)
    save_metric_rows(
        engine=engine,
        model_name=MODEL_NAME,
        model_version=model_version,
        eval_type="in_sample_total",
        metrics_df=metric_df,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in years")
    parser.add_argument("--retrain", action="store_true", help="Force retraining model")
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=35, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device",
    )
    parser.add_argument("--model-version", type=str, default=None, help="Model version tag, e.g. v20260301_exp1")
    parser.add_argument("--notes", type=str, default=None, help="Optional notes for run registry")
    args = parser.parse_args()
    main(
        horizon_years=args.horizon,
        retrain=args.retrain,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        device_arg=args.device,
        model_version=args.model_version,
        notes=args.notes,
    )
