import argparse
from datetime import datetime

from src.ml.forecast_yearly_nn import main as run_nn


def main(epochs: int, lr: float, patience: int, device: str, model_version: str | None, notes: str | None):
    version = model_version or datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
    for i, h in enumerate(range(5, 11)):
        run_nn(
            horizon_years=h,
            retrain=(i == 0),
            epochs=epochs,
            lr=lr,
            patience=patience,
            device_arg=device,
            model_version=version,
            notes=notes,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--model-version", type=str, default=None, help="Version tag for the whole 5..10 run")
    parser.add_argument("--notes", type=str, default=None, help="Optional notes saved in run registry")
    args = parser.parse_args()
    main(
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        model_version=args.model_version,
        notes=args.notes,
    )
