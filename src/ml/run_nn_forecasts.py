from src.ml.forecast_yearly_nn import main as run_nn


def main():
    for h in range(5, 11):
        run_nn(horizon_years=h, retrain=False)


if __name__ == "__main__":
    main()
