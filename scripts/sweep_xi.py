"""Small hyperparameter sweep: time-decay xi, scored on the 2025-26
walk-forward backtest. Selection metric = log-loss (calibration-sensitive)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.db import connect
import models.backtest as B
from models import dixon_coles


def main():
    conn = connect()
    df = B.load_matches(conn)
    for xi in [0.0005, 0.001, 0.0018, 0.003, 0.005]:
        orig_fit = dixon_coles.DixonColes.fit

        def fit_with_xi(self, matches, as_of=None, _xi=xi):
            self.xi = _xi
            return orig_fit(self, matches, as_of=as_of)

        dixon_coles.DixonColes.fit = fit_with_xi
        bt = B.run_backtest(df, "2526")
        m = B.metrics(bt)
        dixon_coles.DixonColes.fit = orig_fit
        print(f"xi={xi:<7} acc={m['accuracy']:.3f} brier={m['brier']:.4f} "
              f"logloss={m['logloss']:.4f}")


if __name__ == "__main__":
    main()
