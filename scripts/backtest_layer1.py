"""Backtest Layer 1 (season-average Dixon-Coles, no lineup weighting)
on the most recent complete season, 2025-26.

Writes results CSV + calibration table to data/backtests/.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.db import connect
from models.backtest import load_matches, run_backtest, metrics, calibration_table
from models.config import TRAIN_SEASONS, make_model

OUT = ROOT / "data" / "backtests"


def main():
    conn = connect()
    df = load_matches(conn)
    df = df[df["season"].isin(TRAIN_SEASONS)]
    print(f"training pool: {len(df)} matches, seasons {sorted(df.season.unique())}")
    bt = run_backtest(df, target_season="2526", make_model=make_model)
    OUT.mkdir(parents=True, exist_ok=True)
    bt.to_csv(OUT / "layer1_season_avg_2526.csv", index=False)

    m = metrics(bt)
    print("\nLayer 1 — season-average ratings, 2025-26 backtest")
    print(f"  matches:  {m['n']}  (provisional-rated: {int(bt.provisional_rating.sum())})")
    print(f"  accuracy: {m['accuracy']:.3f}")
    print(f"  brier:    {m['brier']:.4f}   (uniform = 0.667)")
    print(f"  logloss:  {m['logloss']:.4f} (uniform = 1.099)")
    print("\ncalibration (home-win prob):")
    print(calibration_table(bt).to_string(index=False))


if __name__ == "__main__":
    main()
