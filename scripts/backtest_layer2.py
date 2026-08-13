"""Backtest Layer 1 + Layer 2 (style matchup adjustment) on 2025-26.

Style buckets and pair adjustments are refit at each model refit date using
only prior matches (same anti-leakage rule as everything else).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.db import connect
from models.backtest import load_matches, metrics, calibration_table, promoted_prior
from models.dixon_coles import DixonColes
from models.style import StyleAdjuster

OUT = ROOT / "data" / "backtests"
TARGET = "2526"


def main():
    conn = connect()
    df = load_matches(conn)
    tid = {r["name"]: r["id"] for r in conn.execute("SELECT id, name FROM teams")}
    team_names = {v: k for k, v in tid.items()}

    target = df[df["season"] == TARGET].sort_values("date")
    records = []
    model, adjuster, last_fit = None, None, None
    for _, row in target.iterrows():
        date = pd.Timestamp(row["date"])
        if model is None or (date - last_fit).days >= 7:
            train = df[pd.to_datetime(df["date"]) < date]
            model = DixonColes().fit(train.to_dict("records"), as_of=date)
            adjuster = StyleAdjuster(conn, model, date, team_names)
            last_fit = date
        for t in (row["home"], row["away"]):
            if t not in model.attack:
                model.attack[t], model.defence[t] = promoted_prior(model)
        lam, mu = model.expected_goals(row["home"], row["away"])
        lam, mu = adjuster.adjust(tid[row["home"]], tid[row["away"]], lam, mu)
        p_h, p_d, p_a = model.outcome_probs(lam, mu)
        probs = np.array([p_h, p_d, p_a])
        records.append({
            "date": row["date"], "home": row["home"], "away": row["away"],
            "p_home": p_h, "p_draw": p_d, "p_away": p_a,
            "actual": {"H": 0, "D": 1, "A": 2}[row["ftr"]],
            "pred": int(np.argmax(probs)), "provisional_rating": False,
        })
    bt = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    bt.to_csv(OUT / "layer2_style_2526.csv", index=False)

    m = metrics(bt)
    print("Layer 1 + Layer 2 (style) — 2025-26 backtest")
    print(f"  accuracy: {m['accuracy']:.3f}")
    print(f"  brier:    {m['brier']:.4f}")
    print(f"  logloss:  {m['logloss']:.4f}")
    print("\ncalibration (home-win prob):")
    print(calibration_table(bt).to_string(index=False))


if __name__ == "__main__":
    main()
