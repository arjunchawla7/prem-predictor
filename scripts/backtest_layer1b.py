"""Backtest Layer 1 + lineup weighting + fatigue + travel/congestion on
2025-26, using the ACTUAL starting XIs (as a confirmed-lineup prediction
would have known them at kickoff). Player ratings come from 2024-25 data
only — no target-season leakage.

Compares against the season-average baseline from backtest_layer1.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.db import connect
from models.backtest import load_matches, run_backtest, metrics, calibration_table
from models.player_ratings import RatingBook
from models.lineup import lineup_multiplier
from models.travel import travel_multiplier, rest_days, congestion_multipliers

OUT = ROOT / "data" / "backtests"
TARGET, RATING_SOURCE = "2526", "2425"


def main():
    conn = connect()
    df = load_matches(conn)
    book = RatingBook(conn, RATING_SOURCE)
    tid = {r["name"]: r["id"] for r in conn.execute("SELECT id, name FROM teams")}
    coords = {r["id"]: (r["lat"], r["lon"])
              for r in conn.execute("SELECT id, lat, lon FROM teams")}

    no_lineup = [0]

    def multipliers(row, model):
        h, a = tid[row["home"]], tid[row["away"]]
        date = str(row["date"])
        lam_mult = mu_mult = 1.0
        for team, is_home in ((h, True), (a, False)):
            starters = [r["player_id"] for r in conn.execute(
                """SELECT pmm.player_id FROM player_match_minutes pmm
                   WHERE pmm.match_id=? AND pmm.team_id=? AND pmm.started=1""",
                (row["id"], team))]
            if len(starters) != 11:
                no_lineup[0] += 1
                continue
            mult, _ = lineup_multiplier(conn, book, team, starters, date, TARGET)
            if is_home:
                lam_mult *= mult
            else:
                mu_mult *= mult
        tm, _, _ = travel_multiplier(coords[a], coords[h])
        mu_mult *= tm
        hm, am, _ = congestion_multipliers(rest_days(conn, h, date),
                                           rest_days(conn, a, date))
        return lam_mult * hm, mu_mult * am

    bt = run_backtest(df, TARGET, xg_multipliers=multipliers)
    OUT.mkdir(parents=True, exist_ok=True)
    bt.to_csv(OUT / "layer1_lineup_weighted_2526.csv", index=False)

    m = metrics(bt)
    print("Layer 1 + lineup/fatigue/travel — 2025-26 backtest (actual XIs)")
    print(f"  matches:  {m['n']}  (team-sides without full lineup: {no_lineup[0]})")
    print(f"  accuracy: {m['accuracy']:.3f}")
    print(f"  brier:    {m['brier']:.4f}")
    print(f"  logloss:  {m['logloss']:.4f}")
    print("\ncalibration (home-win prob):")
    print(calibration_table(bt).to_string(index=False))


if __name__ == "__main__":
    main()
