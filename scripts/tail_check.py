"""Step 1: does the scoreline grid really under-predict 4+ goal games?

Walks 2025-26 exactly as the accuracy pass does — refit weekly, train only on
matches already played — but records the model's own probability that one side
scores 4 or more, so predicted can be set against actual on the same fixtures.

Split by favourite gap, because the complaint is specifically about mismatches:
if the model is fine in even games and short in lopsided ones, an across-the-
board change would be the wrong fix.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.db import connect
from models.backtest import load_matches, promoted_prior
from models.config import TRAIN_SEASONS, make_model

TARGET = "2526"


def p_four_plus(lam, mu):
    """P(at least one side scores 4+), from the same independent-Poisson
    marginals the grid is built from. The Dixon-Coles tau correction only
    touches scorelines of 1-1 and below, so it cannot affect this."""
    p_home_lt4 = poisson.cdf(3, lam)
    p_away_lt4 = poisson.cdf(3, mu)
    return 1 - p_home_lt4 * p_away_lt4


def main():
    conn = connect()
    df = load_matches(conn)
    df = df[df["season"].isin(TRAIN_SEASONS)]

    # 1. raw rate across the training data
    all4 = ((df["fthg"] >= 4) | (df["ftag"] >= 4)).mean()
    print(f"1. Real rate of a 4+ goal side, {len(df)} matches "
          f"({'/'.join(TRAIN_SEASONS)}): {all4:.4f}  ({100*all4:.1f}%)")
    for s in TRAIN_SEASONS:
        d = df[df["season"] == s]
        r = ((d["fthg"] >= 4) | (d["ftag"] >= 4)).mean()
        print(f"     {s}: {100*r:.1f}%")

    # 2. walk-forward predicted vs actual on the holdout
    target = df[df["season"] == TARGET].sort_values("date")
    rows, model, last_fit = [], None, None
    for _, r in target.iterrows():
        date = pd.Timestamp(r["date"])
        if model is None or (date - last_fit).days >= 7:
            train = df[pd.to_datetime(df["date"]) < date]
            if len(train) < 380:
                continue
            model = make_model().fit(train.to_dict("records"), as_of=date)
            last_fit = date
        for t in (r["home"], r["away"]):
            if t not in model.attack:
                model.attack[t], model.defence[t] = promoted_prior(model)
        lam, mu = model.expected_goals(r["home"], r["away"])
        rows.append({
            "lam": lam, "mu": mu, "gap": abs(lam - mu), "total_xg": lam + mu,
            "p4": p_four_plus(lam, mu),
            "actual4": int(r["fthg"] >= 4 or r["ftag"] >= 4),
            "goals": int(r["fthg"] + r["ftag"]),
        })
    bt = pd.DataFrame(rows)

    print(f"\n2. Holdout {TARGET}, {len(bt)} matches")
    print(f"   predicted P(4+ side): {bt.p4.mean():.4f}   "
          f"actual: {bt.actual4.mean():.4f}   "
          f"gap: {bt.actual4.mean() - bt.p4.mean():+.4f}")
    exp, act = bt.p4.sum(), bt.actual4.sum()
    print(f"   expected {exp:.1f} such matches, saw {act}")
    # is that gap bigger than noise? sd of the sum of independent Bernoullis
    sd = float(np.sqrt((bt.p4 * (1 - bt.p4)).sum()))
    print(f"   {(act - exp) / sd:+.2f} sd  (sd = {sd:.1f})")

    # 3. by favourite gap
    print(f"\n3. By favourite gap (|home xG - away xG|), terciles")
    bt["band"] = pd.qcut(bt["gap"], 3, labels=["even", "middling", "mismatch"])
    g = bt.groupby("band", observed=True).agg(
        n=("p4", "size"), mean_gap=("gap", "mean"),
        predicted=("p4", "mean"), actual=("actual4", "mean"))
    g["diff"] = g["actual"] - g["predicted"]
    print(g.round(4).to_string())

    print(f"\n   the same split, but on total expected goals")
    bt["tband"] = pd.qcut(bt["total_xg"], 3, labels=["low", "mid", "high"])
    g2 = bt.groupby("tband", observed=True).agg(
        n=("p4", "size"), mean_total_xg=("total_xg", "mean"),
        predicted=("p4", "mean"), actual=("actual4", "mean"))
    g2["diff"] = g2["actual"] - g2["predicted"]
    print(g2.round(4).to_string())

    # goals distribution sanity: is the whole total-goals curve short, or the tail?
    print(f"\n4. Total goals in a match — model vs actual")
    lam, mu = bt.lam.to_numpy(), bt.mu.to_numpy()
    for k in range(0, 8):
        pk = np.mean([poisson.pmf(np.arange(0, k + 1), l)
                      @ poisson.pmf(np.arange(k, -1, -1), m)
                      for l, m in zip(lam, mu)])
        ak = (bt.goals == k).mean()
        print(f"   {k} goals  predicted {pk:.4f}  actual {ak:.4f}  {ak-pk:+.4f}")
    hi_p = 1 - sum(np.mean([poisson.pmf(np.arange(0, k + 1), l)
                            @ poisson.pmf(np.arange(k, -1, -1), m)
                            for l, m in zip(lam, mu)]) for k in range(0, 8))
    hi_a = (bt.goals >= 8).mean()
    print(f"   8+ goals  predicted {hi_p:.4f}  actual {hi_a:.4f}  {hi_a-hi_p:+.4f}")

    bt.to_csv(ROOT / "data" / "backtests" / "tail_check.csv", index=False)


if __name__ == "__main__" and "--sweep" not in sys.argv:
    main()


def sweep():
    """Is the 4+ gap an artefact of how heavily recent matches are weighted?

    Re-runs the same measurement at several time-decay rates. xi=0.001 is what
    ships; its half-life is ln2/0.001 = 693 days, so a season-level shift in
    scoring takes roughly two years to be absorbed. Higher xi forgets faster.
    """
    conn = connect()
    df = load_matches(conn)
    df = df[df["season"].isin(TRAIN_SEASONS)]
    target = df[df["season"] == TARGET].sort_values("date")
    actual = ((target["fthg"] >= 4) | (target["ftag"] >= 4)).mean()

    print(f"actual 4+ rate on {TARGET}: {actual:.4f}\n")
    print(f"{'xi':>8} {'half-life':>10} {'predicted':>10} {'gap':>8} {'early':>8} {'late':>8}")
    for xi in (0.001, 0.002, 0.003, 0.005, 0.008):
        rows, model, last_fit = [], None, None
        for _, r in target.iterrows():
            date = pd.Timestamp(r["date"])
            if model is None or (date - last_fit).days >= 7:
                train = df[pd.to_datetime(df["date"]) < date]
                if len(train) < 380:
                    continue
                m = make_model()
                m.xi = xi
                model = m.fit(train.to_dict("records"), as_of=date)
                last_fit = date
            for t in (r["home"], r["away"]):
                if t not in model.attack:
                    model.attack[t], model.defence[t] = promoted_prior(model)
            lam, mu = model.expected_goals(r["home"], r["away"])
            rows.append({"p4": p_four_plus(lam, mu), "date": date})
        b = pd.DataFrame(rows)
        half = len(b) // 2
        print(f"{xi:>8} {0.693/xi:>9.0f}d {b.p4.mean():>10.4f} "
              f"{actual - b.p4.mean():>+8.4f} "
              f"{b.p4.iloc[:half].mean():>8.4f} {b.p4.iloc[half:].mean():>8.4f}")

    # what the CURRENT live model would say going into 2026-27
    live = make_model().fit(df.to_dict("records"))
    teams = sorted(live.attack)
    ps = []
    for h in teams:
        for a in teams:
            if h == a:
                continue
            lam, mu = live.expected_goals(h, a)
            ps.append(p_four_plus(lam, mu))
    print(f"\nlive model, every possible fixture: mean P(4+ side) = {np.mean(ps):.4f}")
    print(f"  vs 2025-26 actual {actual:.4f}, vs 4-season actual "
          f"{((df['fthg'] >= 4) | (df['ftag'] >= 4)).mean():.4f}")


if __name__ == "__main__" and "--sweep" in sys.argv:
    sweep()
