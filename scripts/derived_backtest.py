"""Do the grid-derived match stats actually hold up? (Tier 1)

Total-goals and both-teams-score probabilities are sums over the same
scoreline grid the result probabilities come from, so they are free — but free
is not the same as good. This scores them on the 2025-26 holdout with the same
walk-forward discipline as the main accuracy pass: refit weekly, train only on
matches already played, never shuffled.

Both are binary, so the reference points differ from the three-way result
model. Reported per stat:

  base rate     how often it actually happened
  accuracy      share called correctly at a 0.5 threshold
  base-rate acc what always guessing the majority class would score
  Brier         mean (p - outcome)^2   (lower better)
  log-loss      mean -log p(actual)    (lower better)
  vs base rate  the same two scores for a model that always predicts the
                base rate — the honest floor, since a stat that happens 52%
                of the time is nearly a coin flip and any model must beat
                predicting 52% every week to have earned anything
  calibration   predicted vs actual by probability bucket

  .venv\\Scripts\\python scripts\\derived_backtest.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.db import connect
from models.backtest import load_matches, promoted_prior
from models.config import TRAIN_SEASONS, make_model
from models.derived import both_teams_score, goals_over

TARGET = "2526"
LINE = 2.5


def walk(df, target_season, refit_every=7, min_train=380):
    """Same walk-forward contract as models/backtest.run_backtest, but keeping
    the grid-derived quantities rather than the W/D/L probabilities."""
    target = df[df["season"] == target_season].sort_values("date")
    rows, model, last_fit = [], None, None
    for _, r in target.iterrows():
        date = pd.Timestamp(r["date"])
        if model is None or (date - last_fit).days >= refit_every:
            train = df[pd.to_datetime(df["date"]) < date]
            if len(train) < min_train:
                continue
            model = make_model().fit(train.to_dict("records"), as_of=date)
            last_fit = date
        for t in (r["home"], r["away"]):
            if t not in model.attack:
                model.attack[t], model.defence[t] = promoted_prior(model)
        lam, mu = model.expected_goals(r["home"], r["away"])
        grid = model.score_grid(lam, mu)
        rows.append({
            "date": r["date"], "home": r["home"], "away": r["away"],
            "p_over": goals_over(grid, LINE),
            "p_btts": both_teams_score(grid),
            "actual_over": int(r["fthg"] + r["ftag"] > LINE),
            "actual_btts": int(r["fthg"] >= 1 and r["ftag"] >= 1),
            "goals": int(r["fthg"] + r["ftag"]),
        })
    return pd.DataFrame(rows)


def binary_metrics(p, y):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, int)
    return {
        "n": len(y),
        "base_rate": float(y.mean()),
        "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
        "majority_acc": float(max(y.mean(), 1 - y.mean())),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "mean_pred": float(p.mean()),
    }


def base_rate_reference(y):
    """What a model that always predicts the base rate would score. Uses the
    in-sample base rate, which flatters it — deliberately, so beating it means
    something."""
    y = np.asarray(y, int)
    b = float(y.mean())
    p = np.full(len(y), b)
    return {"brier": float(np.mean((p - y) ** 2)),
            "logloss": float(-np.mean(y * np.log(b) + (1 - y) * np.log(1 - b)))}


def calibration(p, y, bins=6):
    p, y = np.asarray(p, float), np.asarray(y, int)
    edges = np.linspace(p.min(), p.max() + 1e-9, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() >= 5:
            out.append({"bucket": f"{edges[i]:.2f}-{edges[i+1]:.2f}",
                        "n": int(m.sum()), "predicted": float(p[m].mean()),
                        "actual": float(y[m].mean())})
    return pd.DataFrame(out)


def report(name, p, y):
    m = binary_metrics(p, y)
    ref = base_rate_reference(y)
    print(f"\n=== {name}  (n={m['n']})")
    print(f"  happened            {m['base_rate']:.1%} of matches")
    print(f"  mean prediction     {m['mean_pred']:.1%}"
          f"   (gap {m['mean_pred'] - m['base_rate']:+.1%})")
    print(f"  accuracy @0.5       {m['accuracy']:.1%}"
          f"   vs always-majority {m['majority_acc']:.1%}"
          f"   ({m['accuracy'] - m['majority_acc']:+.1%})")
    print(f"  Brier               {m['brier']:.4f}"
          f"   vs base-rate model {ref['brier']:.4f}"
          f"   ({m['brier'] - ref['brier']:+.4f})")
    print(f"  log-loss            {m['logloss']:.4f}"
          f"   vs base-rate model {ref['logloss']:.4f}"
          f"   ({m['logloss'] - ref['logloss']:+.4f})")
    cal = calibration(p, y)
    if not cal.empty:
        print("  calibration:")
        for _, c in cal.iterrows():
            flag = "" if abs(c["predicted"] - c["actual"]) < 0.06 else "   <-- off"
            print(f"    {c['bucket']}  n={int(c['n']):>3}  "
                  f"predicted {c['predicted']:.1%}  actual {c['actual']:.1%}{flag}")
    return m, ref


def main():
    conn = connect()
    df = load_matches(conn)
    df = df[df["season"].isin(TRAIN_SEASONS)]
    bt = walk(df, TARGET)
    if bt.empty:
        print("no backtest rows — check the match data")
        return

    print(f"Tier 1 derived stats, walk-forward on {TARGET}")
    print(f"Line: {LINE} goals. Refit weekly, trained only on prior matches.")
    report(f"More than {LINE} goals", bt.p_over, bt.actual_over)
    report("Both teams score", bt.p_btts, bt.actual_btts)

    # 2025-26 was an unusually low-scoring season (see scripts/tail_check.py),
    # so a systematic over-prediction here is expected and worth separating
    # from any flaw in the derivation itself.
    print(f"\n=== context")
    print(f"  mean goals per match, {TARGET}: {bt.goals.mean():.2f}")
    prior = df[df["season"] != TARGET]
    print(f"  mean goals per match, training seasons: "
          f"{(prior.fthg + prior.ftag).mean():.2f}")
    over_prior = ((prior.fthg + prior.ftag) > LINE).mean()
    btts_prior = ((prior.fthg >= 1) & (prior.ftag >= 1)).mean()
    print(f"  >{LINE} goals: {bt.actual_over.mean():.1%} in {TARGET} vs "
          f"{over_prior:.1%} across the training seasons")
    print(f"  both score: {bt.actual_btts.mean():.1%} in {TARGET} vs "
          f"{btts_prior:.1%} across the training seasons")

    out = ROOT / "data" / "backtests" / "derived_tier1.csv"
    bt.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
