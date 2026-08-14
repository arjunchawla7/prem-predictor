"""Does Championship history beat the weakest-3 prior for promoted teams?

A promoted side has no top-flight matches, so the model currently rates it as
the average of the three weakest fitted teams — a prior that knows nothing
about the specific club. Two seasons of second-tier results are now loaded
(division='E1'), so the alternative is to rate it from those.

The catch is scale. Championship ratings are estimated against Championship
opposition, so they cannot be read straight onto the top-flight scale — a side
that scores freely against weak opposition would look like a good Premier
League team. The bridge is the clubs that appear in BOTH divisions inside the
window (Burnley, Leeds and Sunderland went up; Ipswich, Leicester and
Southampton came down). Fitting one pooled model over E0+E1 puts every team on
a single scale, anchored by those clubs, and the pooled ratings are then
shifted so the shared clubs line up with the top-flight-only fit.

Design choice worth being explicit about: the pooled fit is used ONLY to seed
teams the top-flight fit has never seen. Ratings for established clubs still
come from the validated E0-only model, so this cannot quietly move predictions
for the other seventeen teams.

Scored two ways — over the whole season, and over just the matches involving a
promoted side, which is the only place this can possibly make a difference and
where a whole-season number would drown it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.db import connect
from models.backtest import load_matches, run_backtest, metrics, promoted_prior
from models.config import TRAIN_SEASONS, make_model
from models.evaluate import PCOLS

OUT = ROOT / "data" / "backtests"
TARGET = "2526"


def aligned_pooled_fit(pooled_df, as_of, e0_model):
    """Fit over E0+E1, then shift onto the E0-only model's scale.

    The shift is the mean log difference across clubs present in both fits, so
    the pooled ratings are expressed in the same units as the model actually
    making predictions.
    """
    train = pooled_df[pd.to_datetime(pooled_df["date"]) < as_of]
    if train.empty:
        return None
    pooled = make_model().fit(train.to_dict("records"), as_of=as_of)

    shared = [t for t in pooled.attack if t in e0_model.attack]
    if len(shared) < 5:
        return None
    da = np.mean([np.log(e0_model.attack[t]) - np.log(pooled.attack[t])
                  for t in shared])
    dd = np.mean([np.log(e0_model.defence[t]) - np.log(pooled.defence[t])
                  for t in shared])
    pooled.attack = {t: float(v * np.exp(da)) for t, v in pooled.attack.items()}
    pooled.defence = {t: float(v * np.exp(dd)) for t, v in pooled.defence.items()}
    return pooled


def main():
    conn = connect()
    e0 = load_matches(conn)
    e0 = e0[e0["season"].isin(TRAIN_SEASONS)]
    pooled_df = load_matches(conn, ("E0", "E1"))
    pooled_df = pooled_df[pooled_df["season"].isin(TRAIN_SEASONS + ["2425"])]
    OUT.mkdir(parents=True, exist_ok=True)

    # who is promoted into the target season (no prior top-flight match)
    before = e0[e0["season"] != TARGET]
    seen = set(before["home"]) | set(before["away"])
    tgt = e0[e0["season"] == TARGET]
    promoted = sorted((set(tgt["home"]) | set(tgt["away"])) - seen)
    print(f"promoted into {TARGET}: {promoted}")

    state = {"pooled": None}

    def on_refit(model, date):
        state["pooled"] = aligned_pooled_fit(pooled_df, date, model)

    used = {}

    def cross_league_prior(model, team):
        p = state["pooled"]
        if p is not None and team in p.attack:
            used[team] = (round(p.attack[team], 3), round(p.defence[team], 3))
            return p.attack[team], p.defence[team]
        return promoted_prior(model)

    rows, frames = [], {}
    for label, pf in (("weakest-3 prior (current)", None),
                      ("championship prior", cross_league_prior)):
        bt = run_backtest(e0, TARGET, make_model=make_model, prior_fn=pf,
                          on_refit=on_refit if pf else None)
        frames[label] = bt
        m = metrics(bt)
        sub = bt[bt["home"].isin(promoted) | bt["away"].isin(promoted)]
        ms = metrics(sub)
        rows.append({"variant": label, "n": m["n"],
                     "accuracy": round(m["accuracy"], 4),
                     "brier": round(m["brier"], 4),
                     "logloss": round(m["logloss"], 4),
                     "promoted_n": ms["n"],
                     "promoted_acc": round(ms["accuracy"], 4),
                     "promoted_brier": round(ms["brier"], 4),
                     "promoted_logloss": round(ms["logloss"], 4)})
        print(f"  {rows[-1]}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "promoted_prior_pass.csv", index=False)
    print("\n=== promoted-team prior, 2025-26 holdout ===")
    print(table.to_string(index=False))
    print("\nchampionship-derived ratings actually used:")
    for t, v in sorted(used.items()):
        print(f"  {t:<20} attack={v[0]:<7} defence={v[1]}")
    a, b = frames["weakest-3 prior (current)"], frames["championship prior"]
    diff = (a[PCOLS].to_numpy() - b[PCOLS].to_numpy())
    print(f"\nmatches whose probabilities changed at all: "
          f"{int((np.abs(diff).max(axis=1) > 1e-6).sum())} of {len(a)}")
    for label, f in frames.items():
        slug = "".join(c if c.isalnum() else "_" for c in label)[:40]
        f.to_csv(OUT / f"prior_{slug}.csv", index=False)


if __name__ == "__main__":
    main()
