"""Accuracy pass: score model variants against the 2025-26 holdout.

Every variant is the SAME walk-forward harness as the existing Layer 1/2
backtests (refit weekly, train only on matches strictly before the fixture
date), so the numbers are comparable to what is already on /performance.

Variants:
  baseline        goals target, xi=0.001, 4 training seasons  (what ships now)
  shrink-*        Gaussian prior on log attack/defence
  xg / blend-*    xG or goals-xG blended fitting target        (4a)
  xi-*            time-decay grid                               (4b)
  seasons-7       three extra historical seasons                (4e)
  market          closing-odds implied probabilities alone      (4d reference)
  blend-market    50/50 model + market                          (4d)

4c (draw calibration) is a diagnostic on the chosen frame, not a variant, and
is printed separately.

Results land in data/backtests/accuracy_pass.csv plus per-variant frames, so
nothing here has to be recomputed to be inspected later.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.db import connect
from models.backtest import load_matches, run_backtest, metrics
from models.dixon_coles import DixonColes
from models.evaluate import (PCOLS, market_probs, blend, with_probs,
                             calibration, outcome_summary)

OUT = ROOT / "data" / "backtests"
TARGET = "2526"
FOUR = ["2223", "2324", "2425", "2526"]
SEVEN = ["1920", "2021", "2122"] + FOUR


def factory(**kw):
    def make():
        m = DixonColes()
        for k, v in kw.items():
            setattr(m, k, v)
        return m
    return make


# label -> (season list, DixonColes field overrides)
VARIANTS = [
    ("baseline (goals, xi=0.001, 4 seasons)", FOUR, {}),
    # --- rating shrinkage (found while verifying the baseline) ---
    ("shrink-1  (prior=1)", FOUR, dict(prior_strength=1.0)),
    ("shrink-5  (prior=5)", FOUR, dict(prior_strength=5.0)),
    ("shrink-20 (prior=20)", FOUR, dict(prior_strength=20.0)),
    # --- 4a: xG as fitting target ---
    ("4a xg-only        (blend_w=0.0)", FOUR, dict(blend_w=0.0)),
    ("4a blend-25/75    (blend_w=0.25)", FOUR, dict(blend_w=0.25)),
    ("4a blend-50/50    (blend_w=0.5)", FOUR, dict(blend_w=0.5)),
    ("4a blend-75/25    (blend_w=0.75)", FOUR, dict(blend_w=0.75)),
    # --- 4b: time decay ---
    ("4b xi=0.0003", FOUR, dict(xi=0.0003)),
    ("4b xi=0.0005", FOUR, dict(xi=0.0005)),
    ("4b xi=0.0018", FOUR, dict(xi=0.0018)),
    ("4b xi=0.0030", FOUR, dict(xi=0.0030)),
    ("4b xi=0.0050", FOUR, dict(xi=0.0050)),
    # --- 4e: more history ---
    ("4e seasons-7 (goals)", SEVEN, {}),
    # --- combinations: shrinkage and the xG target fix different faults
    # (unidentifiable low-data teams vs noisy scorelines), so they are tested
    # together. xi is re-swept on top, because the first xi sweep was measured
    # with the degenerate-rating pathology still in the frame.
    ("combo xg+shrink5", FOUR, dict(blend_w=0.0, prior_strength=5.0)),
    ("combo xg+shrink20", FOUR, dict(blend_w=0.0, prior_strength=20.0)),
    ("combo xg+shrink5, xi=0.0005", FOUR,
     dict(blend_w=0.0, prior_strength=5.0, xi=0.0005)),
    ("combo xg+shrink5, xi=0.0018", FOUR,
     dict(blend_w=0.0, prior_strength=5.0, xi=0.0018)),
    ("combo xg+shrink5, xi=0.0030", FOUR,
     dict(blend_w=0.0, prior_strength=5.0, xi=0.0030)),
    ("combo xg+shrink5, 7 seasons", SEVEN,
     dict(blend_w=0.0, prior_strength=5.0)),
]


def score(label, bt):
    m = metrics(bt)
    return {"variant": label, "n": m["n"],
            "accuracy": round(m["accuracy"], 4),
            "brier": round(m["brier"], 4),
            "logloss": round(m["logloss"], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="substring filter on variant label")
    args = ap.parse_args()

    conn = connect()
    df = load_matches(conn)
    OUT.mkdir(parents=True, exist_ok=True)

    rows, frames = [], {}
    for label, seasons, kw in VARIANTS:
        if args.only and not any(s in label for s in args.only):
            continue
        sub = df[df["season"].isin(seasons)]
        bt = run_backtest(sub, TARGET, make_model=factory(**kw))
        frames[label] = bt
        rows.append(score(label, bt))
        print(f"  {rows[-1]}", flush=True)

    # --- 4d: market baseline and model+market blend, on the best model frame ---
    if frames:
        best = min(rows, key=lambda r: r["logloss"])["variant"]
        bt = frames[best]
        priced = bt.dropna(subset=["odds_home", "odds_draw", "odds_away"])
        if len(priced):
            mp = market_probs(priced)
            rows.append(score(f"4d market only (closing odds)",
                              with_probs(priced, mp)))
            print(f"  {rows[-1]}", flush=True)
            for w in (0.75, 0.5, 0.25):
                bl = blend(priced[PCOLS].to_numpy(float), mp, w)
                rows.append(score(
                    f"4d blend {int(w*100)}/{int((1-w)*100)} model/market "
                    f"[on {best}]", with_probs(priced, bl)))
                print(f"  {rows[-1]}", flush=True)
            frames["4d market only"] = with_probs(priced, mp)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "accuracy_pass.csv", index=False)
    print("\n=== accuracy pass, 2025-26 holdout ===")
    print(table.to_string(index=False))

    # --- 4c: draw calibration, baseline vs best ---
    for label in dict.fromkeys(["baseline (goals, xi=0.001, 4 seasons)",
                                min(rows, key=lambda r: r["logloss"])["variant"]]):
        if label not in frames:
            continue
        print(f"\n--- outcome summary: {label} ---")
        print(outcome_summary(frames[label]).to_string(index=False))
        print(f"--- draw calibration: {label} ---")
        print(calibration(frames[label], "draw").to_string(index=False))

    for label, f in frames.items():
        slug = "".join(c if c.isalnum() else "_" for c in label)[:50]
        f.to_csv(OUT / f"ap_{slug}.csv", index=False)


if __name__ == "__main__":
    main()
