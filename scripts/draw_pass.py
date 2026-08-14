"""Draw-calibration pass: does fixing the draw shortfall actually pay?

The model under-produces draws — mean predicted .234 against an actual .274 on
2025-26 — and a draw is never its argmax pick. Two candidate fixes, scored the
same way as everything else:

  boost   an extra parameter on the drawn cells of the score grid, fitted by
          maximum likelihood on the training window at each walk-forward refit
  scale   post-hoc vector scaling of the three probabilities, fitted on
          EARLIER SEASONS' out-of-sample predictions only

Both are strictly leak-free: `boost` never sees the target season, and `scale`
is fitted on 2023-24 + 2024-25 walk-forward output before being applied to
2025-26.

The honest question this answers is not "can accuracy be pushed up" — inflating
draws far enough does raise the hit-rate for a while — but "does the model
become better calibrated", which is what Brier and log-loss measure.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.db import connect
from models.backtest import load_matches, run_backtest, metrics
from models.config import TRAIN_SEASONS, make_model
from models.evaluate import (PCOLS, calibration, outcome_summary, with_probs,
                             vector_scale_fit, vector_scale_apply)

OUT = ROOT / "data" / "backtests"
TARGET = "2526"
CALIB_SEASONS = ["2324", "2425"]      # out-of-sample frames for vector scaling


def plain_model():
    """The model WITHOUT either draw correction — the comparison baseline.

    make_model() now fits the draw boost and carries DRAW_SCALE, so the
    "before" case has to switch both off explicitly.
    """
    m = make_model()
    from models.dixon_coles import DixonColes
    plain = DixonColes()
    plain.blend_w, plain.prior_strength, plain.xi = m.blend_w, m.prior_strength, m.xi
    return plain


def boost_only_model():
    """Fitted draw_boost, but without the out-of-sample DRAW_SCALE on top."""
    m = make_model()
    m.draw_scale = 1.0
    return m


def score(label, bt):
    m = metrics(bt)
    p = bt[PCOLS].to_numpy(float)
    return {"variant": label, "n": m["n"],
            "accuracy": round(m["accuracy"], 4),
            "brier": round(m["brier"], 4),
            "logloss": round(m["logloss"], 4),
            "mean_p_draw": round(float(p[:, 1].mean()), 4),
            "draw_picks": int((p.argmax(1) == 1).sum())}


def main():
    conn = connect()
    df = load_matches(conn)
    df = df[df["season"].isin(TRAIN_SEASONS)]
    OUT.mkdir(parents=True, exist_ok=True)
    rows, frames = [], {}

    base = run_backtest(df, TARGET, make_model=plain_model)
    frames["current model"] = base
    rows.append(score("no draw fix", base))
    print(f"  {rows[-1]}", flush=True)

    boost = run_backtest(df, TARGET, make_model=boost_only_model)
    frames["draw boost"] = boost
    rows.append(score("draw boost (fitted per refit)", boost))
    print(f"  {rows[-1]}", flush=True)

    shipped = run_backtest(df, TARGET, make_model=make_model)
    frames["boost + DRAW_SCALE"] = shipped
    rows.append(score("boost + DRAW_SCALE (as configured)", shipped))
    print(f"  {rows[-1]}", flush=True)

    # --- calibrators fitted on earlier seasons' out-of-sample output ---
    # Fitted on the UNCORRECTED model, so the scale it learns is the full
    # shortfall rather than whatever the boost left behind.
    cal = pd.concat([run_backtest(df, s, make_model=plain_model)
                     for s in CALIB_SEASONS], ignore_index=True)
    a, b = vector_scale_fit(cal[PCOLS].to_numpy(float), cal["actual"].to_numpy())
    print(f"  vector scaling fitted on {len(cal)} matches from "
          f"{'+'.join(CALIB_SEASONS)}: a={a.round(3)} b={b.round(3)}", flush=True)
    for label, src in (("current model", base), ("draw boost", boost)):
        q = vector_scale_apply(src[PCOLS].to_numpy(float), a, b)
        sc = with_probs(src, q)
        frames[f"{label} + scaling"] = sc
        rows.append(score(f"{label} + vector scaling", sc))
        print(f"  {rows[-1]}", flush=True)

    # --- single-parameter draw scaling, fitted out-of-sample ---
    # The per-refit boost is fitted IN-sample, where the model is already
    # better calibrated than it will be on unseen fixtures, so it
    # systematically under-corrects. Vector scaling is fitted out-of-sample but
    # has six parameters and drags home/away with it. This is the middle
    # option: one parameter, fitted on earlier seasons' out-of-sample output.
    def draw_only_fit(frame):
        from scipy.optimize import minimize_scalar
        p = frame[PCOLS].to_numpy(float)
        act = frame["actual"].to_numpy()

        def nll(s):
            q = p.copy()
            q[:, 1] *= s
            q /= q.sum(axis=1, keepdims=True)
            return -np.log(np.clip(q[np.arange(len(q)), act], 1e-12, None)).mean()

        return float(minimize_scalar(nll, bounds=(0.8, 2.0),
                                     method="bounded").x)

    def draw_only_apply(frame, s):
        q = frame[PCOLS].to_numpy(float).copy()
        q[:, 1] *= s
        q /= q.sum(axis=1, keepdims=True)
        return with_probs(frame, q)

    s = draw_only_fit(cal)
    print(f"  draw-only scale fitted out-of-sample on "
          f"{'+'.join(CALIB_SEASONS)}: s={s:.4f}", flush=True)
    for label, src in (("current model", base), ("draw boost", boost)):
        sc = draw_only_apply(src, s)
        frames[f"{label} + draw-only scale"] = sc
        rows.append(score(f"{label} + draw-only scale (oos)", sc))
        print(f"  {rows[-1]}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "draw_pass.csv", index=False)
    print("\n=== draw pass, 2025-26 holdout ===")
    print(table.to_string(index=False))
    print(f"\nactual draw rate: {(base['actual'] == 1).mean():.4f}")

    for label in ("current model", "draw boost", "draw boost + scaling"):
        if label not in frames:
            continue
        print(f"\n--- draw calibration: {label} ---")
        print(calibration(frames[label], "draw").to_string(index=False))
        print(outcome_summary(frames[label]).to_string(index=False))

    for label, f in frames.items():
        slug = "".join(c if c.isalnum() else "_" for c in label)[:40]
        f.to_csv(OUT / f"draw_{slug}.csv", index=False)


if __name__ == "__main__":
    main()
