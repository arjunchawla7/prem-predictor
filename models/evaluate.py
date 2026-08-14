"""Shared scoring helpers for the accuracy-pass experiments.

`models.backtest.metrics` scores a finished backtest frame; this module adds
the pieces the accuracy pass needs on top:

  market_probs     closing odds -> overround-free implied probabilities
  blend            convex mix of two probability sets
  calibration      reliability buckets for ANY outcome, not just home wins
  outcome_summary  per-outcome predicted-vs-actual rates (draw bias check)
"""
import numpy as np
import pandas as pd

OUTCOMES = {"home": 0, "draw": 1, "away": 2}
PCOLS = ["p_home", "p_draw", "p_away"]


def market_probs(df):
    """(n, 3) implied probabilities from decimal odds, overround removed.

    Bookmakers price 1/odds to sum above 1 (the margin); proportional
    normalisation is the standard first-order way to strip it. Rows with any
    missing price come back as NaN so callers can decide what to do.
    """
    raw = np.column_stack([1.0 / df["odds_home"].to_numpy(float),
                           1.0 / df["odds_draw"].to_numpy(float),
                           1.0 / df["odds_away"].to_numpy(float)])
    return raw / raw.sum(axis=1, keepdims=True)


def blend(p_model, p_other, w=0.5):
    """w*model + (1-w)*other, falling back to the model where `other` is NaN."""
    out = w * p_model + (1 - w) * p_other
    bad = ~np.isfinite(out).all(axis=1)
    out[bad] = p_model[bad]
    return out


def with_probs(bt, probs):
    """Copy of a backtest frame carrying replacement probabilities."""
    out = bt.copy()
    out[PCOLS] = probs
    out["pred"] = probs.argmax(axis=1)
    return out


def calibration(bt, outcome="home", bins=8, min_n=5):
    """Reliability buckets for one outcome's predicted probability."""
    k = OUTCOMES[outcome]
    p = bt[PCOLS[k]].to_numpy(float)
    hit = (bt["actual"].to_numpy() == k).astype(float)
    edges = np.linspace(p.min(), p.max() + 1e-9, bins + 1)
    rows = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() >= min_n:
            rows.append({"bucket": f"{edges[i]:.2f}-{edges[i+1]:.2f}",
                         "n": int(m.sum()),
                         "predicted": round(float(p[m].mean()), 4),
                         "actual": round(float(hit[m].mean()), 4),
                         "gap": round(float(hit[m].mean() - p[m].mean()), 4)})
    return pd.DataFrame(rows)


def vector_scale_fit(probs, actual):
    """Fit per-class log-probability scaling: q ∝ p^a * exp(b).

    Standard multiclass recalibration (vector scaling). Three temperature
    terms and three biases, so it can correct a systematic draw shortfall
    without being told which class is wrong. Returns (a, b).
    """
    from scipy.optimize import minimize
    lp = np.log(np.clip(probs, 1e-12, None))
    onehot = np.eye(3)[actual]

    def nll(t):
        a, b = t[:3], t[3:]
        z = a * lp + b
        z -= z.max(axis=1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=1, keepdims=True)
        return -(onehot * np.log(np.clip(q, 1e-12, None))).sum()

    res = minimize(nll, np.array([1., 1., 1., 0., 0., 0.]), method="L-BFGS-B")
    return res.x[:3], res.x[3:]


def vector_scale_apply(probs, a, b):
    lp = np.log(np.clip(probs, 1e-12, None))
    z = a * lp + b
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=1, keepdims=True)


def outcome_summary(bt):
    """Mean predicted probability vs realised rate vs how often each outcome
    was the argmax pick. The draw row is the one that matters for 4c."""
    rows = []
    for name, k in OUTCOMES.items():
        p = bt[PCOLS[k]].to_numpy(float)
        rows.append({
            "outcome": name,
            "mean_pred_prob": round(float(p.mean()), 4),
            "actual_rate": round(float((bt["actual"] == k).mean()), 4),
            "picked_as_argmax": round(float((bt["pred"] == k).mean()), 4),
            "max_prob_seen": round(float(p.max()), 4),
            "recall": round(float(
                (bt.loc[bt["actual"] == k, "pred"] == k).mean()), 4),
        })
    return pd.DataFrame(rows)
