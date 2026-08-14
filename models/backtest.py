"""Time-series backtest harness.

Walks the target season in strict date order. Before each match date the
model is refitted using ONLY matches played before that date (across all
loaded seasons) — never shuffled, no leakage. Promoted teams with no prior
top-flight matches in the window get a "promoted prior" (the average
attack/defence of the 3 weakest fitted teams) and the prediction is flagged.

Metrics:
  accuracy   share of matches where argmax(pH,pD,pA) equalled the result
  brier      multiclass Brier score, mean over matches of
             sum_k (p_k - 1{k})^2   (lower better; 0.667 = uniform guess)
  logloss    mean -log p(actual outcome) (lower better; 1.099 = uniform)
  calibration  bucketed predicted-vs-actual frequencies for the home-win prob
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.dixon_coles import DixonColes


def load_matches(conn):
    return pd.read_sql_query(
        """SELECT m.id, m.season, m.date, h.name AS home, a.name AS away,
                  m.fthg, m.ftag, m.ftr, m.hthg, m.htag, m.htr,
                  m.home_xg, m.away_xg,
                  m.odds_home, m.odds_draw, m.odds_away
           FROM matches m
           JOIN teams h ON h.id = m.home_team_id
           JOIN teams a ON a.id = m.away_team_id
           ORDER BY m.date""", conn)


def promoted_prior(model):
    """Fallback rating for a team the fitted model has never seen: average of
    the 3 weakest teams (lowest attack, highest defence factor)."""
    atk = sorted(model.attack.values())
    dfc = sorted(model.defence.values(), reverse=True)
    return float(np.mean(atk[:3])), float(np.mean(dfc[:3]))


def run_backtest(df, target_season, xg_multipliers=None, refit_every=7,
                 min_train=380, make_model=None):
    """xg_multipliers: optional callable (row, model) -> (lam_mult, mu_mult)
    letting lineup/fatigue/travel layers plug into the same harness.
    refit_every: refit cadence in days (7 = roughly each gameweek).
    make_model: zero-arg factory returning a fresh, configured DixonColes
    (used by the accuracy-pass variants to vary xi / blend_w)."""
    make_model = make_model or DixonColes
    target = df[df["season"] == target_season].sort_values("date")
    records = []
    model, last_fit = None, None

    for _, row in target.iterrows():
        date = pd.Timestamp(row["date"])
        if model is None or (date - last_fit).days >= refit_every:
            train = df[pd.to_datetime(df["date"]) < date]
            if len(train) < min_train:
                continue
            model = make_model().fit(train.to_dict("records"), as_of=date)
            last_fit = date

        flagged = False
        for t in (row["home"], row["away"]):
            if t not in model.attack:
                model.attack[t], model.defence[t] = promoted_prior(model)
                flagged = True

        lam_mult = mu_mult = 1.0
        if xg_multipliers is not None:
            lam_mult, mu_mult = xg_multipliers(row, model)
        pred = model.predict(row["home"], row["away"], lam_mult, mu_mult)

        actual = {"H": 0, "D": 1, "A": 2}[row["ftr"]]
        probs = np.array([pred["p_home"], pred["p_draw"], pred["p_away"]])
        records.append({
            "date": row["date"], "home": row["home"], "away": row["away"],
            "p_home": probs[0], "p_draw": probs[1], "p_away": probs[2],
            "actual": actual, "pred": int(np.argmax(probs)),
            "provisional_rating": flagged,
            # carried through so market-blend and situational variants can be
            # scored off the same frame without a second walk-forward pass
            "odds_home": row.get("odds_home"), "odds_draw": row.get("odds_draw"),
            "odds_away": row.get("odds_away"),
            "hthg": row.get("hthg"), "htag": row.get("htag"),
        })
    return pd.DataFrame(records)


def metrics(bt):
    probs = bt[["p_home", "p_draw", "p_away"]].to_numpy()
    onehot = np.eye(3)[bt["actual"].to_numpy()]
    return {
        "n": len(bt),
        "accuracy": float((bt["pred"] == bt["actual"]).mean()),
        "brier": float(((probs - onehot) ** 2).sum(axis=1).mean()),
        "logloss": float(-np.log(np.clip(
            probs[np.arange(len(bt)), bt["actual"]], 1e-12, None)).mean()),
    }


def calibration_table(bt, bins=8):
    """Reliability of the home-win probability, bucketed."""
    p = bt["p_home"].to_numpy()
    hit = (bt["actual"] == 0).to_numpy(float)
    edges = np.linspace(p.min(), p.max() + 1e-9, bins + 1)
    rows = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() >= 5:
            rows.append({"bucket": f"{edges[i]:.2f}-{edges[i+1]:.2f}",
                         "n": int(m.sum()),
                         "predicted": float(p[m].mean()),
                         "actual": float(hit[m].mean())})
    return pd.DataFrame(rows)
