"""Layer 2 — tactical style buckets + residualised matchup adjustment.

Style metrics per team, computed over their last STYLE_WINDOW league matches
before a given date (proxies only — no possession/pressing data in the DB):

  shot_volume    shots taken per 90
  shot_quality   xG per shot taken (direct/counter teams score high here)
  concede_vol    shots conceded per 90 (low-block teams concede many shots
                 but often low-quality ones)
  concede_qual   opponent xG per shot conceded
  set_piece      corners won per 90 (crude directness/territory proxy)

Teams are clustered into K=4 buckets with k-means on z-scored metrics.

Matchup adjustment — the residualisation step (important): we do NOT learn
"style A beats style B" from raw results, because that mostly rediscovers
"good teams beat bad teams". Instead, for every training match we take the
RESIDUAL between actual goals and what Layer 1 (Dixon-Coles) expected, then
average residuals per (attacking style, defending style) pair. Only the part
of the outcome Layer 1 could NOT explain is attributed to style.

Application: expected goals += pair adjustment, capped at ±ADJ_CAP goals —
a small nudge on top of Layer 1, never a replacement. Pairs with fewer than
MIN_PAIR_N training matches get zero adjustment (no invented signal).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

K = 4
STYLE_WINDOW = 30          # matches per team for style metrics
ADJ_CAP = 0.15             # max xG nudge per side
MIN_PAIR_N = 25            # min matches behind a pair adjustment


def team_style_metrics(conn, as_of_date):
    """DataFrame indexed by team_id with the five style metrics."""
    q = """
    SELECT s.team_id, s.is_home, s.shots, s.corners,
           CASE WHEN s.is_home=1 THEN m.home_xg ELSE m.away_xg END AS xg_for,
           CASE WHEN s.is_home=1 THEN m.away_xg ELSE m.home_xg END AS xg_ag,
           opp.shots AS shots_ag
    FROM team_match_stats s
    JOIN matches m ON m.id = s.match_id
    JOIN team_match_stats opp ON opp.match_id = s.match_id
                              AND opp.team_id != s.team_id
    WHERE m.date < ? AND m.home_xg IS NOT NULL
    ORDER BY m.date DESC
    """
    df = pd.read_sql_query(q, conn, params=(str(as_of_date)[:10],))
    df = df.groupby("team_id").head(STYLE_WINDOW)
    g = df.groupby("team_id").agg(
        n=("shots", "size"), shots=("shots", "mean"),
        shots_ag=("shots_ag", "mean"), corners=("corners", "mean"),
        xg_for=("xg_for", "mean"), xg_ag=("xg_ag", "mean"))
    g = g[g["n"] >= 10]
    out = pd.DataFrame(index=g.index)
    out["shot_volume"] = g["shots"]
    out["shot_quality"] = g["xg_for"] / g["shots"].clip(lower=1)
    out["concede_vol"] = g["shots_ag"]
    out["concede_qual"] = g["xg_ag"] / g["shots_ag"].clip(lower=1)
    out["set_piece"] = g["corners"]
    return out


def cluster_styles(metrics: pd.DataFrame, seed=0):
    """k-means (scipy) on z-scored metrics -> {team_id: bucket 0..K-1}."""
    from scipy.cluster.vq import kmeans2
    X = ((metrics - metrics.mean()) / metrics.std(ddof=0)).to_numpy()
    _, labels = kmeans2(X, K, minit="++", seed=seed)
    return dict(zip(metrics.index, labels.astype(int)))


def fit_pair_adjustments(conn, dc_model, styles, as_of_date, team_names):
    """Mean Layer-1 residual per (attacking bucket, defending bucket).

    team_names: {team_id: DC model team key}. Returns dict
    {(atk_bucket, def_bucket): goals_adjustment} with small-sample pairs 0.
    """
    q = """
    SELECT m.home_team_id AS h, m.away_team_id AS a, m.fthg, m.ftag
    FROM matches m WHERE m.date < ?
    """
    df = pd.read_sql_query(q, conn, params=(str(as_of_date)[:10],))
    sums, counts = {}, {}
    for r in df.itertuples():
        hk, ak = team_names.get(r.h), team_names.get(r.a)
        if (hk not in dc_model.attack or ak not in dc_model.attack
                or r.h not in styles or r.a not in styles):
            continue
        lam, mu = dc_model.expected_goals(hk, ak)
        for atk_t, def_t, actual, exp in ((r.h, r.a, r.fthg, lam),
                                          (r.a, r.h, r.ftag, mu)):
            key = (styles[atk_t], styles[def_t])
            sums[key] = sums.get(key, 0.0) + (actual - exp)
            counts[key] = counts.get(key, 0) + 1
    return {k: float(np.clip(sums[k] / counts[k], -ADJ_CAP, ADJ_CAP))
            if counts[k] >= MIN_PAIR_N else 0.0
            for k in sums}


class StyleAdjuster:
    """Bundles the three steps; built per fit date in backtests."""

    def __init__(self, conn, dc_model, as_of_date, team_names):
        self.metrics = team_style_metrics(conn, as_of_date)
        self.styles = (cluster_styles(self.metrics)
                       if len(self.metrics) >= K else {})
        self.pairs = (fit_pair_adjustments(conn, dc_model, self.styles,
                                           as_of_date, team_names)
                      if self.styles else {})

    def adjust(self, home_id, away_id, lam, mu):
        """Returns adjusted (lam, mu). Unknown style -> no change."""
        sh, sa = self.styles.get(home_id), self.styles.get(away_id)
        if sh is None or sa is None:
            return lam, mu
        lam = max(0.05, lam + self.pairs.get((sh, sa), 0.0))
        mu = max(0.05, mu + self.pairs.get((sa, sh), 0.0))
        return lam, mu
