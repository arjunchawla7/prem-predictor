"""Player rating system — a documented heuristic, not an official rating.

Data basis: previous-season understat aggregates (player_season_stats).
Understat has no tackle/interception counts, so the defensive proxy is:
  - team goals conceded per 90 in the league matches the player appeared in
    (from player_match_minutes + match scores), inverted, plus
  - xGBuildup per 90 (involvement in build-up play).
That gap (no true defensive actions) is a known limitation, noted rather
than papered over with invented numbers.

Scoring (all per-90, z-scored WITHIN position group — never across):
  GK / DEF : 0.65 * z(-team goals conceded per90 while on pitch)
           + 0.35 * z(xGBuildup per90)
  MID      : 0.50 * z(npxG+xA per90) + 0.30 * z(xGChain per90)
           + 0.20 * z(-team conceded per90 while on pitch)
  FWD      : 0.80 * z(npxG+xA per90) + 0.20 * z(xGChain per90)

Tier: quintile of the score within the position group → 5 (top) .. 1.
Strength index (used by the lineup-weighted team rating):
  index = 1 + 0.08 * clip(z_composite, -2, 2)      # so 0.84 .. 1.16

Provisional rule: players with < MIN_MINUTES rated minutes in the source
season (new signings, youth) get index 1.0, tier 3, provisional=True —
a flagged league-average fallback, not a fabricated confident number.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIN_MINUTES = 900          # ~10 full matches to earn a real rating
INDEX_SLOPE = 0.08         # index spread per z-unit
Z_CLIP = 2.0


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 1e-9 else s * 0


def conceded_per90_while_playing(conn, season: str) -> pd.DataFrame:
    """Team goals conceded per 90 across the matches each player appeared in."""
    q = """
    SELECT pmm.player_id,
           SUM(pmm.minutes) AS mins,
           SUM(CASE WHEN pmm.team_id = m.home_team_id THEN m.ftag
                    ELSE m.fthg END * pmm.minutes / 90.0) AS conceded_w
    FROM player_match_minutes pmm
    JOIN matches m ON m.id = pmm.match_id
    WHERE m.season = ?
    GROUP BY pmm.player_id
    """
    df = pd.read_sql_query(q, conn, params=(season,))
    df["conceded_p90"] = np.where(df["mins"] > 0,
                                  90.0 * df["conceded_w"] / df["mins"], np.nan)
    return df[["player_id", "conceded_p90"]]


def compute_ratings(conn, source_season: str) -> pd.DataFrame:
    """Rate every player from `source_season` stats. Returns one row per
    player: score, tier (1-5), index, provisional flag."""
    st = pd.read_sql_query(
        "SELECT * FROM player_season_stats WHERE season = ?",
        conn, params=(source_season,))
    st = st.merge(conceded_per90_while_playing(conn, source_season),
                  on="player_id", how="left")

    st["mins"] = st["minutes"].fillna(0)
    per90 = lambda col: 90.0 * st[col] / st["mins"].clip(lower=1)
    st["att_p90"] = per90("npxg") + per90("xa")
    st["chain_p90"] = per90("xg_chain")
    st["buildup_p90"] = per90("xg_buildup")

    rated = st["mins"] >= MIN_MINUTES
    st["score"] = np.nan
    for pos, weights in {
        "GK":  [("conceded_p90", -0.65), ("buildup_p90", 0.35)],
        "DEF": [("conceded_p90", -0.65), ("buildup_p90", 0.35)],
        "MID": [("att_p90", 0.50), ("chain_p90", 0.30), ("conceded_p90", -0.20)],
        "FWD": [("att_p90", 0.80), ("chain_p90", 0.20)],
    }.items():
        m = rated & (st["position"] == pos)
        if m.sum() < 5:
            continue
        score = sum(w * _z(st.loc[m, col].fillna(st.loc[m, col].median()))
                    for col, w in weights)
        st.loc[m, "score"] = score

    st["provisional"] = ~rated | st["score"].isna()
    st["index"] = 1.0
    ok = ~st["provisional"]
    st.loc[ok, "index"] = 1.0 + INDEX_SLOPE * st.loc[ok, "score"].clip(-Z_CLIP, Z_CLIP)
    st["tier"] = 3
    for pos in ("GK", "DEF", "MID", "FWD"):
        m = ok & (st["position"] == pos)
        if m.sum() >= 5:
            st.loc[m, "tier"] = (
                pd.qcut(st.loc[m, "score"].rank(method="first"), 5,
                        labels=[1, 2, 3, 4, 5]).astype(int))
    return st[["player_id", "team_id", "position", "mins", "score",
               "tier", "index", "provisional"]]


class RatingBook:
    """Lookup helper: player_id -> (index, tier, position, provisional)."""

    def __init__(self, conn, source_season: str):
        df = compute_ratings(conn, source_season)
        self.by_id = df.set_index("player_id").to_dict("index")
        self.source_season = source_season

    def index_of(self, player_id: int) -> float:
        r = self.by_id.get(player_id)
        return r["index"] if r else 1.0     # unknown player = provisional avg

    def is_provisional(self, player_id: int) -> bool:
        r = self.by_id.get(player_id)
        return r["provisional"] if r else True
