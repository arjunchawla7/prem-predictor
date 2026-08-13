"""Derived team profile: preferred formation + tactical trait labels.

FORMATION — derived, not scraped. For each past match we take the eleven
players who started, read their understat slot code, and bucket each into a
line:

    GK                          -> keeper (never counted in the string)
    DL DC DR DML DMR            -> defence
    DMC MC ML MR                -> midfield
    AMC AML AMR                 -> attacking midfield
    FW FWL FWR                  -> attack

The formation string is those counts, dropping any empty line:
4/2/3/1 -> "4-2-3-1", 4/3/0/3 -> "4-3-3", 3/5/0/2 -> "3-5-2". The team's
"preferred" formation is simply the one used most across the window, and
the runners-up are reported with it so a 50/50 split is visible rather than
hidden behind a single label.

TACTICS — trait labels from league percentile ranks over the same window,
using the proxies available in this database (no possession or pressing
data exists in it, so those two are inferred cautiously or omitted):

    shots per 90                 high -> "high volume attack"
    xG per shot                  high -> "patient chance quality"
                                 low  -> "shoots from range / direct"
    opponent shots per 90        low  -> "restricts shot volume (low block)"
                                 high -> "open, concedes chances"
    opponent xG per shot         low  -> "funnels opponents to poor chances"
    corners per 90               high -> "set-piece heavy"

Every label is a relative statement about this league season, and each is
tagged with the percentile it came from so the frontend can show the
evidence instead of asserting a style.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WINDOW = 30          # matches per team
LINE_OF = {
    "GK": "gk",
    "DL": "def", "DC": "def", "DR": "def", "DML": "def", "DMR": "def",
    "DMC": "mid", "MC": "mid", "ML": "mid", "MR": "mid",
    "AMC": "am", "AML": "am", "AMR": "am",
    "FW": "fwd", "FWL": "fwd", "FWR": "fwd",
}
HIGH, LOW = 0.75, 0.25


def match_formations(conn, team_id, before_date=None, window=WINDOW):
    """[(match_date, formation_string)] most recent first."""
    rows = conn.execute(
        """SELECT m.id, m.date, pmm.slot_pos
           FROM player_match_minutes pmm
           JOIN matches m ON m.id = pmm.match_id
           WHERE pmm.team_id = ? AND pmm.started = 1
             AND pmm.slot_pos IS NOT NULL
             AND (? IS NULL OR m.date < ?)
           ORDER BY m.date DESC""",
        (team_id, before_date, before_date)).fetchall()
    per_match = {}
    for r in rows:
        per_match.setdefault((r["id"], r["date"]), []).append(r["slot_pos"])

    out = []
    for (mid, date), slots in list(per_match.items())[:window]:
        counts = {"gk": 0, "def": 0, "mid": 0, "am": 0, "fwd": 0}
        unknown = 0
        for s in slots:
            line = LINE_OF.get(s)
            if line:
                counts[line] += 1
            else:
                unknown += 1
        outfield = counts["def"] + counts["mid"] + counts["am"] + counts["fwd"]
        # a lineup with substitutes miscoded (understat marks a few starters
        # 'Sub') can't be shaped reliably — skip rather than invent a shape
        if unknown or outfield != 10 or counts["gk"] != 1:
            continue
        shape = "-".join(str(counts[k]) for k in ("def", "mid", "am", "fwd")
                         if counts[k] > 0)
        out.append((date, shape))
    return out


def preferred_formation(conn, team_id, before_date=None):
    """{'formation', 'share', 'sample', 'alternatives': [(shape, n), ...]}"""
    fs = match_formations(conn, team_id, before_date)
    if not fs:
        return None
    counts = {}
    for _, shape in fs:
        counts[shape] = counts.get(shape, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    top, n = ranked[0]
    return {"formation": top, "sample": len(fs),
            "share": round(n / len(fs), 3),
            "alternatives": [{"formation": s, "matches": c}
                             for s, c in ranked[1:4]]}


def _team_metrics(conn, before_date=None):
    q = """
    SELECT s.team_id, s.shots, s.corners, opp.shots AS shots_ag,
           CASE WHEN s.is_home=1 THEN m.home_xg ELSE m.away_xg END AS xg_for,
           CASE WHEN s.is_home=1 THEN m.away_xg ELSE m.home_xg END AS xg_ag,
           m.date
    FROM team_match_stats s
    JOIN matches m ON m.id = s.match_id
    JOIN team_match_stats opp ON opp.match_id = s.match_id
                             AND opp.team_id != s.team_id
    WHERE m.home_xg IS NOT NULL AND (? IS NULL OR m.date < ?)
    ORDER BY m.date DESC
    """
    df = pd.read_sql_query(q, conn, params=(before_date, before_date))
    df = df.groupby("team_id").head(WINDOW)
    g = df.groupby("team_id").agg(
        n=("shots", "size"), shots=("shots", "mean"),
        shots_ag=("shots_ag", "mean"), corners=("corners", "mean"),
        xg_for=("xg_for", "mean"), xg_ag=("xg_ag", "mean"))
    g = g[g["n"] >= 10]
    g["xg_per_shot"] = g["xg_for"] / g["shots"].clip(lower=1)
    g["xg_per_shot_ag"] = g["xg_ag"] / g["shots_ag"].clip(lower=1)
    return g


def tactical_traits(conn, team_id, before_date=None):
    """Trait labels with the percentile each came from (empty list is a
    valid answer — a team with no standout metric gets no invented style)."""
    g = _team_metrics(conn, before_date)
    if team_id not in g.index:
        return []
    pr = g.rank(pct=True)
    row, me = pr.loc[team_id], g.loc[team_id]
    traits = []

    def add(label, pctile, detail):
        traits.append({"label": label, "percentile": round(100 * pctile),
                       "detail": detail})

    if row["shots"] >= HIGH:
        add("High-volume attack", row["shots"], f"{me['shots']:.1f} shots/match")
    elif row["shots"] <= LOW:
        add("Low-volume attack", row["shots"], f"{me['shots']:.1f} shots/match")
    if row["xg_per_shot"] >= HIGH:
        add("Works the ball into good positions", row["xg_per_shot"],
            f"{me['xg_per_shot']:.3f} xG per shot")
    elif row["xg_per_shot"] <= LOW:
        add("Shoots from lower-value spots", row["xg_per_shot"],
            f"{me['xg_per_shot']:.3f} xG per shot")
    if row["shots_ag"] <= LOW:
        add("Restricts shot volume", 1 - row["shots_ag"],
            f"{me['shots_ag']:.1f} shots faced/match")
    elif row["shots_ag"] >= HIGH:
        add("Open, concedes shots", row["shots_ag"],
            f"{me['shots_ag']:.1f} shots faced/match")
    if row["xg_per_shot_ag"] <= LOW:
        add("Funnels opponents to poor chances", 1 - row["xg_per_shot_ag"],
            f"{me['xg_per_shot_ag']:.3f} xG per shot faced")
    if row["corners"] >= HIGH:
        add("Set-piece heavy", row["corners"], f"{me['corners']:.1f} corners/match")
    return traits


def team_profile(conn, team_id, before_date=None):
    return {"formation": preferred_formation(conn, team_id, before_date),
            "traits": tactical_traits(conn, team_id, before_date)}
