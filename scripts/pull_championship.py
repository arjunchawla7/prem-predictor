"""Pull Championship (E1) results from football-data.co.uk.

Why: promoted sides arrive with no top-flight history, so the model falls back
to a "weakest-3 average" prior that knows nothing about the specific team. Two
seasons of second-tier results is the only free, structured record of how those
teams actually played.

What this does NOT give you:

  no xG        Understat covers six top-flight leagues and returns 404 for the
               Championship, so these rows are goals-only. The live model fits
               on xG (models/config.BLEND_W = 0), and DixonColes.fit falls back
               to goals per row where xG is missing — so E1 rows are fitted on
               a different target from E0 rows. That is a real inconsistency
               and a reason to be suspicious of the ratings it produces.
  no lineups   football-data.co.uk carries results only, so formations cannot
               be derived from these rows. Nothing here fixes the "not enough
               lineup data" fallback for a promoted side.

Rows are tagged division='E1' so nothing reads them as top-flight by accident.
Cross-league ratings are only as good as the division adjustment applied to
them — see scripts/promoted_prior_pass.py for whether that actually pays.
"""
import argparse
import io
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from net import session
from backend.db import connect, log_pull

SRC = "https://www.football-data.co.uk/mmz4281/{s}/E1.csv"
RAW = ROOT / "data" / "raw" / "championship"

# football-data.co.uk spells some clubs differently in E1 than we seeded them
# from E0 / the PL squad API. Without this the same club would be inserted a
# second time under a near-identical name and its history would split in two.
NAME_MAP = {
    "Coventry": "Coventry City",
    "Hull": "Hull City",
    "Sheffield Weds": "Sheffield Wednesday",
    "Nott'm Forest": "Nott'm Forest",
}

ODDS = [("AvgCH", "AvgCD", "AvgCA"), ("AvgH", "AvgD", "AvgA"),
        ("B365CH", "B365CD", "B365CA"), ("B365H", "B365D", "B365A")]

HTTP = session()


def team_id(conn, name):
    name = NAME_MAP.get(name, name)
    row = conn.execute("SELECT id FROM teams WHERE name=?", (name,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute("INSERT INTO teams (name) VALUES (?)", (name,))
    print(f"  new team '{name}' (no stadium coords — second tier only)")
    return cur.lastrowid


def _i(v):
    return None if v is None or pd.isna(v) else int(v)


def load_season(conn, season, refresh=False):
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / f"E1_{season}.csv"
    if not f.exists() or refresh:
        r = HTTP.get(SRC.format(s=season), timeout=40)
        r.raise_for_status()
        f.write_bytes(r.content)
    df = pd.read_csv(io.BytesIO(f.read_bytes()))
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    dates = pd.to_datetime(df["Date"], dayfirst=True)
    trip = next((t for t in ODDS if all(c in df.columns for c in t)), None)

    n = 0
    for i, r in df.iterrows():
        hid, aid = team_id(conn, r["HomeTeam"]), team_id(conn, r["AwayTeam"])
        cur = conn.execute(
            """INSERT INTO matches (season, date, home_team_id, away_team_id,
                 fthg, ftag, ftr, hthg, htag, htr, referee, division)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,'E1')
               ON CONFLICT(season, date, home_team_id, away_team_id)
               DO NOTHING""",
            (season, dates[i].strftime("%Y-%m-%d"), hid, aid,
             int(r["FTHG"]), int(r["FTAG"]), r["FTR"],
             _i(r.get("HTHG")), _i(r.get("HTAG")), r.get("HTR"),
             r.get("Referee")))
        if cur.rowcount == 0:
            continue
        mid = cur.lastrowid
        if trip:
            h, d, a = (r.get(c) for c in trip)
            if not (pd.isna(h) or pd.isna(d) or pd.isna(a)):
                conn.execute(
                    """UPDATE matches SET odds_home=?, odds_draw=?, odds_away=?
                       WHERE id=?""", (float(h), float(d), float(a), mid))
        for tid, pre, home in ((hid, "H", 1), (aid, "A", 0)):
            conn.execute(
                """INSERT OR IGNORE INTO team_match_stats
                     (match_id, team_id, is_home, shots, shots_on_target,
                      corners, fouls, yellows, reds)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (mid, tid, home, _i(r.get(pre + "S")), _i(r.get(pre + "ST")),
                 _i(r.get(pre + "C")), _i(r.get(pre + "F")),
                 _i(r.get(pre + "Y")), _i(r.get(pre + "R"))))
        n += 1
    conn.commit()
    print(f"E1 {season}: +{n} matches"
          + (f" (odds via {trip[0][:-1]})" if trip else " (no odds columns)"))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", default=["2425", "2526"])
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    conn = connect()
    total = 0
    for s in args.seasons:
        try:
            total += load_season(conn, s, args.refresh)
        except Exception as e:
            print(f"E1 {s}: FAILED {e}")
            log_pull(conn, "championship", False, f"{s}: {e}")
    log_pull(conn, "championship", True, f"{total} E1 matches")

    rows = conn.execute(
        """SELECT t.name, COUNT(*) n, MIN(m.date) a, MAX(m.date) b
           FROM matches m JOIN teams t
             ON t.id IN (m.home_team_id, m.away_team_id)
           WHERE m.division='E1' GROUP BY t.name ORDER BY t.name""").fetchall()
    print(f"\nE1 coverage ({total} matches loaded):")
    for r in rows:
        print(f"  {r['name']:<22} {r['n']:>3} matches  {r['a']} .. {r['b']}")


if __name__ == "__main__":
    main()
