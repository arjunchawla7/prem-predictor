"""Load data/raw/E0_*.csv (football-data.co.uk schema) into SQLite.

Idempotent: upserts on (season, date, home, away), so re-running after a
weekly refresh only adds new matches.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.db import connect

RAW = ROOT / "data" / "raw"


def team_id(conn, name: str) -> int:
    row = conn.execute("SELECT id FROM teams WHERE name=?", (name,)).fetchone()
    if row:
        return row["id"]
    # Unknown (e.g. newly promoted) team: insert with NULL coords, flag later.
    cur = conn.execute("INSERT INTO teams (name) VALUES (?)", (name,))
    print(f"  note: unknown team '{name}' inserted without stadium coords")
    return cur.lastrowid


def load_season(conn, csv: Path):
    season = csv.stem.split("_")[1]
    df = pd.read_csv(csv)
    # Mirror uses ISO yyyy-mm-dd; football-data.co.uk uses dd/mm/yy(yy).
    # Detect explicitly — mixing dayfirst with ISO swaps month/day silently.
    sample = str(df["Date"].iloc[0])
    if len(sample.split("-")[0]) == 4:
        dates = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    else:
        dates = pd.to_datetime(df["Date"], dayfirst=True)
    n = 0
    for i, r in df.iterrows():
        hid, aid = team_id(conn, r["HomeTeam"]), team_id(conn, r["AwayTeam"])
        cur = conn.execute(
            """INSERT INTO matches (season, date, home_team_id, away_team_id,
                 fthg, ftag, ftr, hthg, htag, htr, referee)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(season, date, home_team_id, away_team_id) DO NOTHING""",
            (season, dates[i].strftime("%Y-%m-%d"), hid, aid,
             int(r["FTHG"]), int(r["FTAG"]), r["FTR"],
             _i(r.get("HTHG")), _i(r.get("HTAG")), r.get("HTR"),
             r.get("Referee")))
        if cur.rowcount == 0:
            continue
        mid = cur.lastrowid
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
    print(f"{csv.name}: +{n} new matches")


def _i(v):
    return None if v is None or pd.isna(v) else int(v)


def main():
    conn = connect()
    for csv in sorted(RAW.glob("E0_*.csv")):
        load_season(conn, csv)
    total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    print(f"total matches: {total}")


if __name__ == "__main__":
    main()
