"""Pull historical closing odds from football-data.co.uk into matches.

the-odds-api (scripts/pull_odds.py) only serves odds for UPCOMING fixtures, so
it can never supply a market baseline for a backtest. football-data.co.uk
publishes per-season CSVs carrying the market's closing prices, which is what a
market-blend variant has to be scored against.

Columns used, in preference order:
  AvgC{H,D,A}  market-average CLOSING odds  (best available signal)
  AvgO{H,D,A}  market-average opening odds
  B365C{H,D,A} / B365{H,D,A}   single-book fallback

Odds are stored raw; overround removal happens at read time so the
normalisation rule stays visible where it is used.

Note: this host reaches football-data.co.uk directly. scripts/download_history.py
uses the GitHub mirror because the domain was FortiGuard-blocked as "Gambling"
when that script was written; the mirror drops the odds columns entirely, which
is why this puller goes to the source instead.
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

SRC = "https://www.football-data.co.uk/mmz4281/{s}/E0.csv"
RAW = ROOT / "data" / "raw" / "odds"

TRIPLETS = [("AvgCH", "AvgCD", "AvgCA"), ("AvgH", "AvgD", "AvgA"),
            ("B365CH", "B365CD", "B365CA"), ("B365H", "B365D", "B365A")]

HTTP = session()


def add_odds_columns(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(matches)")}
    for c in ("odds_home", "odds_draw", "odds_away"):
        if c not in cols:
            conn.execute(f"ALTER TABLE matches ADD COLUMN {c} REAL")
    conn.commit()


def fetch_season(season: str, refresh: bool) -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / f"E0_{season}.csv"
    if not f.exists() or refresh:
        r = HTTP.get(SRC.format(s=season), timeout=30)
        r.raise_for_status()
        f.write_bytes(r.content)
    return pd.read_csv(io.BytesIO(f.read_bytes()))


def load_season(conn, season: str, refresh: bool = False) -> int:
    df = fetch_season(season, refresh)
    trip = next((t for t in TRIPLETS if all(c in df.columns for c in t)), None)
    if trip is None:
        print(f"{season}: no usable odds columns")
        return 0
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    n = 0
    for _, r in df.iterrows():
        h, d, a = (r.get(c) for c in trip)
        if pd.isna(h) or pd.isna(d) or pd.isna(a):
            continue
        cur = conn.execute(
            """UPDATE matches SET odds_home=?, odds_draw=?, odds_away=?
               WHERE season=? AND home_team_id=(SELECT id FROM teams WHERE name=?)
                 AND away_team_id=(SELECT id FROM teams WHERE name=?)""",
            (float(h), float(d), float(a), season, r["HomeTeam"], r["AwayTeam"]))
        n += cur.rowcount
    conn.commit()
    print(f"{season}: {n} matches priced (using {trip[0][:-1]})")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*",
                    default=["1920", "2021", "2122", "2223", "2324", "2425", "2526"])
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    conn = connect()
    add_odds_columns(conn)
    total = 0
    for s in args.seasons:
        try:
            total += load_season(conn, s, args.refresh)
        except Exception as e:
            print(f"{s}: FAILED {e}")
            log_pull(conn, "odds history", False, f"{s}: {e}")
    log_pull(conn, "odds history", True, f"{total} matches priced")
    print(f"total priced: {total}")


if __name__ == "__main__":
    main()
