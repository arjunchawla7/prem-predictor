"""Pull EPL match odds from the-odds-api.com (free tier: 500 req/month).

Needs an API key in the ODDS_API_KEY environment variable (register free at
https://the-odds-api.com). Without a key this logs the gap and exits 0 so
the weekly refresh keeps working — the frontend then shows "no market odds".

Implied probabilities are computed in the frontend after removing the
bookmaker overround (normalise 1/odds to sum to 1).
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from net import session
from backend.db import connect, log_pull

API = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
# the-odds-api team names -> our teams.name
NAME_MAP = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves", "Brighton and Hove Albion": "Brighton",
    "Leeds United": "Leeds", "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich", "Luton Town": "Luton",
    "Sheffield United": "Sheffield United", "AFC Bournemouth": "Bournemouth",
}


def main():
    key = os.environ.get("ODDS_API_KEY")
    conn = connect()
    if not key:
        log_pull(conn, "odds", False, "ODDS_API_KEY not set — market odds skipped")
        print("no ODDS_API_KEY set — market odds skipped (register free at "
              "the-odds-api.com, then setx ODDS_API_KEY <key>)")
        return
    s = session()
    try:
        r = s.get(API, params={"apiKey": key, "regions": "uk,eu",
                               "markets": "h2h", "oddsFormat": "decimal"},
                  timeout=30)
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        log_pull(conn, "odds", False, str(e))
        print(f"odds pull FAILED: {e}")
        sys.exit(1)

    tid = {r["name"]: r["id"] for r in conn.execute("SELECT id, name FROM teams")}
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    for ev in events:
        h = NAME_MAP.get(ev["home_team"], ev["home_team"])
        a = NAME_MAP.get(ev["away_team"], ev["away_team"])
        if h not in tid or a not in tid:
            continue
        f = conn.execute(
            """SELECT id FROM fixtures WHERE home_team_id=? AND away_team_id=?
               AND status='scheduled'""", (tid[h], tid[a])).fetchone()
        if not f or not ev.get("bookmakers"):
            continue
        # median across bookmakers is robust to one bad price
        import statistics
        prices = {"H": [], "D": [], "A": []}
        for bk in ev["bookmakers"]:
            for mk in bk.get("markets", []):
                if mk["key"] != "h2h":
                    continue
                for o in mk["outcomes"]:
                    if o["name"] == ev["home_team"]:
                        prices["H"].append(o["price"])
                    elif o["name"] == ev["away_team"]:
                        prices["A"].append(o["price"])
                    else:
                        prices["D"].append(o["price"])
        if not all(prices.values()):
            continue
        conn.execute(
            """INSERT OR REPLACE INTO market_odds
                 (fixture_id, ts, bookmaker, odds_home, odds_draw, odds_away)
               VALUES (?,?,?,?,?,?)""",
            (f["id"], now, f"median of {len(ev['bookmakers'])} books",
             statistics.median(prices["H"]), statistics.median(prices["D"]),
             statistics.median(prices["A"])))
        n += 1
    conn.commit()
    log_pull(conn, "odds", True, f"{n} fixtures priced")
    print(f"odds stored for {n} fixtures")


if __name__ == "__main__":
    main()
