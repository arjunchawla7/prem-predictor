"""Pull EPL match odds from the-odds-api.com, repeatedly, as kickoff nears.

Needs an API key in the ODDS_API_KEY environment variable (register free at
https://the-odds-api.com). Without a key this logs the gap and exits 0 so
the weekly refresh keeps working — the frontend then shows "no market odds".

WHY IT RUNS MORE THAN ONCE
A single early snapshot is the market's opinion days before team news. Prices
move on injuries, confirmed XIs and money; the blend is only worth having if it
tracks that. One /odds call prices EVERY upcoming EPL fixture at once, so a
refresh costs the same whether one match or ten are pending — cadence is cheap,
which is what makes this practical on the free tier at all.

CREDITS
Cost per call = [markets] × [regions]; this asks for h2h across ODDS_REGIONS
(default uk,eu) = 2 credits. The free tier is 500 credits/month, so roughly 250
refreshes — about 8 a day, comfortably more than needed. Set ODDS_REGIONS=uk to
halve it. Remaining quota comes back in the x-requests-remaining header and is
logged after every call; --reserve refuses to spend the last of it.

The guards below mean this can be scheduled hourly and will simply do nothing
(zero credits, no HTTP request) outside the hours before a kickoff:

  --window-hours H     only call when a fixture kicks off within H hours
  --min-interval M     skip if the newest snapshot is younger than M minutes
  --reserve N          skip when fewer than N credits are known to remain
  --force              ignore all three

Every pull inserts a new row in market_odds (PK fixture_id + ts), so the price
history accumulates and movement is reported against the previous snapshot.
Implied probabilities and the model+market blend are derived at read time in
backend/market.py.
"""
import argparse
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from net import session
from backend.db import connect, log_pull
from backend.market import implied_probs
from backend.predict import CURRENT_SEASON

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
REGIONS = os.environ.get("ODDS_REGIONS", "uk,eu")
QUOTA_TAG = "odds quota"       # pull_log source used to remember credits left


def _now():
    return datetime.now(timezone.utc)


def hours_to_next_kickoff(conn):
    """Hours until the next scheduled kickoff, or None if none upcoming."""
    row = conn.execute(
        """SELECT MIN(date) d FROM fixtures
           WHERE season=? AND status='scheduled' AND date IS NOT NULL""",
        (CURRENT_SEASON,)).fetchone()
    if not row or not row["d"]:
        return None
    try:
        ko = datetime.fromisoformat(str(row["d"]).replace(" ", "T"))
    except ValueError:
        return None
    if ko.tzinfo is None:
        ko = ko.replace(tzinfo=timezone.utc)
    return (ko - _now()).total_seconds() / 3600


def minutes_since_last_snapshot(conn):
    row = conn.execute("SELECT MAX(ts) t FROM market_odds").fetchone()
    if not row or not row["t"]:
        return None
    try:
        t = datetime.fromisoformat(row["t"])
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (_now() - t).total_seconds() / 60


def credits_remaining(conn):
    """Last known x-requests-remaining, from the previous call's log line."""
    row = conn.execute(
        """SELECT detail FROM pull_log WHERE source=? ORDER BY ts DESC LIMIT 1""",
        (QUOTA_TAG,)).fetchone()
    if not row or not row["detail"]:
        return None
    try:
        return int(str(row["detail"]).split()[0])
    except (ValueError, IndexError):
        return None


def should_skip(conn, args):
    """Reason to skip, or None to go ahead. Runs before any HTTP request."""
    if args.force:
        return None
    left = credits_remaining(conn)
    if left is not None and left < args.reserve:
        return (f"only {left} API credits known to remain "
                f"(reserve {args.reserve}) — not spending them")
    mins = minutes_since_last_snapshot(conn)
    if mins is not None and mins < args.min_interval:
        return (f"last snapshot {mins:.0f} min old "
                f"(min interval {args.min_interval} min)")
    h = hours_to_next_kickoff(conn)
    if h is None:
        return "no scheduled fixtures"
    if h > args.window_hours:
        return (f"next kickoff is {h:.0f}h away "
                f"(refresh window {args.window_hours}h)")
    if h < -3:
        return "next scheduled fixture is in the past — fixture list looks stale"
    return None


def median_prices(ev):
    """Median h2h price per outcome across books — robust to one bad line."""
    prices = {"H": [], "D": [], "A": []}
    for bk in ev.get("bookmakers", []):
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
        return None
    return (statistics.median(prices["H"]), statistics.median(prices["D"]),
            statistics.median(prices["A"]))


def previous_snapshot(conn, fixture_id):
    return conn.execute(
        """SELECT ts, odds_home, odds_draw, odds_away FROM market_odds
           WHERE fixture_id=? ORDER BY ts DESC LIMIT 1""",
        (fixture_id,)).fetchone()


def describe_move(prev, new):
    """Movement as a home-probability shift; None when nothing moved."""
    if not prev:
        return None
    before = implied_probs(prev["odds_home"], prev["odds_draw"],
                           prev["odds_away"])
    after = implied_probs(*new)
    if not before or not after:
        return None
    d = [a - b for a, b in zip(after, before)]
    if max(abs(x) for x in d) < 0.005:      # half a point: noise, not news
        return None
    return f"{d[0]:+.1%}/{d[1]:+.1%}/{d[2]:+.1%}"


def pull(conn, args):
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        log_pull(conn, "odds", False, "ODDS_API_KEY not set — market odds skipped")
        print("no ODDS_API_KEY set — market odds skipped (register free at "
              "the-odds-api.com, then setx ODDS_API_KEY <key>)")
        return 0

    skip = should_skip(conn, args)
    if skip:
        print(f"odds refresh skipped: {skip}")
        return 0

    s = session()
    try:
        r = s.get(API, params={"apiKey": key, "regions": REGIONS,
                               "markets": "h2h", "oddsFormat": "decimal"},
                  timeout=30)
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        log_pull(conn, "odds", False, str(e))
        print(f"odds pull FAILED: {e}")
        sys.exit(1)

    # Remember the quota so the next run can refuse to spend the last of it.
    left = r.headers.get("x-requests-remaining")
    used = r.headers.get("x-requests-used")
    cost = r.headers.get("x-requests-last")
    if left is not None:
        log_pull(conn, QUOTA_TAG, True,
                 f"{left} remaining, {used} used, {cost} for this call "
                 f"(regions {REGIONS})")

    tid = {r_["name"]: r_["id"] for r_ in
           conn.execute("SELECT id, name FROM teams")}
    now = _now().isoformat()
    n, moved, comparable = 0, [], 0
    for ev in events:
        h = NAME_MAP.get(ev["home_team"], ev["home_team"])
        a = NAME_MAP.get(ev["away_team"], ev["away_team"])
        if h not in tid or a not in tid:
            continue
        f = conn.execute(
            """SELECT id FROM fixtures WHERE home_team_id=? AND away_team_id=?
               AND status='scheduled'""", (tid[h], tid[a])).fetchone()
        if not f:
            continue
        prices = median_prices(ev)
        if not prices:
            continue
        prev = previous_snapshot(conn, f["id"])
        comparable += bool(prev)
        move = describe_move(prev, prices)
        conn.execute(
            """INSERT OR REPLACE INTO market_odds
                 (fixture_id, ts, bookmaker, odds_home, odds_draw, odds_away)
               VALUES (?,?,?,?,?,?)""",
            (f["id"], now, f"median of {len(ev['bookmakers'])} books", *prices))
        n += 1
        if move:
            moved.append(f"{h} v {a} {move}")
    conn.commit()
    detail = f"{n} fixtures priced"
    if left is not None:
        detail += f"; {left} credits left"
    if moved:
        detail += f"; moved: {', '.join(moved)}"
    log_pull(conn, "odds", True, detail)
    print(f"odds stored for {n} fixtures"
          + (f" ({left} credits left)" if left is not None else ""))
    for m in moved:
        print(f"  moved  {m}")
    if n and not moved:
        print("  opening snapshot — nothing to compare against yet"
              if not comparable else
              "  no meaningful movement since the previous snapshot")
    return n


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window-hours", type=float, default=72,
                    help="only refresh when a kickoff is this close (default 72)")
    ap.add_argument("--min-interval", type=float, default=180,
                    help="minutes between snapshots (default 180)")
    ap.add_argument("--reserve", type=int, default=25,
                    help="leave this many API credits unspent (default 25)")
    ap.add_argument("--force", action="store_true",
                    help="ignore the window, interval and reserve guards")
    return ap.parse_args(argv)


def main():
    pull(connect(), build_args())


if __name__ == "__main__":
    main()
