"""Pull the current season's fixture list (with gameweeks and kickoff times)
from the official Premier League Pulselive API, and mark played fixtures.

This replaced football-data.co.uk's fixtures.csv (FortiGuard-blocked on this
network) and understat (2026/27 not yet published there).

Promoted teams not in the seed list are inserted; if their stadium
coordinates are in PROMOTED_COORDS they get real coords, otherwise NULL and
travel adjustments for their home games are flagged partial.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from net import session
from backend.db import connect, log_pull

CURRENT_SEASON = "2627"

# pulselive club name -> our teams.name (football-data spelling)
PULSE_TO_LOCAL = {
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa",
    "AFC Bournemouth": "Bournemouth", "Bournemouth": "Bournemouth",
    "Brentford": "Brentford", "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Burnley": "Burnley", "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace", "Everton": "Everton",
    "Fulham": "Fulham", "Ipswich Town": "Ipswich", "Leeds United": "Leeds",
    "Leicester City": "Leicester", "Liverpool": "Liverpool",
    "Luton Town": "Luton", "Manchester City": "Man City",
    "Manchester United": "Man United", "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest", "Sheffield United": "Sheffield United",
    "Southampton": "Southampton", "Sunderland": "Sunderland",
    "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}
# stadium coords for plausible promoted sides (extend as needed)
PROMOTED_COORDS = {
    "Coventry City": (52.4481, -1.4956),      # CBS Arena
    "West Bromwich Albion": (52.5090, -1.9639),
    "Middlesbrough": (54.5781, -1.2166),
    "Norwich City": (52.6220, 1.3091),
    "Watford": (51.6499, -0.4015),
    "Hull City": (53.7466, -0.3675),
    "Stoke City": (52.9884, -2.1754),
    "Blackburn Rovers": (53.7286, -2.4894),
    "Bristol City": (51.4400, -2.6208),
    "Millwall": (51.4859, -0.0510),
    "Preston North End": (53.7722, -2.6882),
    "Charlton Athletic": (51.4865,  0.0364),
    "Wrexham": (53.0518, -3.0036),
    "Birmingham City": (52.4756, -1.8683),
    "Sheffield Wednesday": (53.4115, -1.5006),
    "Portsmouth": (50.7963, -1.0639),
    "Derby County": (52.9150, -1.4472),
    "Queens Park Rangers": (51.5093, -0.2321),
    "Swansea City": (51.6422, -3.9351),
    "Cardiff City": (51.4728, -3.2030),
}

HTTP = session()
HTTP.headers["User-Agent"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/126.0.0.0 Safari/537.36")
HTTP.headers["Origin"] = "https://www.premierleague.com"
API = "https://footballapi.pulselive.com/football"


def team_id_local(conn, pulse_name: str) -> int:
    local = PULSE_TO_LOCAL.get(pulse_name, pulse_name)
    row = conn.execute("SELECT id FROM teams WHERE name=?", (local,)).fetchone()
    if row:
        return row["id"]
    coords = PROMOTED_COORDS.get(pulse_name, (None, None))
    cur = conn.execute(
        "INSERT INTO teams (name, understat_name, lat, lon) VALUES (?,?,?,?)",
        (local, pulse_name, *coords))
    note = "with" if coords[0] else "WITHOUT"
    print(f"  new team '{local}' inserted {note} stadium coords")
    return cur.lastrowid


def pull(conn) -> int:
    import datetime
    fixtures, page = [], 0
    while True:
        r = HTTP.get(f"{API}/fixtures", params={
            "comps": 1, "compSeasons": "841", "pageSize": 100, "page": page,
            "sort": "asc"}, timeout=30)
        r.raise_for_status()
        d = r.json()
        fixtures += d["content"]
        if page >= d["pageInfo"]["numPages"] - 1:
            break
        page += 1

    n = 0
    for f in fixtures:
        home, away = (t["team"]["club"]["name"] for t in f["teams"])
        hid, aid = team_id_local(conn, home), team_id_local(conn, away)
        ko = f.get("kickoff", {}).get("millis")
        dt = (datetime.datetime.fromtimestamp(ko / 1000, datetime.UTC)
              .strftime("%Y-%m-%d %H:%M:%S") if ko else None)
        gw = f.get("gameweek", {}).get("gameweek")
        status = "played" if f.get("status") == "C" else "scheduled"
        conn.execute(
            """INSERT INTO fixtures (season, gameweek, date, home_team_id,
                 away_team_id, pulse_id, status)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(season, home_team_id, away_team_id) DO UPDATE SET
                 gameweek=excluded.gameweek, date=excluded.date,
                 pulse_id=excluded.pulse_id, status=excluded.status""",
            (CURRENT_SEASON, int(gw) if gw else None, dt, hid, aid,
             int(f["id"]), status))
        n += 1
    conn.commit()
    return n


def main():
    conn = connect()
    try:
        n = pull(conn)
        log_pull(conn, "pulselive fixtures", True, f"{n} fixtures")
        print(f"fixtures upserted: {n}")
    except Exception as e:
        log_pull(conn, "pulselive fixtures", False, str(e))
        print(f"fixture pull FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
