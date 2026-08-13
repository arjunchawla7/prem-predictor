"""Store each team's crest URL (official PL badge CDN) on the teams table.

Badge URL pattern: https://resources.premierleague.com/premierleague/badges/50/{opta}.png
where {opta} is the club's Opta id (e.g. t1 = Arsenal), obtained from the
Pulselive teams endpoint with altIds=true. Covers current + recent seasons'
clubs; teams without a match get NULL and the frontend shows a monogram.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from net import session
from backend.db import connect, log_pull
from pull_fixtures import PULSE_TO_LOCAL

HTTP = session()
HTTP.headers["User-Agent"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/126.0.0.0 Safari/537.36")
HTTP.headers["Origin"] = "https://www.premierleague.com"
BADGE = "https://resources.premierleague.com/premierleague/badges/50/{opta}.png"
# recent compSeasons so relegated clubs (Leicester, Luton, ...) get crests too
COMP_SEASONS = ["841", "719", "678", "578", "489"]


def main():
    conn = connect()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(teams)")}
    if "crest_url" not in cols:
        conn.execute("ALTER TABLE teams ADD COLUMN crest_url TEXT")

    seen = {}
    for cs in COMP_SEASONS:
        try:
            r = HTTP.get("https://footballapi.pulselive.com/football/teams",
                         params={"comps": 1, "compSeasons": cs,
                                 "pageSize": 50, "altIds": "true"}, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"compSeason {cs}: fetch failed ({e})")
            continue
        for t in r.json().get("content", []):
            name = t.get("club", {}).get("name") or t.get("name")
            opta = (t.get("altIds") or {}).get("opta")
            if name and opta and name not in seen:
                seen[name] = BADGE.format(opta=opta)

    n = 0
    for pulse_name, url in seen.items():
        local = PULSE_TO_LOCAL.get(pulse_name, pulse_name)
        cur = conn.execute("UPDATE teams SET crest_url=? WHERE name=?",
                           (url, local))
        n += cur.rowcount
    conn.commit()
    missing = [r["name"] for r in
               conn.execute("SELECT name FROM teams WHERE crest_url IS NULL")]
    log_pull(conn, "badges", True, f"{n} crests; missing: {missing}")
    print(f"crests set for {n} teams; missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
