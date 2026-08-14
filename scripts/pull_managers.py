"""Pull the current manager of each club from the official PL API.

Same staff endpoint as the squad sync, but reads the `officials` list
instead of `players`, keeping the entry whose role is Manager or Head
Coach (assistant managers and coaches are ignored).

Stores the Opta code so the frontend can show the manager's photo from the
same CDN as the players.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from net import session
from backend.db import connect, log_pull
from pull_fixtures import PULSE_TO_LOCAL

API = "https://footballapi.pulselive.com/football"
COMP_SEASON = "841"
WANTED_ROLES = ("manager", "head coach")

# Clubs where the feed lists several active "Manager" officials with nothing to
# distinguish them. Keyed by our team name, valued by the display name of the
# actual first-team manager. The pull prints a note for any other club that
# starts returning more than one, so this list surfaces rather than rots.
FIRST_TEAM_BOSS = {
    "Chelsea": "Xabi Alonso",
}

HTTP = session()
HTTP.headers["User-Agent"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/126.0.0.0 Safari/537.36")
HTTP.headers["Origin"] = "https://www.premierleague.com"


def main():
    conn = connect()
    now = datetime.now(timezone.utc).isoformat()
    r = HTTP.get(f"{API}/teams", params={"comps": 1, "compSeasons": COMP_SEASON,
                                         "pageSize": 50}, timeout=30)
    r.raise_for_status()
    clubs = r.json()["content"]

    found, missing = 0, []
    for c in clubs:
        local = PULSE_TO_LOCAL.get(c["club"]["name"], c["club"]["name"])
        t = conn.execute("SELECT id FROM teams WHERE name=?", (local,)).fetchone()
        if not t:
            continue
        sr = HTTP.get(f"{API}/teams/{int(c['id'])}/compseasons/{COMP_SEASON}/staff",
                      params={"altIds": "true"}, timeout=30)
        sr.raise_for_status()
        officials = sr.json().get("officials", [])
        candidates = [o for o in officials
                      if (o.get("role") or "").lower() in WANTED_ROLES
                      and o.get("active", True)]
        # Some clubs list more than one active "Manager" and the feed gives no
        # way to tell the first-team boss from the rest — same role, same
        # active flag, no join date, nothing. Chelsea returns Calum McFarlane
        # ahead of Xabi Alonso, so taking the first entry silently picked the
        # wrong man. Rather than invent a tie-break that would guess wrong for
        # some other club, the real one is named here.
        pick = FIRST_TEAM_BOSS.get(local)
        boss = None
        if pick:
            boss = next((o for o in candidates
                         if (o.get("name") or {}).get("display") == pick), None)
            if boss is None and candidates:
                print(f"  note: {local} override '{pick}' not in the feed — "
                      f"using {(candidates[0].get('name') or {}).get('display')}")
        if boss is None:
            boss = candidates[0] if candidates else None
        if len(candidates) > 1 and not pick:
            names = ", ".join((o.get("name") or {}).get("display")
                              for o in candidates)
            print(f"  note: {local} lists {len(candidates)} managers ({names})"
                  f" — took the first; add to FIRST_TEAM_BOSS if wrong")
        if not boss:
            missing.append(local)
            continue
        birth = boss.get("birth") or {}
        conn.execute(
            """INSERT INTO managers (team_id, pulse_id, opta_code, name, role,
                 nationality, dob, updated_at)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(pulse_id) DO UPDATE SET
                 team_id=excluded.team_id, name=excluded.name,
                 role=excluded.role, opta_code=excluded.opta_code,
                 updated_at=excluded.updated_at""",
            (t["id"], int(boss["id"]), (boss.get("altIds") or {}).get("opta"),
             boss["name"]["display"], boss.get("role"),
             (birth.get("country") or {}).get("country"),
             ((birth.get("date") or {}).get("label")), now))
        found += 1
    # a club can only have one current manager: drop stale rows
    conn.execute("""DELETE FROM managers WHERE id NOT IN
                      (SELECT MAX(id) FROM managers GROUP BY team_id)""")
    conn.commit()
    log_pull(conn, "managers", True, f"{found} managers; missing: {missing}")
    print(f"managers: {found} clubs; missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
