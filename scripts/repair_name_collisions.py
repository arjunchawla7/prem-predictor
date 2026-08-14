"""Undo squad rows that a loose name match bound to the wrong person.

find_player's fallbacks used to guess across clubs, so a squad member could be
matched onto an existing player who merely shared a surname or whose name was a
token subset. That does not create a duplicate — it overwrites the real
player's club and logs a transfer that never happened, then bounces it back and
forth on every subsequent sync.

Signature of a bad bind: our stored name is a STRICT token subset of the live
squad name the pulse_id belongs to (ours 'Gabriel', theirs 'Joseph Gabriel'),
and the player's recorded minutes are overwhelmingly for a different club than
the one they have now been assigned to.

Repair: unbind the pulse_id, restore the club its appearance record supports,
and delete the phantom transfers. The next sync then matches it properly,
because the fallbacks are now fenced to one club.
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from net import session
from backend.db import connect, log_pull
from pull_squads import norm, COMP_SEASON, API

HTTP = session()
HEADERS = {"Origin": "https://www.premierleague.com"}


def live_names(conn):
    """pulse_id -> display name, across every current squad."""
    out = {}
    for tid, name, pid in conn.execute(
            """SELECT id, name, pulse_id FROM teams WHERE pulse_id IS NOT NULL
               AND id IN (SELECT home_team_id FROM fixtures WHERE season='2627')"""):
        r = HTTP.get(f"{API}/teams/{pid}/compseasons/{COMP_SEASON}/staff",
                     headers=HEADERS, timeout=40)
        r.raise_for_status()
        for p in r.json().get("players", []):
            key = int(p.get("playerId") or p.get("id"))
            nm = (p.get("name") or {}).get("display")
            if nm:
                out[key] = (nm, tid, name)
    return out


def main():
    conn = connect()
    live = live_names(conn)
    suspects = []

    for r in conn.execute(
            """SELECT id, name, team_id, pulse_id, understat_id
               FROM players WHERE pulse_id IS NOT NULL"""):
        entry = live.get(r["pulse_id"])
        if not entry:
            continue
        their_name, their_team, their_team_name = entry
        ours, theirs = set(norm(r["name"]).split()), set(norm(their_name).split())
        if not (ours < theirs):          # strict subset only
            continue
        # where do this player's actual appearances say they belong?
        mins = Counter()
        for t, m in conn.execute(
                """SELECT team_id, SUM(minutes) FROM player_match_minutes
                   WHERE player_id=? GROUP BY team_id""", (r["id"],)):
            mins[t] = m or 0
        if not mins:
            home = None
        else:
            home = mins.most_common(1)[0][0]
        if home is not None and home != r["team_id"]:
            suspects.append((r, their_name, their_team_name, home, sum(mins.values())))

    if not suspects:
        print("no name-collision rows found")
        log_pull(conn, "repair collisions", True, "none found")
        return

    print(f"{len(suspects)} mis-bound row(s):\n")
    for r, their_name, their_team, home_team, total in suspects:
        home_name = conn.execute("SELECT name FROM teams WHERE id=?",
                                 (home_team,)).fetchone()["name"]
        cur = conn.execute("SELECT name FROM teams WHERE id=?",
                           (r["team_id"],)).fetchone()
        print(f"  '{r['name']}' (id {r['id']}, {total} recorded minutes for "
              f"{home_name})")
        print(f"     bound to pulse {r['pulse_id']} = '{their_name}' of "
              f"{their_team}, and moved to {cur['name'] if cur else '?'}")
        conn.execute(
            "UPDATE players SET pulse_id=NULL, team_id=?, in_current_squad=0 "
            "WHERE id=?", (home_team, r["id"]))
        n = conn.execute("DELETE FROM transfers WHERE player_id=?",
                         (r["id"],)).rowcount
        print(f"     -> restored to {home_name}, pulse_id cleared, "
              f"{n} phantom transfer(s) removed")
    conn.commit()
    log_pull(conn, "repair collisions", True, f"{len(suspects)} rows repaired")
    print("\nRe-run scripts/pull_squads.py to re-match them correctly.")


if __name__ == "__main__":
    main()
