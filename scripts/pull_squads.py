"""Squad sync — registered 2026-27 squads from the official PL Pulselive API.

For each of the 20 current clubs (footballapi.pulselive.com — NOT the
sdp-prem-prod host, which 400s on this network):
  GET /football/teams/{id}/compseasons/841/staff  -> players[]

Effects:
  - players get pulse_id, shirt_num, position, current team_id and
    in_current_squad=1 (everyone else is flagged 0 first)
  - existing players are matched by pulse_id, then by normalised name
    (team-scoped last-name fallback); unmatched squad members are inserted
    as NEW players — new signings with no understat history, which the
    rating system already treats as flagged provisional (league-average)
  - a detected club change is recorded in `transfers` (from NULL = new to
    the league). Departures (in DB at a PL club, absent from every current
    squad) are recorded with to_team_id NULL once, then team_id cleared.

Run as part of the weekly refresh or standalone.
"""
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from net import session
from backend.db import connect, log_pull
from pull_fixtures import PULSE_TO_LOCAL

API = "https://footballapi.pulselive.com/football"
COMP_SEASON = "841"     # 2026/27
POS = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}


def detail_pos(position_info: str) -> str:
    """Compress pulselive positionInfo ('Left/Centre/Right Striker',
    'Centre Defensive Midfielder', ...) to a familiar shorthand."""
    if not position_info:
        return None
    pi = position_info.lower()
    left, right = "left" in pi, "right" in pi
    if "goalkeeper" in pi:
        return "GK"
    if "full back" in pi or "wing back" in pi:
        return "LB" if left and not right else "RB" if right and not left else "FB"
    if "defender" in pi:
        return "CB"
    if "defensive midfielder" in pi:
        return "CDM"
    if "attacking midfielder" in pi:
        return "CAM"
    if "winger" in pi:
        return "LW" if left and not right else "RW" if right and not left else "W"
    if "second striker" in pi:
        return "SS"
    if "striker" in pi:
        return "ST"
    if "midfielder" in pi:
        return "CM"
    return None

HTTP = session()
HTTP.headers["User-Agent"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/126.0.0.0 Safari/537.36")
HTTP.headers["Origin"] = "https://www.premierleague.com"


_LETTER_MAP = str.maketrans({
    "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "å": "a", "Å": "A",
    "ð": "d", "Ð": "D", "þ": "th", "Þ": "Th", "ł": "l", "Ł": "L",
    "đ": "d", "Đ": "D", "ß": "ss", "ı": "i",
})


def norm(name: str) -> str:
    """Accent-fold for name matching. NFKD strips combining accents (é->e)
    but NOT standalone letters like Ø/ł/ß — those need the explicit map,
    otherwise 'Martin Ødegaard' (PL) never matches 'Martin Odegaard'
    (understat) and a duplicate player row gets created."""
    s = unicodedata.normalize("NFKD", name.translate(_LETTER_MAP))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def display_name(p) -> str:
    nm = p.get("name") or {}
    return (nm.get("display")
            or " ".join(x for x in (nm.get("first"), nm.get("last")) if x))


def find_player(conn, pulse_id, name, team_id):
    """Locate an existing row for this squad member. Returns id or None.

    Every fallback below is a guess, and a wrong guess is expensive: it does
    not create a duplicate, it OVERWRITES a real player's club and logs a
    transfer that never happened. So each one is fenced by two rules —

      1. never adopt a row that already carries a DIFFERENT pulse_id. The
         official feed has already identified that row as somebody else.
      2. loose name matching stays inside the club being synced.

    Rule 2 is what was missing from the token-subset fallback, and it cost
    Arsenal their centre-half: 'Joseph Gabriel' of Man United tokenises to
    {joseph, gabriel}, which contains {gabriel}, so it matched Arsenal's
    Gabriel Magalhães across clubs, moved him to Man United and recorded a
    bogus transfer that then bounced back and forth on every sync.
    """
    r = conn.execute("SELECT id FROM players WHERE pulse_id=?",
                     (pulse_id,)).fetchone()
    if r:
        return r["id"]
    n = norm(name)
    rows = conn.execute(
        "SELECT id, name, team_id, pulse_id FROM players").fetchall()
    # a row already bound to another pulse id is a different, known person
    free = [x for x in rows if x["pulse_id"] is None]

    full = [x for x in free if norm(x["name"]) == n]
    if len(full) == 1:
        return full[0]["id"]
    if len(full) > 1:   # same name twice: prefer same club
        same = [x for x in full if x["team_id"] == team_id]
        return (same or full)[0]["id"]
    # last-name fallback, only within the same club's known players
    last = n.split()[-1]
    cand = [x for x in free if x["team_id"] == team_id
            and norm(x["name"]).split()[-1] == last]
    if len(cand) == 1:
        return cand[0]["id"]
    # token-subset fallback for understat mononyms: 'Gabriel' should match
    # 'Gabriel Magalhães', 'Casemiro' should match 'Carlos Henrique Casemiro'.
    # Same club only — see the docstring for what happens without that.
    toks = set(n.split())
    cand = [x for x in free
            if x["team_id"] == team_id
            and set(norm(x["name"]).split()) <= toks
            and set(norm(x["name"]).split()) != toks]
    if len(cand) == 1:
        return cand[0]["id"]
    return None


def main():
    conn = connect()
    now = datetime.now(timezone.utc).isoformat()

    r = HTTP.get(f"{API}/teams", params={"comps": 1, "compSeasons": COMP_SEASON,
                                         "pageSize": 50}, timeout=30)
    r.raise_for_status()
    clubs = r.json()["content"]

    conn.execute("UPDATE players SET in_current_squad=0")
    new_signings, moves, total = 0, 0, 0

    for c in clubs:
        pulse_team = int(c["id"])
        local = PULSE_TO_LOCAL.get(c["club"]["name"], c["club"]["name"])
        t = conn.execute("SELECT id FROM teams WHERE name=?", (local,)).fetchone()
        if not t:
            print(f"  ! no local team for '{c['club']['name']}' — skipped")
            continue
        team_id = t["id"]
        conn.execute("UPDATE teams SET pulse_id=? WHERE id=?",
                     (pulse_team, team_id))

        sr = HTTP.get(f"{API}/teams/{pulse_team}/compseasons/{COMP_SEASON}/staff",
                      params={"altIds": "true"}, timeout=30)
        sr.raise_for_status()
        players = sr.json().get("players", [])
        for p in players:
            name = display_name(p)
            if not name:
                continue
            pulse_pid = int(p.get("playerId") or p.get("id"))
            info = p.get("info") or {}
            pos = POS.get(info.get("position"), None)
            dpos = detail_pos(info.get("positionInfo"))
            shirt = info.get("shirtNum")
            opta = (p.get("altIds") or {}).get("opta")
            pid = find_player(conn, pulse_pid, name, team_id)
            if pid is None:
                conn.execute(
                    """INSERT INTO players (name, team_id, position, pulse_id,
                         shirt_num, detail_pos, opta_code, in_current_squad)
                       VALUES (?,?,?,?,?,?,?,1)""",
                    (name, team_id, pos, pulse_pid, shirt, dpos, opta))
                conn.execute(
                    """INSERT INTO transfers (player_id, detected_at,
                         from_team_id, to_team_id)
                       VALUES ((SELECT id FROM players WHERE pulse_id=?),
                               ?, NULL, ?)""", (pulse_pid, now, team_id))
                new_signings += 1
            else:
                old = conn.execute("SELECT team_id FROM players WHERE id=?",
                                   (pid,)).fetchone()["team_id"]
                if old is not None and old != team_id:
                    conn.execute(
                        """INSERT INTO transfers (player_id, detected_at,
                             from_team_id, to_team_id) VALUES (?,?,?,?)""",
                        (pid, now, old, team_id))
                    moves += 1
                conn.execute(
                    """UPDATE players SET team_id=?, pulse_id=?, shirt_num=?,
                         detail_pos=?, opta_code=?, in_current_squad=1,
                         position=COALESCE(?, position)
                       WHERE id=?""",
                    (team_id, pulse_pid, shirt, dpos, opta, pos, pid))
            total += 1

    # departures: still attached to a current PL club but in no squad list
    current_clubs = [r["id"] for r in conn.execute(
        "SELECT DISTINCT home_team_id AS id FROM fixtures WHERE season='2627'")]
    qmarks = ",".join("?" * len(current_clubs))
    gone = conn.execute(
        f"""SELECT id, team_id FROM players
            WHERE in_current_squad=0 AND team_id IN ({qmarks})
              AND id NOT IN (SELECT player_id FROM transfers
                             WHERE to_team_id IS NULL)""",
        current_clubs).fetchall()
    for g in gone:
        conn.execute(
            """INSERT INTO transfers (player_id, detected_at, from_team_id,
                 to_team_id) VALUES (?,?,?,NULL)""",
            (g["id"], now, g["team_id"]))
    conn.commit()

    log_pull(conn, "squads", True,
             f"{total} squad members, {new_signings} new, {moves} moved, "
             f"{len(gone)} departed")
    print(f"squads synced: {total} players across {len(clubs)} clubs")
    print(f"  new signings (no PL history here): {new_signings}")
    print(f"  club changes detected: {moves}")
    print(f"  departures flagged: {len(gone)}")


if __name__ == "__main__":
    main()
