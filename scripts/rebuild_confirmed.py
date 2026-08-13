"""Mode 2 — confirmed-lineup rebuild. Run manually ~1h before kickoff.

Fetches confirmed starting XIs from the official Premier League API
(teamLists populate shortly before kickoff), matches players by normalised
name against our understat-based player table, regenerates predictions
tagged 'final' (provisional rows are kept, not overwritten), and notes any
meaningful shift vs the latest provisional prediction.

Fixtures locked to manual mode are skipped. Unmatched player names are
listed rather than silently dropped; a lineup with <11 matched players
falls back to season-average weighting and flags partial data.

Usage: python scripts/rebuild_confirmed.py [gameweek]   (default: next GW)
"""
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from net import session
from backend.db import connect, log_pull
from backend.predict import Predictor, next_gameweek, CURRENT_SEASON

HTTP = session()
HTTP.headers["User-Agent"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/126.0.0.0 Safari/537.36")
HTTP.headers["Origin"] = "https://www.premierleague.com"


from pull_squads import norm  # shared accent-folding name normaliser


def match_players(conn, team_id, pulse_names):
    """Map pulselive player names -> our player ids for one team.
    Match by full normalised name (team-scoped, then global), then by
    team-scoped unique last name. Returns (ids, unmatched_names)."""
    ids, unmatched = [], []
    team_rows = conn.execute(
        """SELECT DISTINCT p.id, p.name FROM players p
           LEFT JOIN player_match_minutes pmm ON pmm.player_id = p.id
           WHERE p.team_id = ? OR pmm.team_id = ?""",
        (team_id, team_id)).fetchall()
    by_full = {}
    by_last = {}
    for r in team_rows:
        n = norm(r["name"])
        by_full[n] = r["id"]
        last = n.split()[-1]
        by_last.setdefault(last, set()).add(r["id"])
    for name in pulse_names:
        n = norm(name)
        if n in by_full:
            ids.append(by_full[n])
            continue
        g = conn.execute("SELECT id FROM players WHERE lower(name)=?",
                         (n,)).fetchall()
        if len(g) == 1:
            ids.append(g[0]["id"])
            continue
        cand = by_last.get(n.split()[-1], set())
        if len(cand) == 1:
            ids.append(next(iter(cand)))
        else:
            unmatched.append(name)
    return ids, unmatched


def confirmed_xi(fixture_detail, side_index):
    """Extract starting-XI names from a pulselive teamLists entry."""
    tl = fixture_detail.get("teamLists") or []
    if len(tl) <= side_index or not tl[side_index]:
        return None
    entry = tl[side_index]
    lineup = entry.get("lineup") or []
    names = []
    for p in lineup:
        nm = p.get("name", {})
        display = nm.get("display") or " ".join(
            x for x in (nm.get("first"), nm.get("last")) if x)
        if display:
            names.append(display)
    return names or None


def main():
    conn = connect()
    gw = int(sys.argv[1]) if len(sys.argv) > 1 else next_gameweek(conn)
    fixtures = conn.execute(
        """SELECT f.*, h.name hn, a.name an FROM fixtures f
           JOIN teams h ON h.id=f.home_team_id
           JOIN teams a ON a.id=f.away_team_id
           WHERE f.season=? AND f.gameweek=? ORDER BY f.date""",
        (CURRENT_SEASON, gw)).fetchall()
    if not fixtures:
        print(f"no fixtures for gameweek {gw}")
        return
    print(f"Mode 2 rebuild, gameweek {gw} ({len(fixtures)} fixtures)")
    predictor = None

    for f in fixtures:
        label = f"{f['hn']} v {f['an']}"
        if f["lineup_mode"] == "manual":
            print(f"  {label}: manual mode — skipped")
            continue
        if not f["pulse_id"]:
            print(f"  {label}: no pulselive id — skipped")
            continue
        try:
            r = HTTP.get("https://footballapi.pulselive.com/football/"
                         f"fixtures/{f['pulse_id']}", timeout=30)
            r.raise_for_status()
            det = r.json()
        except Exception as e:
            log_pull(conn, f"pulselive teamlist {f['pulse_id']}", False, str(e))
            print(f"  {label}: fetch FAILED ({e})")
            continue

        lineups, problems = {}, []
        for side, team_id in ((0, f["home_team_id"]), (1, f["away_team_id"])):
            names = confirmed_xi(det, side)
            if not names:
                problems.append("lineup not announced yet")
                continue
            ids, unmatched = match_players(conn, team_id, names[:11])
            if unmatched:
                problems.append(f"unmatched players: {', '.join(unmatched)}")
            lineups[side] = ids if len(ids) == 11 else None
        if not lineups.get(0) and not lineups.get(1):
            print(f"  {label}: {'; '.join(problems) or 'no lineups'} — skipped")
            continue

        if predictor is None:
            predictor = Predictor(conn)
        prev = conn.execute(
            """SELECT home_xg, away_xg FROM predictions
               WHERE fixture_id=? AND tag='provisional'
               ORDER BY created_at DESC LIMIT 1""", (f["id"],)).fetchone()
        pred = predictor.predict_fixture(
            f["id"], tag="final",
            home_lineup=lineups.get(0), away_lineup=lineups.get(1))
        note = "; ".join(problems)
        if prev:
            dh = pred["home_xg"] - prev["home_xg"]
            da = pred["away_xg"] - prev["away_xg"]
            if abs(dh) >= 0.1 or abs(da) >= 0.1:
                note = (f"confirmed XI shifted xG: home {dh:+.2f}, "
                        f"away {da:+.2f}" + (f"; {note}" if note else ""))
        if note:
            conn.execute(
                "UPDATE predictions SET notes = notes || ? WHERE id = "
                "(SELECT MAX(id) FROM predictions WHERE fixture_id=?)",
                (("; " if note else "") + note, f["id"]))
            conn.commit()
        print(f"  {label}: final {pred['p_home']:.0%}/{pred['p_draw']:.0%}/"
              f"{pred['p_away']:.0%}" + (f"  [{note}]" if note else ""))


if __name__ == "__main__":
    main()
