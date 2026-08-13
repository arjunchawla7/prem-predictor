"""Pull understat.com data into SQLite.

Understat season labels use the START year: understat 2025 == our '2526'.

Three jobs (all idempotent, raw JSON cached under data/raw/understat/):
  1. League data per season (getLeagueData/EPL/{year}) —
       - per-match xG onto matches (home_xg/away_xg/understat_id)
       - player season aggregates into player_season_stats
  2. Match rosters (getMatchData/{id}) for the player-data seasons —
       per-player minutes/goals/assists/started into player_match_minutes
  3. Current-season fixture list into fixtures (empty until understat opens
       the new season — logged as a failed pull, not silently skipped)

Endpoints need X-Requested-With: XMLHttpRequest or they 404.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from net import session
from backend.db import connect, log_pull

CACHE = ROOT / "data" / "raw" / "understat"
# understat start-year -> our season label
SEASONS = {"2022": "2223", "2023": "2324", "2024": "2425", "2025": "2526"}
CURRENT_US, CURRENT_SEASON = "2026", "2627"
ROSTER_SEASONS = ["2024", "2025"]     # player-level data: last 2 seasons
DELAY = 0.35                          # be polite between uncached requests

HTTP = session()
HTTP.headers["X-Requested-With"] = "XMLHttpRequest"
HTTP.headers["User-Agent"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/126.0.0.0 Safari/537.36")


def get_json(path: str, cache_name: str, refresh: bool = False):
    f = CACHE / f"{cache_name}.json"
    if f.exists() and not refresh:
        return json.loads(f.read_text(encoding="utf-8"))
    r = HTTP.get(f"https://understat.com/{path}", timeout=30)
    r.raise_for_status()
    data = r.json()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(DELAY)
    return data


def team_id_by_us(conn, us_name: str):
    row = conn.execute("SELECT id FROM teams WHERE understat_name=?",
                       (us_name,)).fetchone()
    if row:
        return row["id"]
    # Promoted team not in the seed list: insert without coordinates.
    cur = conn.execute(
        "INSERT INTO teams (name, understat_name) VALUES (?,?)",
        (us_name, us_name))
    print(f"  note: unknown team '{us_name}' inserted without stadium coords")
    return cur.lastrowid


def pos_group(understat_pos: str) -> str:
    """'GK'->GK, 'D C'/'DC'->DEF, 'M...'->MID, 'F...'/'S'->FWD-ish.

    League payload positions look like 'F S', 'M C', 'GK'; the first
    non-'S' letter is the real position. Pure 'Sub' has no info -> None.
    """
    for tok in understat_pos.replace(",", " ").split():
        c = tok[0].upper()
        if tok.upper().startswith("GK"):
            return "GK"
        if c == "D":
            return "DEF"
        if c == "M":
            return "MID"
        if c == "F":
            return "FWD"
    return None


def load_league_season(conn, us_year: str, our_season: str, refresh=False):
    data = get_json(f"getLeagueData/EPL/{us_year}", f"league_{us_year}",
                    refresh=refresh)
    dates, players = data["dates"], data["players"]

    # --- xG onto matches (match on date + understat team names) ---
    matched = missing = 0
    for m in dates:
        if not m["isResult"]:
            continue
        hid = team_id_by_us(conn, m["h"]["title"])
        aid = team_id_by_us(conn, m["a"]["title"])
        day = m["datetime"][:10]
        cur = conn.execute(
            """UPDATE matches SET understat_id=?, home_xg=?, away_xg=?
               WHERE season=? AND home_team_id=? AND away_team_id=?
                 AND date BETWEEN date(?, '-1 day') AND date(?, '+1 day')""",
            (int(m["id"]), float(m["xG"]["h"]), float(m["xG"]["a"]),
             our_season, hid, aid, day, day))
        matched += cur.rowcount
        missing += 1 - cur.rowcount

    # --- player season aggregates ---
    for p in players:
        tid = team_id_by_us(conn, p["team_title"].split(",")[0])
        conn.execute(
            """INSERT INTO players (understat_id, name, team_id, position)
               VALUES (?,?,?,?)
               ON CONFLICT(understat_id) DO UPDATE SET
                 name=excluded.name, team_id=excluded.team_id,
                 position=COALESCE(excluded.position, players.position)""",
            (int(p["id"]), p["player_name"], tid, pos_group(p["position"])))
        pid = conn.execute("SELECT id FROM players WHERE understat_id=?",
                           (int(p["id"]),)).fetchone()["id"]
        conn.execute(
            """INSERT OR REPLACE INTO player_season_stats
                 (player_id, season, team_id, position, minutes, games, goals,
                  assists, xg, xa, npxg, shots, key_passes, xg_buildup, xg_chain)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (pid, our_season, tid, pos_group(p["position"]),
             int(p["time"]), int(p["games"]), int(p["goals"]),
             int(p["assists"]), float(p["xG"]), float(p["xA"]),
             float(p["npxG"]), int(p["shots"]), int(p["key_passes"]),
             float(p["xGBuildup"]), float(p["xGChain"])))
    conn.commit()
    print(f"{our_season}: xG matched {matched}, unmatched {missing}, "
          f"players {len(players)}")
    if missing:
        log_pull(conn, f"understat league {us_year}", False,
                 f"{missing} matches had no DB row to attach xG")


def load_rosters(conn, us_year: str, our_season: str):
    rows = conn.execute(
        "SELECT id, understat_id FROM matches WHERE season=? AND understat_id "
        "IS NOT NULL", (our_season,)).fetchall()
    done = 0
    for row in rows:
        data = get_json(f"getMatchData/{row['understat_id']}",
                        f"match_{row['understat_id']}")
        for side in ("h", "a"):
            for entry in data["rosters"][side].values():
                us_pid = int(entry["player_id"])
                conn.execute(
                    """INSERT INTO players (understat_id, name, position)
                       VALUES (?,?,?)
                       ON CONFLICT(understat_id) DO NOTHING""",
                    (us_pid, entry["player"], pos_group(entry["position"])))
                pid = conn.execute(
                    "SELECT id FROM players WHERE understat_id=?",
                    (us_pid,)).fetchone()["id"]
                tid_col = "home_team_id" if side == "h" else "away_team_id"
                tid = conn.execute(
                    f"SELECT {tid_col} AS t FROM matches WHERE id=?",
                    (row["id"],)).fetchone()["t"]
                started = 1 if entry.get("roster_in") == "0" else 0
                conn.execute(
                    """INSERT OR REPLACE INTO player_match_minutes
                         (match_id, player_id, team_id, minutes, goals,
                          assists, started)
                       VALUES (?,?,?,?,?,?,?)""",
                    (row["id"], pid, tid, int(entry["time"]),
                     int(entry["goals"]), int(entry["assists"]), started))
        done += 1
        if done % 50 == 0:
            conn.commit()
            print(f"  rosters {our_season}: {done}/{len(rows)}")
    conn.commit()
    print(f"rosters {our_season}: {done} matches loaded")


def load_current_fixtures(conn):
    try:
        data = get_json(f"getLeagueData/EPL/{CURRENT_US}",
                        f"league_{CURRENT_US}", refresh=True)
    except Exception as e:
        log_pull(conn, "understat current fixtures", False, str(e))
        print(f"current-season fixture pull FAILED: {e}")
        return
    dates = data["dates"]
    if not dates:
        log_pull(conn, "understat current fixtures", False,
                 "understat 2026/27 season not yet published")
        print("understat has no 2026/27 fixtures yet — logged, nothing loaded")
        return
    for m in dates:
        hid = team_id_by_us(conn, m["h"]["title"])
        aid = team_id_by_us(conn, m["a"]["title"])
        conn.execute(
            """INSERT INTO fixtures (season, date, home_team_id, away_team_id,
                 understat_id, status)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(season, home_team_id, away_team_id) DO UPDATE SET
                 date=excluded.date, status=excluded.status,
                 understat_id=excluded.understat_id""",
            (CURRENT_SEASON, m["datetime"], hid, aid, int(m["id"]),
             "played" if m["isResult"] else "scheduled"))
    conn.commit()
    log_pull(conn, "understat current fixtures", True, f"{len(dates)} fixtures")
    print(f"current season fixtures: {len(dates)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rosters", action="store_true")
    ap.add_argument("--refresh-league", action="store_true",
                    help="re-fetch league JSON instead of using cache")
    args = ap.parse_args()

    conn = connect()
    for us_year, our in SEASONS.items():
        load_league_season(conn, us_year, our, refresh=args.refresh_league)
    if not args.skip_rosters:
        for us_year in ROSTER_SEASONS:
            load_rosters(conn, us_year, SEASONS[us_year])
    load_current_fixtures(conn)


if __name__ == "__main__":
    main()
