"""Weekly refresh — ONE command (run whenever, e.g. a few days pre-gameweek):

  1. pull new results (season CSVs via mirror)  -> matches
  2. pull understat xG + rosters for the current season's new matches
  3. pull/refresh the fixture list (kickoff moves, postponements)
  4. Mode 1 (ideal XI) provisional predictions for the next gameweek

Run: .venv\\Scripts\\python scripts\\refresh_week.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PY = sys.executable


def run(script, *args):
    print(f"\n=== {script} {' '.join(args)}")
    r = subprocess.run([PY, str(ROOT / "scripts" / script), *args])
    if r.returncode != 0:
        print(f"!! {script} failed (see data pull log) — continuing")
    return r.returncode == 0


def main():
    run("download_history.py")
    run("load_matches.py")
    run("pull_understat.py", "--refresh-league")
    run("pull_fixtures.py")
    run("pull_squads.py")
    run("pull_managers.py")
    run("backfill_slots.py")
    run("pull_odds.py")

    from backend.db import connect
    from backend.predict import Predictor, next_gameweek
    conn = connect()
    gw = next_gameweek(conn)
    if gw is None:
        print("no scheduled fixtures found")
        return
    print(f"\n=== Mode 1 provisional predictions, gameweek {gw}")
    p = Predictor(conn)
    for pred in p.predict_gameweek(gw):
        f = conn.execute(
            """SELECT h.name h, a.name a FROM fixtures f
               JOIN teams h ON h.id=f.home_team_id
               JOIN teams a ON a.id=f.away_team_id WHERE f.id=?""",
            (pred["fixture_id"],)).fetchone()
        star = " *partial data" if pred["partial_data"] else ""
        print(f"  {f['h']} v {f['a']}: {pred['p_home']:.0%}/"
              f"{pred['p_draw']:.0%}/{pred['p_away']:.0%} "
              f"xG {pred['home_xg']}-{pred['away_xg']}{star}")


if __name__ == "__main__":
    main()
