"""Regenerate Mode 1 predictions for the next gameweek (e.g. after a squad
sync changed who's available for the ideal XI)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.db import connect
from backend.predict import Predictor, next_gameweek

conn = connect()
gw = int(sys.argv[1]) if len(sys.argv) > 1 else next_gameweek(conn)
p = Predictor(conn)
for pred in p.predict_gameweek(gw):
    f = conn.execute(
        """SELECT h.name h, a.name a FROM fixtures f
           JOIN teams h ON h.id=f.home_team_id
           JOIN teams a ON a.id=f.away_team_id WHERE f.id=?""",
        (pred["fixture_id"],)).fetchone()
    star = " *partial" if pred["partial_data"] else ""
    print(f"{f['h']:18} v {f['a']:18} {pred['p_home']:.0%}/{pred['p_draw']:.0%}/"
          f"{pred['p_away']:.0%}  xG {pred['home_xg']:.2f}-{pred['away_xg']:.2f}{star}")
