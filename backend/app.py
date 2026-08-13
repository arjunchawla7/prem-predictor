"""Flask app — local web frontend.

Run:  .venv\\Scripts\\python backend\\app.py   ->  http://127.0.0.1:5000

Pages
  /                     gameweek view (?gw=N)
  /performance          backtest + provisional-vs-final tracking

JSON API
  GET  /api/gameweek/<gw>            fixtures + active prediction each
  GET  /api/squad/<team_id>          squad list for the lineup editor
  POST /api/fixture/<id>/manual      {"home": [11 ids] | null, "away": ...}
                                     -> manual prediction, locks fixture
  POST /api/fixture/<id>/reset       back to automatic mode
  GET  /api/performance              backtest metrics + prov-vs-final stats
"""
import json
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.db import connect
from backend.predict import Predictor, next_gameweek, CURRENT_SEASON
from models.fatigue import player_fatigue
from models.player_ratings import RatingBook

app = Flask(__name__, static_folder=str(ROOT / "frontend"))
_predictor = None


def predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor(connect())
    return _predictor


def db():
    return connect()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/performance")
def performance_page():
    return send_from_directory(app.static_folder, "performance.html")


def active_prediction(conn, fixture):
    """Which prediction the fixture shows: manual-locked -> latest manual;
    else latest final, else latest provisional."""
    order = (["manual"] if fixture["lineup_mode"] == "manual"
             else ["final", "provisional"])
    for tag in order:
        row = conn.execute(
            """SELECT * FROM predictions WHERE fixture_id=? AND tag=?
               ORDER BY created_at DESC LIMIT 1""",
            (fixture["id"], tag)).fetchone()
        if row:
            return row
    return None


def lineup_detail(conn, book, lineup_json, as_of):
    if not lineup_json:
        return None
    out = []
    for pid in json.loads(lineup_json):
        p = conn.execute("SELECT name, position FROM players WHERE id=?",
                         (pid,)).fetchone()
        r = book.by_id.get(pid)
        out.append({
            "id": pid, "name": p["name"] if p else f"#{pid}",
            "position": (p["position"] if p else None) or "?",
            "tier": r["tier"] if r else 3,
            "provisional": bool(r["provisional"]) if r else True,
            "fatigue": round(player_fatigue(conn, pid, as_of)),
        })
    return out


@app.route("/api/gameweek/<int:gw>")
def api_gameweek(gw):
    conn = db()
    book = predictor().book
    fixtures = conn.execute(
        """SELECT f.*, h.name hn, a.name an, h.id hid, a.id aid
           FROM fixtures f
           JOIN teams h ON h.id=f.home_team_id
           JOIN teams a ON a.id=f.away_team_id
           WHERE f.season=? AND f.gameweek=? ORDER BY f.date""",
        (CURRENT_SEASON, gw)).fetchall()
    out = []
    for f in fixtures:
        pred = active_prediction(conn, f)
        odds = conn.execute(
            """SELECT * FROM market_odds WHERE fixture_id=?
               ORDER BY ts DESC LIMIT 1""", (f["id"],)).fetchone()
        item = {
            "id": f["id"], "date": f["date"], "gameweek": f["gameweek"],
            "home": {"id": f["hid"], "name": f["hn"]},
            "away": {"id": f["aid"], "name": f["an"]},
            "status": f["status"], "lineup_mode": f["lineup_mode"],
            "prediction": None, "odds": None,
        }
        if pred:
            grid = json.loads(pred["score_grid"])
            flat = [(i, j, p) for i, row in enumerate(grid)
                    for j, p in enumerate(row)]
            top = sorted(flat, key=lambda x: -x[2])[:5]
            as_of = f["date"] or ""
            item["prediction"] = {
                "tag": pred["tag"], "created_at": pred["created_at"],
                "p_home": pred["p_home"], "p_draw": pred["p_draw"],
                "p_away": pred["p_away"],
                "home_xg": pred["home_xg"], "away_xg": pred["away_xg"],
                "top_scores": [{"score": f"{i}-{j}", "p": round(p, 4)}
                               for i, j, p in top],
                "notes": pred["notes"], "partial_data": bool(pred["partial_data"]),
                "home_lineup": lineup_detail(conn, book, pred["home_lineup"], as_of),
                "away_lineup": lineup_detail(conn, book, pred["away_lineup"], as_of),
            }
        if odds:
            inv = [1 / odds["odds_home"], 1 / odds["odds_draw"],
                   1 / odds["odds_away"]]
            s = sum(inv)
            item["odds"] = {
                "bookmaker": odds["bookmaker"], "ts": odds["ts"],
                "decimal": [odds["odds_home"], odds["odds_draw"],
                            odds["odds_away"]],
                "implied": [round(x / s, 4) for x in inv],
            }
        out.append(item)
    return jsonify({"gameweek": gw, "next_gameweek": next_gameweek(conn),
                    "fixtures": out})


@app.route("/api/squad/<int:team_id>")
def api_squad(team_id):
    conn = db()
    book = predictor().book
    rows = conn.execute(
        """SELECT DISTINCT p.id, p.name, p.position,
                  COALESCE(SUM(pmm.minutes), 0) AS mins
           FROM players p
           LEFT JOIN player_match_minutes pmm ON pmm.player_id=p.id
                AND pmm.team_id=?
           WHERE p.team_id=? OR pmm.player_id IS NOT NULL
           GROUP BY p.id HAVING p.team_id=? OR mins > 0
           ORDER BY mins DESC""", (team_id, team_id, team_id)).fetchall()
    return jsonify([{
        "id": r["id"], "name": r["name"], "position": r["position"] or "?",
        "minutes": r["mins"],
        "tier": book.by_id[r["id"]]["tier"] if r["id"] in book.by_id else 3,
    } for r in rows])


@app.route("/api/fixture/<int:fid>/manual", methods=["POST"])
def api_manual(fid):
    body = request.get_json(force=True)
    conn = db()
    for side in ("home", "away"):
        xi = body.get(side)
        if xi is not None and len(xi) != 11:
            return jsonify({"error": f"{side} lineup must have 11 players"}), 400
    conn.execute("UPDATE fixtures SET lineup_mode='manual' WHERE id=?", (fid,))
    conn.commit()
    p = predictor()
    p.conn = conn
    pred = p.predict_fixture(fid, tag="manual",
                             home_lineup=body.get("home"),
                             away_lineup=body.get("away"))
    return jsonify({"ok": True, "prediction": {
        k: pred[k] for k in ("p_home", "p_draw", "p_away", "home_xg",
                             "away_xg", "notes", "partial_data")}})


@app.route("/api/fixture/<int:fid>/reset", methods=["POST"])
def api_reset(fid):
    conn = db()
    conn.execute("UPDATE fixtures SET lineup_mode='auto' WHERE id=?", (fid,))
    conn.commit()
    return jsonify({"ok": True})


@app.route("/api/performance")
def api_performance():
    import pandas as pd
    conn = db()
    out = {"backtests": [], "prov_vs_final": None, "pull_log": []}
    files = {
        "Layer 1 — season-average": "layer1_season_avg_2526.csv",
        "Layer 1 + lineup/fatigue/travel": "layer1_lineup_weighted_2526.csv",
        "Layer 1 + Layer 2 (style)": "layer2_style_2526.csv",
    }
    from models.backtest import metrics, calibration_table
    for label, fn in files.items():
        path = ROOT / "data" / "backtests" / fn
        if not path.exists():
            continue
        bt = pd.read_csv(path)
        m = metrics(bt)
        out["backtests"].append({
            "label": label, **{k: round(v, 4) for k, v in m.items()},
            "calibration": calibration_table(bt).to_dict("records"),
        })
    pv = conn.execute(
        """SELECT AVG(ABS(p1.p_home - p2.p_home)) AS dh, COUNT(*) AS n
           FROM predictions p1 JOIN predictions p2
             ON p1.fixture_id = p2.fixture_id
            AND p1.tag='provisional' AND p2.tag='final'
            AND p1.id = (SELECT MAX(id) FROM predictions WHERE fixture_id=p1.fixture_id AND tag='provisional')
            AND p2.id = (SELECT MAX(id) FROM predictions WHERE fixture_id=p2.fixture_id AND tag='final')
        """).fetchone()
    if pv and pv["n"]:
        out["prov_vs_final"] = {"fixtures": pv["n"],
                                "avg_home_prob_shift": round(pv["dh"], 4)}
    out["pull_log"] = [dict(r) for r in conn.execute(
        "SELECT ts, source, ok, detail FROM pull_log ORDER BY ts DESC LIMIT 30")]
    return jsonify(out)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
