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
from backend.db import connect, DATA_DIR
from backend.counts_service import CountService
from backend.market import market_view
from backend.predict import Predictor, next_gameweek, CURRENT_SEASON
from models.derived import derived_stats
from models.fatigue import player_fatigue
from models.player_ratings import RatingBook

app = Flask(__name__, static_folder=str(ROOT / "frontend"))
_predictor = None
_counts = None


def predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor(connect())
    return _predictor


def counts():
    global _counts
    if _counts is None:
        _counts = CountService(connect())
    return _counts


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


PHOTO = ("https://resources.premierleague.com/premierleague/photos/players/"
         "{size}/{opta}.png")


def refresh_blend_factor(factors, market):
    """Keep the factor list honest about the blend after an odds refresh.

    A prediction generated before its fixture had odds recorded "not applied"
    forever, which now contradicts the live blend shown above it. Rewritten at
    read time only — the stored row still records what fed the prediction.
    """
    if not market or not market.get("blend"):
        return factors
    imp = market["implied"]
    entry = {
        "name": "Model + market blend",
        "active": True,
        "effect": f"{int(100 * market['blend']['weight_model'])}/"
                  f"{int(100 * (1 - market['blend']['weight_model']))} "
                  "model/market — reported separately, not part of the "
                  "model's own number",
        "detail": f"{market.get('bookmaker') or 'market'} implied "
                  f"{imp[0]:.0%}/{imp[1]:.0%}/{imp[2]:.0%}, "
                  f"from the latest price snapshot",
        "status": "separate-output",
    }
    out = [entry if f.get("name") == "Model + market blend" else f
           for f in factors]
    return out if any(f.get("name") == "Model + market blend"
                      for f in factors) else out + [entry]


def photo_url(opta_code, size="40x40"):
    return PHOTO.format(size=size, opta=opta_code) if opta_code else None


def lineup_detail(conn, book, lineup_json, as_of):
    if not lineup_json:
        return None
    out = []
    for pid in json.loads(lineup_json):
        p = conn.execute(
            """SELECT name, position, detail_pos, opta_code, shirt_num
               FROM players WHERE id=?""", (pid,)).fetchone()
        r = book.by_id.get(pid)
        out.append({
            "id": pid, "name": p["name"] if p else f"#{pid}",
            "position": (p["position"] if p else None) or "?",
            "detail_pos": p["detail_pos"] if p else None,
            "shirt": p["shirt_num"] if p else None,
            "photo": photo_url(p["opta_code"]) if p else None,
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
        """SELECT f.*, h.name hn, a.name an, h.id hid, a.id aid,
                  h.crest_url hcrest, a.crest_url acrest
           FROM fixtures f
           JOIN teams h ON h.id=f.home_team_id
           JOIN teams a ON a.id=f.away_team_id
           WHERE f.season=? AND f.gameweek=? ORDER BY f.date""",
        (CURRENT_SEASON, gw)).fetchall()
    # The manual lineup builder lives on this page and needs each side's
    # derived shape to default to. preferred_formation is the trait-free half
    # of team_profile (~1ms per team), so carrying it here costs nothing and
    # means the builder reads the SAME source as the teamsheet panel rather
    # than a hardcoded guess.
    from models.formation import preferred_formation
    out = []
    for f in fixtures:
        pred = active_prediction(conn, f)
        before = (f["date"] or "")[:10] or None
        shapes = {side: preferred_formation(conn, tid, before)
                  for side, tid in (("home", f["hid"]), ("away", f["aid"]))}
        item = {
            "id": f["id"], "date": f["date"], "gameweek": f["gameweek"],
            "home": {"id": f["hid"], "name": f["hn"], "crest": f["hcrest"],
                     "formation": shapes["home"]},
            "away": {"id": f["aid"], "name": f["an"], "crest": f["acrest"],
                     "formation": shapes["away"]},
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
                # sums over the same grid — free, and can never disagree
                # with the scorelines shown next to them
                "derived": derived_stats(grid),
                "notes": pred["notes"], "partial_data": bool(pred["partial_data"]),
                "home_lineup": lineup_detail(conn, book, pred["home_lineup"], as_of),
                "away_lineup": lineup_detail(conn, book, pred["away_lineup"], as_of),
            }
        # Read-time, so a line that moved after the prediction was generated
        # shows the moved price rather than the snapshot it was blended with.
        item["odds"] = market_view(conn, f["id"])
        out.append(item)
    return jsonify({"gameweek": gw, "next_gameweek": next_gameweek(conn),
                    "fixtures": out})


@app.route("/api/squad/<int:team_id>")
def api_squad(team_id):
    """Registered current squad (pull_squads.py); falls back to the
    minutes-based pool if the squad hasn't been synced."""
    conn = db()
    book = predictor().book
    rows = conn.execute(
        """SELECT p.id, p.name, p.position, p.shirt_num, p.detail_pos,
                  p.opta_code, p.understat_id IS NULL AS is_new,
                  COALESCE((SELECT SUM(minutes) FROM player_match_minutes
                            WHERE player_id=p.id), 0) AS mins
           FROM players p WHERE p.team_id=? AND p.in_current_squad=1
           ORDER BY CASE p.position WHEN 'GK' THEN 0 WHEN 'DEF' THEN 1
                    WHEN 'MID' THEN 2 WHEN 'FWD' THEN 3 ELSE 4 END,
                    mins DESC""", (team_id,)).fetchall()
    if not rows:
        rows = conn.execute(
            """SELECT DISTINCT p.id, p.name, p.position, NULL AS shirt_num,
                      NULL AS detail_pos, NULL AS opta_code,
                      0 AS is_new, COALESCE(SUM(pmm.minutes), 0) AS mins
               FROM players p
               LEFT JOIN player_match_minutes pmm ON pmm.player_id=p.id
                    AND pmm.team_id=?
               WHERE p.team_id=? OR pmm.player_id IS NOT NULL
               GROUP BY p.id HAVING p.team_id=? OR mins > 0
               ORDER BY mins DESC""", (team_id, team_id, team_id)).fetchall()
    return jsonify([{
        "id": r["id"], "name": r["name"], "position": r["position"] or "?",
        "detail_pos": r["detail_pos"], "photo": photo_url(r["opta_code"]),
        "minutes": r["mins"], "shirt": r["shirt_num"],
        "new_signing": bool(r["is_new"]),
        "tier": book.by_id[r["id"]]["tier"] if r["id"] in book.by_id else 3,
        "provisional": (book.by_id[r["id"]]["provisional"]
                        if r["id"] in book.by_id else True),
    } for r in rows])


@app.route("/match")
def match_page():
    return send_from_directory(app.static_folder, "match.html")


@app.route("/api/fixture/<int:fid>")
def api_fixture_detail(fid):
    """Everything the match page needs: teams, prediction incl. full
    scoreline grid, lineups grouped by position, odds, transfers in."""
    conn = db()
    book = predictor().book
    f = conn.execute(
        """SELECT f.*, h.name hn, a.name an, h.crest_url hc, a.crest_url ac,
                  h.id hid, a.id aid
           FROM fixtures f JOIN teams h ON h.id=f.home_team_id
           JOIN teams a ON a.id=f.away_team_id WHERE f.id=?""",
        (fid,)).fetchone()
    if not f:
        return jsonify({"error": "no such fixture"}), 404
    pred = active_prediction(conn, f)
    history = [dict(r) for r in conn.execute(
        """SELECT tag, created_at, p_home, p_draw, p_away, home_xg, away_xg
           FROM predictions WHERE fixture_id=? ORDER BY created_at""", (fid,))]
    model_probs = ([pred["p_home"], pred["p_draw"], pred["p_away"]]
                   if pred else None)
    # Latest snapshot, and the blend recomputed against it — the blend columns
    # on the prediction row froze whatever line was current when it was written.
    market = market_view(conn, fid, model_probs)
    odds_history = [dict(r) for r in conn.execute(
        """SELECT ts, odds_home, odds_draw, odds_away FROM market_odds
           WHERE fixture_id=? ORDER BY ts""", (fid,))]
    from models.formation import team_profile
    before = (f["date"] or "")[:10] or None
    profiles = {side: team_profile(conn, tid, before)
                for side, tid in (("home", f["hid"]), ("away", f["aid"]))}
    managers = {}
    for side, tid in (("home", f["hid"]), ("away", f["aid"])):
        m = conn.execute(
            """SELECT name, role, nationality, opta_code FROM managers
               WHERE team_id=?""", (tid,)).fetchone()
        managers[side] = ({"name": m["name"], "role": m["role"],
                           "nationality": m["nationality"],
                           "photo": photo_url(m["opta_code"], "110x140")}
                          if m else None)
    recent_transfers = {}
    for side, tid in (("home", f["hid"]), ("away", f["aid"])):
        recent_transfers[side] = [dict(r) for r in conn.execute(
            """SELECT p.name, ft.name AS from_team
               FROM transfers tr JOIN players p ON p.id=tr.player_id
               LEFT JOIN teams ft ON ft.id=tr.from_team_id
               WHERE tr.to_team_id=? AND tr.from_team_id IS NOT NULL
               ORDER BY tr.detected_at DESC LIMIT 8""", (tid,))]
    out = {
        "id": f["id"], "date": f["date"], "gameweek": f["gameweek"],
        "status": f["status"], "lineup_mode": f["lineup_mode"],
        "home": {"id": f["hid"], "name": f["hn"], "crest": f["hc"]},
        "away": {"id": f["aid"], "name": f["an"], "crest": f["ac"]},
        "prediction": None, "history": history, "odds": market,
        "odds_history": odds_history,
        "transfers_in": recent_transfers,
        "managers": managers, "profiles": profiles,
    }
    if pred:
        as_of = f["date"] or ""
        out["prediction"] = {
            "tag": pred["tag"], "created_at": pred["created_at"],
            "p_home": pred["p_home"], "p_draw": pred["p_draw"],
            "p_away": pred["p_away"], "home_xg": pred["home_xg"],
            "away_xg": pred["away_xg"],
            "grid": json.loads(pred["score_grid"]),
            "derived": derived_stats(json.loads(pred["score_grid"])),
            # fitted separately from the goals grid, and shown as context
            # rather than prediction — see backend/counts_service.py
            "counts": counts().for_fixture(f["hn"], f["an"]),
            "notes": pred["notes"], "partial_data": bool(pred["partial_data"]),
            "home_lineup": lineup_detail(conn, book, pred["home_lineup"], as_of),
            "away_lineup": lineup_detail(conn, book, pred["away_lineup"], as_of),
            # what actually fed THIS prediction; older rows predate the column
            "factors": refresh_blend_factor(
                json.loads(pred["factors"]) if pred["factors"] else [], market),
            # live blend first; the stored columns are the fallback for rows
            # whose fixture has since lost its odds
            "blend": ((market or {}).get("blend")
                      or ({"p_home": pred["blend_home"],
                           "p_draw": pred["blend_draw"],
                           "p_away": pred["blend_away"]}
                          if pred["blend_home"] is not None else None)),
        }
    return jsonify(out)


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
    from models.config import BACKTEST_FILES as files
    from models.backtest import metrics, calibration_table
    for label, fn in files.items():
        path = DATA_DIR / "backtests" / fn
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
    # Local dev only — a real deployment runs this through gunicorn (see
    # Procfile), which binds $PORT itself and never hits this branch.
    import os
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
