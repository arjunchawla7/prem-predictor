"""Market odds: overround removal, the model+market blend, and snapshot history.

Odds are stored raw in market_odds, one row per pull (PK is fixture_id + ts),
so repeated pulls build a price history for free. Everything that turns those
raw prices into probabilities lives here, so the prediction engine and the API
cannot drift apart on how the overround is removed or how the blend is mixed.

The blend stays a SEPARATE output. It never replaces the model's own numbers —
tracking unassisted accuracy only means anything if the model's number stays
the model's number.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.config import MARKET_BLEND_W


def implied_probs(odds_home, odds_draw, odds_away):
    """Overround removed: 1/odds normalised to sum to 1. None if incomplete."""
    if not all((odds_home, odds_draw, odds_away)):
        return None
    inv = [1 / odds_home, 1 / odds_draw, 1 / odds_away]
    s = sum(inv)
    return [x / s for x in inv]


def blend_probs(model, implied, w=MARKET_BLEND_W):
    """w × model + (1-w) × market, per outcome."""
    return [w * p + (1 - w) * q for p, q in zip(model, implied)]


def latest_odds(conn, fixture_id):
    return conn.execute(
        """SELECT * FROM market_odds WHERE fixture_id=?
           ORDER BY ts DESC LIMIT 1""", (fixture_id,)).fetchone()


def _age_hours(ts):
    try:
        t = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - t).total_seconds() / 3600


def market_view(conn, fixture_id, model_probs=None):
    """Everything the UI needs about one fixture's market, as of right now.

    Built from the LATEST snapshot rather than whatever was current when the
    prediction row was written, so a line that moved after the prediction was
    generated shows the moved price — including in the blend, which is pure
    display arithmetic over the model's stored probabilities.
    """
    row = latest_odds(conn, fixture_id)
    if not row:
        return None
    implied = implied_probs(row["odds_home"], row["odds_draw"], row["odds_away"])
    if implied is None:
        return None
    first = conn.execute(
        """SELECT ts, odds_home, odds_draw, odds_away FROM market_odds
           WHERE fixture_id=? ORDER BY ts LIMIT 1""", (fixture_id,)).fetchone()
    n = conn.execute("SELECT COUNT(*) c FROM market_odds WHERE fixture_id=?",
                     (fixture_id,)).fetchone()["c"]
    out = {
        "bookmaker": row["bookmaker"], "ts": row["ts"],
        "age_hours": _age_hours(row["ts"]),
        "snapshots": n,
        "decimal": [row["odds_home"], row["odds_draw"], row["odds_away"]],
        "implied": [round(x, 4) for x in implied],
        "opening": None, "drift": None,
    }
    # Movement since the first snapshot we hold. Shown as a probability drift
    # because a decimal-price delta means little without the other two prices.
    if first and n > 1:
        op = implied_probs(first["odds_home"], first["odds_draw"],
                           first["odds_away"])
        if op:
            out["opening"] = {"ts": first["ts"],
                              "implied": [round(x, 4) for x in op],
                              "decimal": [first["odds_home"], first["odds_draw"],
                                          first["odds_away"]]}
            out["drift"] = [round(a - b, 4) for a, b in zip(implied, op)]
    if model_probs is not None:
        mixed = blend_probs(model_probs, implied)
        out["blend"] = {"p_home": round(mixed[0], 4),
                        "p_draw": round(mixed[1], 4),
                        "p_away": round(mixed[2], 4),
                        "weight_model": MARKET_BLEND_W}
    return out
