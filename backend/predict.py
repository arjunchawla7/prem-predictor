"""Prediction engine — glues Layer 1 + adjustments + lineup modes together.

Every stored prediction row keeps: tag (provisional/final/manual), the
lineups used (or NULL = season-average / no lineup weighting), expected
goals, W/D/L probs, the full scoreline grid, notes, and a partial_data flag.
Predictions are never overwritten; regeneration inserts a new row.

Mode rules:
  provisional  Mode 1 auto (ideal XI from most-used recent lineup)
  final        Mode 2 auto (confirmed XI from the official PL API)
  manual       Mode 3 (user-picked XI); fixtures with lineup_mode='manual'
               are skipped by automatic rebuilds until reset to 'auto'
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.db import connect
from models.dixon_coles import DixonColes
from models.backtest import promoted_prior
from models.player_ratings import RatingBook
from models.lineup import lineup_multiplier, ideal_xi
from models.style import StyleAdjuster
from models.travel import travel_multiplier, rest_days, congestion_multipliers

CURRENT_SEASON = "2627"
RATING_SOURCE = "2526"      # previous season feeds player ratings
LINEUP_SOURCE_FALLBACK = "2526"  # ideal-XI source before 2627 has matches


class Predictor:
    def __init__(self, conn=None):
        self.conn = conn or connect()
        rows = self.conn.execute(
            """SELECT m.date, h.name AS home, a.name AS away, m.fthg, m.ftag
               FROM matches m JOIN teams h ON h.id=m.home_team_id
               JOIN teams a ON a.id=m.away_team_id""").fetchall()
        self.model = DixonColes().fit([dict(r) for r in rows])
        self.book = RatingBook(self.conn, RATING_SOURCE)
        self.names = {r["id"]: r["name"] for r in
                      self.conn.execute("SELECT id, name FROM teams")}
        self.coords = {r["id"]: (r["lat"], r["lon"]) for r in
                       self.conn.execute("SELECT id, lat, lon FROM teams")}
        self.style = StyleAdjuster(self.conn, self.model,
                                   datetime.now(timezone.utc).date().isoformat(),
                                   self.names)

    def _team_key(self, team_id):
        """DC model key for a team; promoted prior + flag if unseen."""
        name = self.names[team_id]
        if name not in self.model.attack:
            self.model.attack[name], self.model.defence[name] = \
                promoted_prior(self.model)
            return name, True
        return name, False

    def _auto_lineup(self, team_id, as_of):
        xi = ideal_xi(self.conn, team_id, CURRENT_SEASON, as_of)
        if len(xi) == 11:
            return xi
        return ideal_xi(self.conn, team_id, LINEUP_SOURCE_FALLBACK,
                        "2026-07-01")

    def predict_fixture(self, fixture_id, tag="provisional",
                        home_lineup=None, away_lineup=None, store=True):
        f = self.conn.execute("SELECT * FROM fixtures WHERE id=?",
                              (fixture_id,)).fetchone()
        if f is None:
            raise ValueError(f"no fixture {fixture_id}")
        as_of = f["date"] or datetime.now(timezone.utc).isoformat()
        notes, partial = [], False

        hk, hp = self._team_key(f["home_team_id"])
        ak, ap = self._team_key(f["away_team_id"])
        if hp or ap:
            notes.append("promoted-team prior rating used "
                         "(no top-flight history in training window)")
            partial = True

        # lineups
        if tag == "provisional":
            home_lineup = home_lineup or self._auto_lineup(f["home_team_id"], as_of)
            away_lineup = away_lineup or self._auto_lineup(f["away_team_id"], as_of)
        lam_mult = mu_mult = 1.0
        for team_id, xi, is_home in ((f["home_team_id"], home_lineup, True),
                                     (f["away_team_id"], away_lineup, False)):
            if not xi or len(xi) != 11:
                notes.append(f"{'home' if is_home else 'away'} lineup "
                             "unavailable — season-average rating used")
                partial = True
                continue
            mult, det = lineup_multiplier(self.conn, self.book, team_id, xi,
                                          str(as_of), CURRENT_SEASON)
            if det.get("partial"):
                notes.append(f"{'home' if is_home else 'away'} lineup mostly "
                             "provisional-rated players")
                partial = True
            if is_home:
                lam_mult *= mult
            else:
                mu_mult *= mult

        # travel + congestion
        tm, dist, tpartial = travel_multiplier(
            self.coords.get(f["away_team_id"], (None, None)),
            self.coords.get(f["home_team_id"], (None, None)))
        if tpartial:
            notes.append("stadium coords missing — no travel adjustment")
        mu_mult *= tm
        hr = rest_days(self.conn, f["home_team_id"], str(as_of)[:10])
        ar = rest_days(self.conn, f["away_team_id"], str(as_of)[:10])
        hm, am, cflags = congestion_multipliers(hr, ar)
        lam_mult *= hm
        mu_mult *= am
        notes += cflags

        # Layer 1 + multipliers, then Layer 2 nudge
        lam, mu = self.model.expected_goals(hk, ak)
        lam, mu = lam * lam_mult, mu * mu_mult
        lam, mu = self.style.adjust(f["home_team_id"], f["away_team_id"], lam, mu)

        grid = self.model.score_grid(lam, mu)
        p_h, p_d, p_a = self.model.outcome_probs(lam, mu)

        pred = {
            "fixture_id": fixture_id, "tag": tag,
            "home_xg": round(float(lam), 3), "away_xg": round(float(mu), 3),
            "p_home": round(p_h, 4), "p_draw": round(p_d, 4),
            "p_away": round(p_a, 4),
            "grid": grid.round(5).tolist(),
            "home_lineup": home_lineup, "away_lineup": away_lineup,
            "notes": "; ".join(notes), "partial_data": int(partial),
            "travel_km": None if dist is None else round(dist),
        }
        if store:
            self.conn.execute(
                """INSERT INTO predictions (fixture_id, created_at, tag,
                     home_xg, away_xg, p_home, p_draw, p_away, score_grid,
                     home_lineup, away_lineup, notes, partial_data)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (fixture_id, datetime.now(timezone.utc).isoformat(), tag,
                 pred["home_xg"], pred["away_xg"], pred["p_home"],
                 pred["p_draw"], pred["p_away"], json.dumps(pred["grid"]),
                 json.dumps(home_lineup) if home_lineup else None,
                 json.dumps(away_lineup) if away_lineup else None,
                 pred["notes"], pred["partial_data"]))
            self.conn.commit()
        return pred

    def predict_gameweek(self, gameweek, tag="provisional"):
        """Mode 1 sweep. Skips fixtures locked to manual mode."""
        rows = self.conn.execute(
            """SELECT id, lineup_mode FROM fixtures
               WHERE season=? AND gameweek=? ORDER BY date""",
            (CURRENT_SEASON, gameweek)).fetchall()
        out = []
        for r in rows:
            if r["lineup_mode"] == "manual":
                print(f"  fixture {r['id']}: manual mode — skipped")
                continue
            out.append(self.predict_fixture(r["id"], tag=tag))
        return out


def next_gameweek(conn):
    row = conn.execute(
        """SELECT MIN(gameweek) AS gw FROM fixtures
           WHERE season=? AND status='scheduled'""",
        (CURRENT_SEASON,)).fetchone()
    return row["gw"]
