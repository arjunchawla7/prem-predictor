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
from backend.market import blend_probs, implied_probs, latest_odds
from models.config import (CHAMPIONSHIP_PRIOR, MARKET_BLEND_W, STYLE_ENABLED,
                           TRAIN_SEASONS, make_model)
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
            f"""SELECT m.date, h.name AS home, a.name AS away, m.fthg, m.ftag,
                       m.home_xg, m.away_xg
                FROM matches m JOIN teams h ON h.id=m.home_team_id
                JOIN teams a ON a.id=m.away_team_id
                WHERE COALESCE(m.division, 'E0') = 'E0'
                  AND m.season IN ({','.join('?' * len(TRAIN_SEASONS))})""",
            TRAIN_SEASONS).fetchall()
        self.model = make_model().fit([dict(r) for r in rows])
        self._xl = False        # cross-league fit, built lazily on first need
        self.book = RatingBook(self.conn, RATING_SOURCE)
        self.names = {r["id"]: r["name"] for r in
                      self.conn.execute("SELECT id, name FROM teams")}
        self.coords = {r["id"]: (r["lat"], r["lon"]) for r in
                       self.conn.execute("SELECT id, lat, lon FROM teams")}
        self.style = StyleAdjuster(self.conn, self.model,
                                   datetime.now(timezone.utc).date().isoformat(),
                                   self.names)

    def _crossleague(self):
        """Championship-scale ratings, built once and only when needed."""
        if self._xl is not False:
            return self._xl
        self._xl = None
        if CHAMPIONSHIP_PRIOR:
            from models.crossleague import aligned_pooled_model
            from models.backtest import load_matches
            pooled = load_matches(self.conn, ("E0", "E1"))
            pooled = pooled[pooled["season"].isin(TRAIN_SEASONS + ["2425"])]
            self._xl = aligned_pooled_model(pooled.to_dict("records"),
                                            self.model, make_model)
        return self._xl

    def _team_key(self, team_id):
        """DC model key for a team; prior + flag if the fit has never seen it.

        Championship results first, generic weakest-3 prior second. Either way
        the caller gets told the rating is provisional.
        """
        name = self.names[team_id]
        if name in self.model.attack:
            return name, None
        xl = self._crossleague()
        if xl is not None and name in xl.attack:
            self.model.attack[name] = xl.attack[name]
            self.model.defence[name] = xl.defence[name]
            return name, "championship"
        self.model.attack[name], self.model.defence[name] = \
            promoted_prior(self.model)
        return name, "weakest3"

    def _auto_lineup(self, team_id, as_of):
        xi = ideal_xi(self.conn, team_id, CURRENT_SEASON, as_of)
        if len(xi) == 11:
            return xi
        return ideal_xi(self.conn, team_id, LINEUP_SOURCE_FALLBACK,
                        "2026-07-01")

    def _market_blend(self, fixture_id, probs, factors):
        """Model + market average, as a SEPARATE output.

        Never replaces the model's own probabilities and is never folded into
        them — the whole point of tracking unassisted accuracy is that the
        model's number stays the model's number. Absent when the fixture has no
        odds, which is the normal state until ODDS_API_KEY is set.
        """
        row = latest_odds(self.conn, fixture_id)
        implied = (implied_probs(row["odds_home"], row["odds_draw"],
                                 row["odds_away"]) if row else None)
        if implied is None:
            factors.append({
                "name": "Model + market blend",
                "active": False,
                "effect": "not applied",
                "detail": "no market odds for this fixture "
                          "(needs ODDS_API_KEY)",
                "status": "unavailable",
            })
            return None

        w = MARKET_BLEND_W
        mixed = blend_probs(probs, implied, w)
        factors.append({
            "name": "Model + market blend",
            "active": True,
            "effect": f"{int(w * 100)}/{int((1 - w) * 100)} model/market — "
                      "reported separately, not part of the model's own number",
            "detail": f"{row['bookmaker'] or 'market'} implied "
                      f"{implied[0]:.0%}/{implied[1]:.0%}/{implied[2]:.0%}",
            "status": "separate-output",
        })
        return {"p_home": round(mixed[0], 4), "p_draw": round(mixed[1], 4),
                "p_away": round(mixed[2], 4),
                "market_implied": [round(x, 4) for x in implied],
                "weight_model": w, "bookmaker": row["bookmaker"]}

    def predict_fixture(self, fixture_id, tag="provisional",
                        home_lineup=None, away_lineup=None, store=True):
        f = self.conn.execute("SELECT * FROM fixtures WHERE id=?",
                              (fixture_id,)).fetchone()
        if f is None:
            raise ValueError(f"no fixture {fixture_id}")
        as_of = f["date"] or datetime.now(timezone.utc).isoformat()
        notes, partial = [], False
        # What actually moved this specific prediction. Each entry is
        # (name, active, effect, status) — see FACTOR STATUSES in
        # factor_list() for what the status strings mean.
        factors = []

        hk, hp = self._team_key(f["home_team_id"])
        ak, ap = self._team_key(f["away_team_id"])
        if hp or ap:
            kinds = {k for k in (hp, ap) if k}
            if "championship" in kinds:
                notes.append("Championship-derived rating used for a promoted "
                             "team — cross-league estimate, rougher than a "
                             "top-flight rating")
            if "weakest3" in kinds:
                notes.append("promoted-team prior rating used "
                             "(no top-flight history in training window)")
            partial = True

        factors.append({
            "name": "Home advantage",
            "active": True,
            "effect": f"x{self.model.gamma:.3f} on home expected goals",
            "detail": f"fitted γ = {self.model.gamma:.4f} "
                      f"({(self.model.gamma - 1) * 100:.1f}% uplift)",
            "status": "validated",
        })
        factors.append({
            "name": "Rating source",
            "active": True,
            "effect": "xG-fitted attack/defence, shrunk toward league average",
            "detail": f"trained on {'/'.join(TRAIN_SEASONS)}; "
                      f"prior strength {self.model.prior_strength:g}",
            "status": "validated",
        })
        draw_mult = self.model.draw_boost * self.model.draw_scale
        if draw_mult != 1.0:
            factors.append({
                "name": "Draw correction",
                "active": True,
                "effect": f"x{draw_mult:.3f} on drawn scorelines",
                "detail": f"fitted boost {self.model.draw_boost:.3f} × "
                          f"out-of-sample scale {self.model.draw_scale:.3f}; "
                          "improves calibration, does not change the pick",
                "status": "validated",
            })
        for side, kind, tid in (("home", hp, f["home_team_id"]),
                                ("away", ap, f["away_team_id"])):
            if not kind:
                continue
            champ = kind == "championship"
            factors.append({
                "name": f"Promoted-team rating ({side})",
                "active": True,
                "effect": ("Championship form, rescaled to the top flight"
                           if champ else
                           "rating borrowed from the 3 weakest fitted teams"),
                "detail": (f"{self.names[tid]} has no top-flight history; "
                           + ("second-tier results projected across divisions "
                              "— a rougher estimate than a top-flight rating"
                              if champ else
                              "no second-tier record either, so a generic "
                              "prior is all there is")),
                "status": "cross-league" if champ else "fallback",
            })

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
                factors.append({
                    "name": f"Lineup weighting "
                            f"({'home' if is_home else 'away'})",
                    "active": False,
                    "effect": "not applied",
                    "detail": "no XI available — season-average rating used",
                    "status": "unavailable",
                })
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
            factors.append({
                "name": f"Lineup weighting ({'home' if is_home else 'away'})",
                "active": True,
                "effect": f"x{mult:.3f} on "
                          f"{'home' if is_home else 'away'} expected goals",
                "detail": "player ratings from the previous season's minutes",
                "status": "reference",
            })

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
        if not tpartial:
            factors.append({
                "name": "Travel",
                "active": True,
                "effect": f"x{tm:.3f} on away expected goals",
                "detail": None if dist is None else f"{round(dist)} km away trip",
                "status": "reference",
            })
        if hm != 1.0 or am != 1.0:
            factors.append({
                "name": "Fixture congestion",
                "active": True,
                "effect": f"x{hm:.3f} home / x{am:.3f} away",
                "detail": f"rest days: home {hr}, away {ar}",
                "status": "reference",
            })

        # Layer 1 + multipliers, then the Layer 2 nudge if it is switched on
        lam, mu = self.model.expected_goals(hk, ak)
        lam, mu = lam * lam_mult, mu * mu_mult
        if STYLE_ENABLED:
            slam, smu = self.style.adjust(f["home_team_id"], f["away_team_id"],
                                          lam, mu)
            if (slam, smu) != (lam, mu):
                factors.append({
                    "name": "Style matchup (Layer 2)",
                    "active": True,
                    "effect": f"xG {lam:.2f}/{mu:.2f} → {slam:.2f}/{smu:.2f}",
                    "detail": "pace/pressing bucket pairing",
                    "status": "reference",
                })
            lam, mu = slam, smu
        else:
            factors.append({
                "name": "Style matchup (Layer 2)",
                "active": False,
                "effect": "not applied",
                "detail": "cost accuracy against the current Layer 1 "
                          "(.466 vs .487) — switched off",
                "status": "disabled",
            })

        grid = self.model.score_grid(lam, mu)
        p_h, p_d, p_a = self.model.outcome_probs(lam, mu)
        blend = self._market_blend(fixture_id, (p_h, p_d, p_a), factors)

        pred = {
            "fixture_id": fixture_id, "tag": tag,
            "home_xg": round(float(lam), 3), "away_xg": round(float(mu), 3),
            "p_home": round(p_h, 4), "p_draw": round(p_d, 4),
            "p_away": round(p_a, 4),
            "grid": grid.round(5).tolist(),
            "home_lineup": home_lineup, "away_lineup": away_lineup,
            "notes": "; ".join(notes), "partial_data": int(partial),
            "travel_km": None if dist is None else round(dist),
            "factors": factors,
            "blend": blend,
        }
        if store:
            self.conn.execute(
                """INSERT INTO predictions (fixture_id, created_at, tag,
                     home_xg, away_xg, p_home, p_draw, p_away, score_grid,
                     home_lineup, away_lineup, notes, partial_data, factors,
                     blend_home, blend_draw, blend_away)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (fixture_id, datetime.now(timezone.utc).isoformat(), tag,
                 pred["home_xg"], pred["away_xg"], pred["p_home"],
                 pred["p_draw"], pred["p_away"], json.dumps(pred["grid"]),
                 json.dumps(home_lineup) if home_lineup else None,
                 json.dumps(away_lineup) if away_lineup else None,
                 pred["notes"], pred["partial_data"], json.dumps(factors),
                 (blend or {}).get("p_home"), (blend or {}).get("p_draw"),
                 (blend or {}).get("p_away")))
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
