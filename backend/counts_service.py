"""Live corners and cards estimates for the web app.

Fits are cheap (~0.07s over four seasons) so the models are built lazily on
first request and cached, rather than stored per prediction. Nothing is
written to the predictions table: these did not earn the status of a stored
prediction — see the honesty note below and /performance#counts.

WHAT THE BACKTEST SAID (2025-26 walk-forward, scripts/counts_backtest.py):

  corners   MAE 2.684 vs 2.658 for simply predicting the league average.
            Worse. At every time-decay setting tried. There is very little
            per-match signal here beyond "about ten corners happen".
  cards     MAE 1.585 vs 1.559 for the league average — also worse on the
            point estimate — though marginally better than the base rate in
            probability terms (Brier -.0020, log-loss -.0039).
  referee   Adding a referee factor made cards WORSE on every measure
            (MAE +.011, Brier +.0014, log-loss +.0029), despite a real
            0.88-1.14 spread in referee strictness. It is fitted, reported
            as context, and deliberately NOT applied.

So these ship as descriptive context — a team's typical corner and card
volume, adjusted for the opponent — and are labelled that way everywhere they
appear. No threshold probabilities are exposed, because a probability that
scores worse than its own base rate is worse than saying nothing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.config import TRAIN_SEASONS
from models.counts import CountModel

# Referee strictness is measured and shown, never multiplied into the estimate.
APPLY_REFEREE = False


def _load(conn, column):
    rows = conn.execute(
        f"""SELECT m.date, h.name AS home, a.name AS away, m.referee,
                   hs.{column} AS home_count, aws.{column} AS away_count
            FROM matches m
            JOIN teams h ON h.id = m.home_team_id
            JOIN teams a ON a.id = m.away_team_id
            JOIN team_match_stats hs ON hs.match_id = m.id AND hs.is_home = 1
            JOIN team_match_stats aws ON aws.match_id = m.id AND aws.is_home = 0
            WHERE COALESCE(m.division, 'E0') = 'E0'
              AND m.season IN ({','.join('?' * len(TRAIN_SEASONS))})
              AND hs.{column} IS NOT NULL AND aws.{column} IS NOT NULL""",
        TRAIN_SEASONS).fetchall()
    return [dict(r) for r in rows]


class CountService:
    def __init__(self, conn):
        self.conn = conn
        self._corners = None
        self._cards = None

    @property
    def corners(self):
        if self._corners is None:
            self._corners = CountModel(use_referee=False).fit(
                _load(self.conn, "corners"))
        return self._corners

    @property
    def cards(self):
        if self._cards is None:
            # referee factors are fitted so they can be REPORTED; whether they
            # are applied is decided by APPLY_REFEREE, which the backtest set
            # to False
            self._cards = CountModel(use_referee=True).fit(
                _load(self.conn, "yellows"))
        return self._cards

    def for_fixture(self, home, away, referee=None):
        ch, ca = self.corners.expected_counts(home, away)
        model = self.cards
        yh, ya = model.expected_counts(
            home, away, referee if APPLY_REFEREE else None)
        ref = {
            "name": referee,
            "factor": model.referee_factor(referee) if referee else None,
            "applied": bool(APPLY_REFEREE and referee),
        }
        return {
            "corners": {"home": round(ch, 1), "away": round(ca, 1),
                        "total": round(ch + ca, 1)},
            "cards": {"home": round(yh, 1), "away": round(ya, 1),
                      "total": round(yh + ya, 1)},
            "referee": ref,
            # every consumer should be able to see the caveat without having
            # to know where it is written down
            "confidence": {
                "corners": "context-only",
                "cards": "low",
            },
        }
