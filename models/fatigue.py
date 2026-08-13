"""Player fatigue model (0-100) — a documented heuristic.

Recent-load definition (exponential decay, half-life ~2.8 days):
  load = sum over league matches in the past 14 days of
         minutes_played * exp(-DECAY * days_since_that_match)

Anchors for the 0-100 scale:
  - one full 90 played 3+ days ago with a normal week's rest ≈ low 30s
  - two full 90s within 4 days ≈ ~70 (heavily loaded)
  - three full 90s inside 8 days → saturates toward 100

  fatigue = 100 * clip(load / SATURATION_LOAD, 0, 1)

Rating discount (fed into the lineup-weighted team rating):
  below FREE_THRESHOLD fatigue there is no penalty; above it the player's
  strength index is scaled down linearly, to at most MAX_DISCOUNT at 100.
  effective_index = index * (1 - MAX_DISCOUNT * max(0, f - FREE) / (100 - FREE))

KNOWN GAP: only league minutes are in the database. Midweek cup / European
minutes are not tracked (no reachable free source wired up yet), so fatigue
is understated for teams in Europe. Flagged here rather than guessed.
"""
import math

DECAY = 0.25              # per day
WINDOW_DAYS = 14
SATURATION_LOAD = 210.0   # ~3 recent full matches
FREE_THRESHOLD = 50.0
MAX_DISCOUNT = 0.10


def fatigue_score(match_history):
    """match_history: iterable of (days_ago: float, minutes: int) within the
    last WINDOW_DAYS. Returns fatigue 0-100."""
    load = sum(mins * math.exp(-DECAY * days)
               for days, mins in match_history
               if 0 < days <= WINDOW_DAYS)
    return 100.0 * min(1.0, load / SATURATION_LOAD)


def fatigue_multiplier(fatigue: float) -> float:
    """Multiplier applied to a player's strength index."""
    if fatigue <= FREE_THRESHOLD:
        return 1.0
    return 1.0 - MAX_DISCOUNT * (fatigue - FREE_THRESHOLD) / (100.0 - FREE_THRESHOLD)


def player_fatigue(conn, player_id: int, as_of_date: str) -> float:
    """Fatigue from league minutes in the DB as of a date (exclusive)."""
    rows = conn.execute(
        """SELECT julianday(?) - julianday(m.date) AS days_ago, pmm.minutes
           FROM player_match_minutes pmm JOIN matches m ON m.id = pmm.match_id
           WHERE pmm.player_id = ? AND m.date < ?
             AND julianday(?) - julianday(m.date) <= ?""",
        (as_of_date, player_id, as_of_date, as_of_date, WINDOW_DAYS)).fetchall()
    return fatigue_score([(r["days_ago"], r["minutes"]) for r in rows])
