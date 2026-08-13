"""Lineup-weighted team strength.

Idea: the Dixon-Coles attack/defence ratings embody the team's *typical*
lineup over the (decay-weighted) training window. When we know or assume a
specific starting XI, we nudge expected goals by how that XI compares to the
team's typical lineup:

  L  = mean over the 11 starters of (rating index * fatigue multiplier)
  B  = minutes-weighted mean rating index over the team's season-to-date
       players (the "typical lineup" the DC rating embodies); early-season
       fallback = plain mean of the XI itself (=> multiplier 1.0)
  lineup multiplier on that team's expected goals = clip(L / B, 0.92, 1.08)

The clip keeps this a nudge, not a replacement for Layer 1. Rating indices
come from PREVIOUS-season data (player_ratings.py), fatigue multipliers from
fatigue.py. Unknown/provisional players contribute index 1.0.

Ideal XI (Mode 1): the 11 players with the most starts (ties: minutes) for
the team in the target season's recent matches — the "most-used lineup",
filled per position group so we don't end up with 5 goalkeepers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.fatigue import player_fatigue, fatigue_multiplier

CLIP_LO, CLIP_HI = 0.92, 1.08


def lineup_multiplier(conn, book, team_id, starter_ids, as_of_date,
                      season):
    """Returns (multiplier, detail dict)."""
    if not starter_ids:
        return 1.0, {"reason": "no lineup available", "partial": True}

    effs, fatigues, provisional = [], {}, 0
    for pid in starter_ids:
        f = player_fatigue(conn, pid, as_of_date)
        eff = book.index_of(pid) * fatigue_multiplier(f)
        effs.append(eff)
        fatigues[pid] = f
        provisional += book.is_provisional(pid)
    L = sum(effs) / len(effs)

    # Typical-lineup baseline: minutes-weighted mean index, season to date.
    rows = conn.execute(
        """SELECT pmm.player_id, SUM(pmm.minutes) AS mins
           FROM player_match_minutes pmm JOIN matches m ON m.id=pmm.match_id
           WHERE pmm.team_id=? AND m.season=? AND m.date<?
           GROUP BY pmm.player_id""",
        (team_id, season, as_of_date)).fetchall()
    total = sum(r["mins"] for r in rows)
    if total < 990 * 3:          # < ~3 matches of data: no reliable baseline
        B = L
    else:
        B = sum(book.index_of(r["player_id"]) * r["mins"] for r in rows) / total

    mult = max(CLIP_LO, min(CLIP_HI, L / B)) if B > 0 else 1.0
    return mult, {"lineup_strength": L, "baseline": B,
                  "provisional_players": provisional, "fatigues": fatigues,
                  "partial": provisional >= 4}


def ideal_xi(conn, team_id, season, as_of_date, last_n=6):
    """Mode 1 'most-used lineup': most-started players in the team's last
    `last_n` matches before as_of_date, slotted per position group
    (1 GK, then DEF/MID/FWD by start counts, topped up to 11)."""
    rows = conn.execute(
        """WITH recent AS (
             SELECT m.id FROM matches m
             WHERE (m.home_team_id=? OR m.away_team_id=?)
               AND m.season=? AND m.date<?
             ORDER BY m.date DESC LIMIT ?)
           SELECT pmm.player_id, p.position,
                  SUM(pmm.started) AS starts, SUM(pmm.minutes) AS mins
           FROM player_match_minutes pmm
           JOIN recent r ON r.id = pmm.match_id
           JOIN players p ON p.id = pmm.player_id
           WHERE pmm.team_id = ?
           GROUP BY pmm.player_id
           ORDER BY starts DESC, mins DESC""",
        (team_id, team_id, season, as_of_date, last_n, team_id)).fetchall()
    if not rows:
        return []

    by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": [], None: []}
    for r in rows:
        by_pos.setdefault(r["position"], by_pos[None]).append(r["player_id"])
    xi = by_pos["GK"][:1]
    # loose 4-4-2-ish caps; real shape doesn't matter, only who plays
    for pos, cap in (("DEF", 5), ("MID", 5), ("FWD", 3)):
        xi += by_pos[pos][:cap]
    if len(xi) > 11:
        xi = xi[:11]
    else:  # top up with next most-started regardless of position
        for r in rows:
            if len(xi) == 11:
                break
            if r["player_id"] not in xi:
                xi.append(r["player_id"])
    return xi
