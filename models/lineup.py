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


# Specific slot roles for a formation string, and how well a player's
# detail_pos suits one. This mirrors specificRoles/fitScore in
# frontend/index.html — the builder needs the same answer client-side for
# shapes the user picks from the dropdown, so the logic exists in both places.
# Keep them in step; the shapes are derived by models/formation.py either way.
LINE_DEF = {3: ["CB", "CB", "CB"], 4: ["LB", "CB", "CB", "RB"],
            5: ["LB", "CB", "CB", "CB", "RB"]}
LINE_FWD = {1: ["ST"], 2: ["ST", "ST"], 3: ["LW", "ST", "RW"]}
FITS = {"GK": ["GK"], "LB": ["LB", "FB", "LWB"], "RB": ["RB", "FB", "RWB"],
        "CB": ["CB"], "CDM": ["CDM", "CM"], "CM": ["CM", "CDM", "CAM"],
        "CAM": ["CAM", "SS", "CM"], "LW": ["LW", "W", "LM"],
        "RW": ["RW", "W", "RM"], "ST": ["ST", "SS", "CF"]}
COARSE = {"GK": "GK", "LB": "DEF", "RB": "DEF", "CB": "DEF", "CDM": "MID",
          "CM": "MID", "CAM": "MID", "LW": "FWD", "RW": "FWD", "ST": "FWD"}


def slot_roles(shape):
    """'4-2-3-1' -> ['GK','LB','CB','CB','RB','CDM','CDM','LW','CAM','RW','ST']"""
    try:
        lines = [int(x) for x in shape.split("-")]
    except (AttributeError, ValueError):
        return None
    if sum(lines) != 10:
        return None
    out, mid_depth = ["GK"], 0
    mid_count = len(lines) - 2
    for li, count in enumerate(lines):
        if li == 0:
            out += LINE_DEF.get(count, ["CB"] * count)
        elif li == len(lines) - 1:
            out += LINE_FWD.get(count, ["ST"] * count)
        else:
            if mid_count == 1:
                base, widen = "CM", count >= 4
            elif mid_depth == 0:
                base, widen = ("CDM" if count <= 2 else "CM"), count >= 4
            else:
                highest = mid_depth == mid_count - 1
                base = "CAM" if highest else "CM"
                widen = highest and count >= 3
            row = [base] * count
            if widen:
                row[0], row[-1] = "LW", "RW"
            out += row
            mid_depth += 1
    return out


def role_fit(detail, position, role):
    """How well a player suits a specific slot. Higher is better, 0 = no."""
    dp = (detail or "").upper()
    opts = FITS.get(role, [])
    if dp in opts:
        return 100 - opts.index(dp) * 5
    if position and position == COARSE.get(role):
        return 30 if dp else 40
    return 0


def ideal_xi(conn, team_id, season, as_of_date, last_n=6):
    """Mode 1 'most-used lineup': most-started players in the team's last
    `last_n` matches before as_of_date, slotted per position group
    (1 GK, then DEF/MID/FWD by start counts, topped up to 11).

    When a synced registered squad exists for the team (pull_squads.py),
    players no longer in it are excluded — otherwise last season's minutes
    would keep picking players who transferred away over the summer."""
    has_squad = conn.execute(
        "SELECT 1 FROM players WHERE team_id=? AND in_current_squad=1 LIMIT 1",
        (team_id,)).fetchone() is not None
    squad_filter = ("AND p.in_current_squad=1 AND p.team_id=:tid"
                    if has_squad else "")
    rows = conn.execute(
        f"""WITH recent AS (
             SELECT m.id FROM matches m
             WHERE (m.home_team_id=:tid OR m.away_team_id=:tid)
               AND m.season=:season AND m.date<:as_of
             ORDER BY m.date DESC LIMIT :n)
           SELECT pmm.player_id, p.position, p.detail_pos,
                  SUM(pmm.started) AS starts, SUM(pmm.minutes) AS mins
           FROM player_match_minutes pmm
           JOIN recent r ON r.id = pmm.match_id
           JOIN players p ON p.id = pmm.player_id
           WHERE pmm.team_id = :tid {squad_filter}
           GROUP BY pmm.player_id
           ORDER BY starts DESC, mins DESC""",
        {"tid": team_id, "season": season, "as_of": as_of_date,
         "n": last_n}).fetchall()
    if not rows:
        return []

    # Fill the team's OWN derived shape, slot by slot, taking the most-started
    # player who actually plays that role.
    #
    # Coarse caps (GK 1, DEF 5, MID 5, FWD 3) were not enough: two right-backs
    # both count as DEF, so Man United's XI came out with Dalot AND Mazraoui
    # and only one natural central midfielder, which left a full-back standing
    # in the holding role. Selecting per specific slot picks a coherent side.
    shape = None
    try:
        from models.formation import preferred_formation
        pf = preferred_formation(conn, team_id, as_of_date)
        shape = pf["formation"] if pf else None
    except Exception:
        shape = None
    roles = slot_roles(shape) if shape else None

    detail = {r["player_id"]: r["detail_pos"] for r in rows}
    xi = []
    if roles:
        # Global greedy over every (slot, player) pair rather than slot by
        # slot. Taking slots in order lets an early slot consume a key player
        # on a mediocre fit — the left-wing slot would swallow a second
        # striker before the centre-forward slot was ever considered.
        # Ties break on starts, so a regular starter beats a fringe player at
        # equal fit.
        pairs = []
        for si, role in enumerate(roles):
            for r in rows:
                s = role_fit(detail.get(r["player_id"]), r["position"], role)
                if s > 0:
                    pairs.append((s, r["starts"] or 0, r["mins"] or 0, si,
                                  r["player_id"]))
        pairs.sort(key=lambda t: (-t[0], -t[1], -t[2]))
        slots, used = {}, set()
        for s, st, mn, si, pid in pairs:
            if si in slots or pid in used:
                continue
            slots[si] = pid
            used.add(pid)
        xi = [slots[i] for i in range(len(roles)) if i in slots]
        # any slot nobody suited falls back to most-started
        for r in rows:
            if len(xi) == 11:
                break
            if r["player_id"] not in used:
                xi.append(r["player_id"])
        return xi[:11]

    # No derived shape (a promoted side with no match history): keep the old
    # coarse behaviour rather than invent one.
    caps = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3, None: 2}
    taken = {k: 0 for k in caps}
    for r in rows:
        pos = r["position"] if r["position"] in caps else None
        if len(xi) < 11 and taken[pos] < caps[pos]:
            xi.append(r["player_id"])
            taken[pos] += 1
    if len(xi) < 11:  # sparse data: top up ignoring caps
        for r in rows:
            if len(xi) == 11:
                break
            if r["player_id"] not in xi:
                xi.append(r["player_id"])
    return xi
