"""Match analytics read straight off the scoreline grid.

Nothing here is fitted. The Dixon-Coles grid already carries P(home=i, away=j)
for every scoreline, so questions like "how likely is a high-scoring match" or
"how likely is it that both sides score" are sums over regions of that same
grid — they inherit the goals model's calibration exactly, and can never
disagree with the scorelines shown alongside them.

That also bounds what they are worth: if the grid is off for a fixture, these
are off in the same direction. They are a different view of one model, not a
second opinion.

Grid convention: rows = home goals, columns = away goals, already
renormalised, so the whole grid sums to 1.
"""


def _as_rows(grid):
    return [list(row) for row in grid]


def total_goals_distribution(grid, max_total=None):
    """P(total goals = k) for k = 0 .. (grid size - 1) * 2."""
    rows = _as_rows(grid)
    n = len(rows)
    top = max_total if max_total is not None else 2 * (n - 1)
    out = [0.0] * (top + 1)
    for i, row in enumerate(rows):
        for j, p in enumerate(row):
            k = i + j
            if k <= top:
                out[k] += p
    return out


def goals_over(grid, line=2.5):
    """P(total goals > line). A .5 line cannot tie, which is the whole reason
    these are quoted at half-goals."""
    rows = _as_rows(grid)
    return float(sum(p for i, row in enumerate(rows)
                     for j, p in enumerate(row) if i + j > line))


def both_teams_score(grid):
    """P(home >= 1 and away >= 1)."""
    rows = _as_rows(grid)
    return float(sum(p for row in rows[1:] for p in row[1:]))


def clean_sheet(grid):
    """(P(away fails to score), P(home fails to score)) — i.e. a clean sheet
    for the home side, then for the away side."""
    rows = _as_rows(grid)
    home_cs = float(sum(row[0] for row in rows))
    away_cs = float(sum(rows[0]))
    return home_cs, away_cs


def derived_stats(grid, line=2.5):
    """Everything Tier 1 exposes, in one pass over the grid.

    Probabilities are rounded for transport; `line` is carried so the UI never
    has to hardcode which threshold was used.
    """
    over = goals_over(grid, line)
    btts = both_teams_score(grid)
    home_cs, away_cs = clean_sheet(grid)
    dist = total_goals_distribution(grid)
    # Expected total goals from the grid rather than lam+mu: the low-score
    # correction and draw boost both move mass around after those are set, so
    # this is the number consistent with what is actually displayed.
    exp_total = sum(k * p for k, p in enumerate(dist))
    return {
        "line": line,
        "over": round(over, 4),
        "under": round(1 - over, 4),
        "btts": round(btts, 4),
        "btts_no": round(1 - btts, 4),
        "home_clean_sheet": round(home_cs, 4),
        "away_clean_sheet": round(away_cs, 4),
        "expected_total_goals": round(exp_total, 3),
        # trimmed: beyond 6 the mass is negligible and the row gets noisy
        "total_goals": [round(p, 4) for p in dist[:7]],
    }
