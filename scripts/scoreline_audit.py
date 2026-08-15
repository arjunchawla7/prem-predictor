"""Is 1-1 genuinely just edging out its neighbours, or dominating?

The gameweek cards used to headline the single most likely scoreline, which
made the model read as if it were calling 1-1 with confidence. Before changing
anything in the maths, this checks whether the grid is behaving the way plain
Poisson says it should: a top scoreline around 12-15% with several rivals a
point or two behind, and no runaway.

Reads the grids ALREADY STORED for scheduled fixtures — the same numbers the
app renders — so this audits what ships, not a re-derivation of it. Sampled
evenly across total expected goals so close matchups and mismatches both
appear.

  .venv\\Scripts\\python scripts\\scoreline_audit.py [-n 20]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.db import connect
from backend.predict import CURRENT_SEASON

WATCH = ["1-1", "1-0", "0-1", "2-1", "1-2", "2-0", "0-2"]


def stored_predictions(conn):
    """Latest prediction per fixture, newest tag order matching the app."""
    rows = conn.execute(
        """SELECT p.fixture_id, p.home_xg, p.away_xg, p.score_grid,
                  h.name AS home, a.name AS away, f.gameweek
           FROM predictions p
           JOIN fixtures f ON f.id = p.fixture_id
           JOIN teams h ON h.id = f.home_team_id
           JOIN teams a ON a.id = f.away_team_id
           WHERE f.season = ?
             AND p.id = (SELECT MAX(id) FROM predictions
                         WHERE fixture_id = p.fixture_id)
           ORDER BY p.home_xg + p.away_xg""", (CURRENT_SEASON,)).fetchall()
    return [dict(r) for r in rows]


def live_predictions(conn, exclude, want):
    """Grids for fixtures that have no stored prediction yet.

    Only GW1 is predicted at rest, which is not enough spread to tell a close
    matchup from a mismatch. These run through the same Predictor the app uses
    with store=False, so nothing is written and no prediction row is created.
    """
    if want <= 0:
        return []
    from backend.predict import Predictor
    rows = conn.execute(
        """SELECT f.id, h.name AS home, a.name AS away, f.gameweek
           FROM fixtures f JOIN teams h ON h.id = f.home_team_id
           JOIN teams a ON a.id = f.away_team_id
           WHERE f.season = ? AND f.id NOT IN ({})
           ORDER BY f.gameweek, f.id""".format(
               ",".join("?" * len(exclude)) or "NULL"),
        (CURRENT_SEASON, *exclude)).fetchall()
    rows = rows[:want]
    print(f"(no stored grid for {len(rows)} of these — predicting live, "
          f"nothing is written)", file=sys.stderr)
    p = Predictor(conn)
    out = []
    for r in rows:
        pred = p.predict_fixture(r["id"], store=False)
        out.append({"fixture_id": r["id"], "home": r["home"],
                    "away": r["away"], "gameweek": r["gameweek"],
                    "home_xg": pred["home_xg"], "away_xg": pred["away_xg"],
                    "score_grid": json.dumps(pred["grid"])})
    return out


def sample(rows, n):
    """n fixtures spread evenly over the total-xG ordering, so the sample
    spans close matchups and mismatches instead of clustering."""
    if len(rows) <= n:
        return rows
    idx = np.linspace(0, len(rows) - 1, n).round().astype(int)
    return [rows[i] for i in dict.fromkeys(idx.tolist())]


def cell(grid, score):
    i, j = (int(x) for x in score.split("-"))
    return grid[i][j]


def poisson_cell(lam, mu, score):
    """What plain independent Poisson would give — no DC tau, no draw boost.
    The reference point for judging whether the shipped grid is distorted."""
    i, j = (int(x) for x in score.split("-"))
    return float(poisson.pmf(i, lam) * poisson.pmf(j, mu))


def main():
    n = 20
    if "-n" in sys.argv:
        n = int(sys.argv[sys.argv.index("-n") + 1])

    conn = connect()
    pool = stored_predictions(conn)
    # a wide pool sampled down beats predicting exactly n, since the spread
    # that matters (close vs mismatch) is only known after the xG is computed
    pool += live_predictions(conn, [r["fixture_id"] for r in pool],
                             max(0, 3 * n - len(pool)))
    pool.sort(key=lambda r: r["home_xg"] + r["away_xg"])
    rows = sample(pool, n)
    if not rows:
        print("no stored predictions — run scripts/refresh_week.py first")
        return

    print(f"Scoreline audit — {len(rows)} fixtures, stored grids, "
          f"season {CURRENT_SEASON}")
    print("gaps are (top - second) in percentage points\n")

    hdr = (f"{'fixture':<34}{'xG h':>6}{'xG a':>6}  " +
           "".join(f"{s:>7}" for s in WATCH) +
           f"   {'top 4 scorelines':<44}{'gap':>6}")
    print(hdr)
    print("-" * len(hdr))

    tops, gaps, shares = [], [], []
    for r in rows:
        grid = json.loads(r["score_grid"])
        flat = sorted(((f"{i}-{j}", p) for i, row in enumerate(grid)
                       for j, p in enumerate(row)),
                      key=lambda x: -x[1])
        top4 = flat[:4]
        gap = 100 * (flat[0][1] - flat[1][1])
        tops.append(flat[0][0])
        gaps.append(gap)
        shares.append(100 * flat[0][1])
        name = f"{r['home']} v {r['away']}"[:32]
        watched = "".join(f"{100 * cell(grid, s):>7.1f}" for s in WATCH)
        top_s = " ".join(f"{s} {100 * p:.1f}" for s, p in top4)
        print(f"{name:<34}{r['home_xg']:>6.2f}{r['away_xg']:>6.2f}  "
              f"{watched}   {top_s:<44}{gap:>6.1f}")

    print(f"\nTop scoreline share: mean {np.mean(shares):.1f}%  "
          f"min {min(shares):.1f}%  max {max(shares):.1f}%")
    print(f"Gap to second:       mean {np.mean(gaps):.1f}pp  "
          f"min {min(gaps):.1f}pp  max {max(gaps):.1f}pp")
    counts = {}
    for s in tops:
        counts[s] = counts.get(s, 0) + 1
    print("Top scoreline was:   " + ", ".join(
        f"{s} in {c}/{len(tops)}" for s, c in
        sorted(counts.items(), key=lambda x: -x[1])))

    # Where 1-1 sits relative to plain Poisson. The DC tau correction and the
    # fitted draw boost both touch 1-1 specifically, so if anything were
    # inflating it disproportionately, it would show up as a large ratio here.
    print("\nShipped grid vs plain Poisson on the same xG "
          "(ratio > 1 = the corrections lift that cell)\n")
    print(f"{'fixture':<34}" + "".join(f"{s:>8}" for s in WATCH))
    ratios = {s: [] for s in WATCH}
    for r in rows:
        grid = json.loads(r["score_grid"])
        lam, mu = r["home_xg"], r["away_xg"]
        line = ""
        for s in WATCH:
            ref = poisson_cell(lam, mu, s)
            ratio = cell(grid, s) / ref if ref > 0 else float("nan")
            ratios[s].append(ratio)
            line += f"{ratio:>8.3f}"
        name = f"{r['home']} v {r['away']}"[:32]
        print(f"{name:<34}{line}")
    print(f"{'mean':<34}" +
          "".join(f"{np.mean(ratios[s]):>8.3f}" for s in WATCH))

    # Would the corrections change WHICH scoreline leads? If they only ever
    # flip near-ties, they are not what makes 1-1 come out on top.
    flips = []
    for r in rows:
        grid = json.loads(r["score_grid"])
        lam, mu = r["home_xg"], r["away_xg"]
        n_g = len(grid)
        raw = [(f"{i}-{j}", float(poisson.pmf(i, lam) * poisson.pmf(j, mu)))
               for i in range(n_g) for j in range(n_g)]
        raw_top = max(raw, key=lambda x: x[1])
        shipped_top = max(((f"{i}-{j}", p) for i, row in enumerate(grid)
                           for j, p in enumerate(row)), key=lambda x: x[1])
        if raw_top[0] != shipped_top[0]:
            flips.append(f"{r['home']} v {r['away']}: "
                         f"{raw_top[0]} -> {shipped_top[0]}")
    print(f"\nTop scoreline changed by the corrections in "
          f"{len(flips)}/{len(rows)} fixtures")
    for f in flips:
        print(f"  {f}")


if __name__ == "__main__":
    main()
