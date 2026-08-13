"""Sanity checks for the Dixon-Coles engine and adjustment layers."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.dixon_coles import DixonColes
from models.fatigue import fatigue_score, fatigue_multiplier
from models.travel import travel_multiplier, congestion_multipliers


def synthetic_matches(strong="Strong", weak="Weak", n_rounds=40):
    """Strong beats weak often; two filler teams keep the fit identified."""
    rng = np.random.default_rng(7)
    teams = [strong, weak, "Mid1", "Mid2"]
    strength = {strong: 1.9, weak: 0.7, "Mid1": 1.2, "Mid2": 1.2}
    rows = []
    day = np.datetime64("2024-08-01")
    for r in range(n_rounds):
        for i, h in enumerate(teams):
            for a in teams:
                if h == a:
                    continue
                lam = strength[h] / strength[a] * 1.3
                mu = strength[a] / strength[h]
                rows.append({"home": h, "away": a,
                             "fthg": rng.poisson(lam), "ftag": rng.poisson(mu),
                             "date": str(day + np.timedelta64(r * 7 + i, "D"))})
    return rows


def fitted():
    return DixonColes().fit(synthetic_matches())


def test_grid_sums_to_one():
    m = fitted()
    grid = m.score_grid(1.5, 1.1)
    assert abs(grid.sum() - 1.0) < 1e-9


def test_outcome_probs_sum_to_one():
    m = fitted()
    p = m.outcome_probs(1.4, 1.2)
    assert abs(sum(p) - 1.0) < 1e-9


def test_stronger_team_has_higher_win_prob():
    m = fitted()
    pred = m.predict("Strong", "Weak")
    assert pred["p_home"] > pred["p_away"]
    # and symmetric: still favoured away
    pred2 = m.predict("Weak", "Strong")
    assert pred2["p_away"] > pred2["p_home"]


def test_home_advantage_increases_home_win_prob():
    m = fitted()
    assert m.gamma > 1.0
    lam, mu = m.expected_goals("Mid1", "Mid2")
    with_ha = m.outcome_probs(lam, mu)[0]
    without_ha = m.outcome_probs(lam / m.gamma, mu)[0]
    assert with_ha > without_ha


def test_fatigue_monotonic_and_bounded():
    fresh = fatigue_score([(10, 90)])
    tired = fatigue_score([(1, 90), (4, 90), (7, 90)])
    assert 0 <= fresh < tired <= 100
    assert fatigue_multiplier(0) == 1.0
    assert fatigue_multiplier(100) < 1.0
    assert fatigue_multiplier(100) >= 0.9


def test_travel_discount_small_and_capped():
    near, _, _ = travel_multiplier((51.48, -0.19), (51.55, -0.11))  # London derby
    far, _, _ = travel_multiplier((50.73, -1.84), (54.98, -1.62))   # Bmth->NUFC
    assert near > far >= 0.97
    missing, d, partial = travel_multiplier((None, None), (51.5, -0.1))
    assert missing == 1.0 and partial


def test_congestion_only_hits_short_rested_side():
    hm, am, flags = congestion_multipliers(3, 7)
    assert hm < 1.0 and am == 1.0 and flags
    hm, am, flags = congestion_multipliers(7, 7)
    assert hm == am == 1.0 and not flags
