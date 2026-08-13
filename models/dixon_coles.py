"""Dixon-Coles match model (Layer 1 core).

The model (Dixon & Coles, 1997):

  Home goals X ~ Poisson(lambda),  lambda = attack_home * defence_away * gamma
  Away goals Y ~ Poisson(mu),      mu     = attack_away * defence_home

  - attack_i  > 0 : team i's goal-scoring strength (bigger = better attack)
  - defence_i > 0 : team i's goal-CONCEDING factor (smaller = better defence)
  - gamma     > 0 : home advantage, applied to the home side's scoring rate

  Independence between X and Y is a poor fit for low-scoring games, so DC
  multiply the joint probability of the four low scores (0-0, 1-0, 0-1, 1-1)
  by a correction tau(x, y; lambda, mu, rho). rho < 0 shifts probability
  toward draws in tight games; rho = 0 recovers plain double-Poisson.

  Time decay: each historical match is weighted w = exp(-xi * days_ago/365)
  in the log-likelihood, so recent form counts more. xi is set by convention
  (~0.0065/day in the original paper's scale); we expose it per-fit.

Fitting: maximise the weighted log-likelihood over
  {attack_i, defence_i, gamma, rho}
with the identifiability constraint mean(attack) = 1 (otherwise attack*k,
defence/k gives the same fit). scipy.optimize.minimize on the negative
log-likelihood, parameters log-transformed so positivity is automatic.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10  # scoreline grid is (0..MAX_GOALS) x (0..MAX_GOALS)


def tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correction factor (vectorised)."""
    t = np.ones_like(lam, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


@dataclass
class DixonColes:
    teams: list = field(default_factory=list)
    attack: dict = field(default_factory=dict)
    defence: dict = field(default_factory=dict)
    gamma: float = 1.3          # home advantage
    rho: float = -0.05          # low-score correction
    xi: float = 0.001           # time decay per day (chosen by 2025-26
                                # walk-forward sweep; response was flat across
                                # 0.0005-0.005, this had the best log-loss)

    def fit(self, matches, as_of=None):
        """matches: iterable of dicts with keys
           home, away (team keys), fthg, ftag, date (np.datetime64/str).
        as_of: only matches strictly before this date are used (backtesting
           guard against leakage); weights decay relative to it."""
        import pandas as pd
        df = pd.DataFrame(matches)
        df["date"] = pd.to_datetime(df["date"])
        if as_of is not None:
            as_of = pd.Timestamp(as_of)
            df = df[df["date"] < as_of]
        else:
            as_of = df["date"].max()
        self.teams = sorted(set(df["home"]) | set(df["away"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        h = df["home"].map(idx).to_numpy()
        a = df["away"].map(idx).to_numpy()
        hg = df["fthg"].to_numpy(float)
        ag = df["ftag"].to_numpy(float)
        days = (as_of - df["date"]).dt.days.to_numpy(float)
        w = np.exp(-self.xi * days)

        def nll(p):
            atk = np.exp(p[:n])
            dfc = np.exp(p[n:2 * n])
            gamma = np.exp(p[2 * n])
            rho = p[2 * n + 1]
            lam = atk[h] * dfc[a] * gamma
            mu = atk[a] * dfc[h]
            ll = (poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
                  + np.log(np.clip(tau(hg, ag, lam, mu, rho), 1e-10, None)))
            # soft identifiability: mean(log attack) = 0
            return -(w * ll).sum() + 1e3 * p[:n].mean() ** 2

        p0 = np.concatenate([np.zeros(n), np.zeros(n), [np.log(1.3), -0.05]])
        res = minimize(nll, p0, method="L-BFGS-B",
                       options={"maxiter": 400})
        p = res.x
        self.attack = {t: float(np.exp(p[idx[t]])) for t in self.teams}
        self.defence = {t: float(np.exp(p[n + idx[t]])) for t in self.teams}
        self.gamma = float(np.exp(p[2 * n]))
        self.rho = float(p[2 * n + 1])
        self.converged = bool(res.success)
        return self

    # ---- prediction ----

    def expected_goals(self, home, away):
        """Baseline (season-average lineup) expected goals for a fixture."""
        lam = self.attack[home] * self.defence[away] * self.gamma
        mu = self.attack[away] * self.defence[home]
        return lam, mu

    def score_grid(self, lam, mu):
        """(MAX_GOALS+1)^2 matrix of P(home=i, away=j), DC-corrected, renormalised."""
        g = np.arange(MAX_GOALS + 1)
        ph = poisson.pmf(g, lam)[:, None]
        pa = poisson.pmf(g, mu)[None, :]
        grid = ph * pa
        # apply tau to the four low-score cells
        for x in (0, 1):
            for y in (0, 1):
                grid[x, y] *= tau(np.array(x), np.array(y),
                                  np.array(lam), np.array(mu), self.rho)
        grid = np.clip(grid, 0, None)
        return grid / grid.sum()

    def outcome_probs(self, lam, mu):
        """(p_home, p_draw, p_away) from the scoreline grid."""
        grid = self.score_grid(lam, mu)
        p_home = float(np.tril(grid, -1).sum())   # i > j
        p_draw = float(np.trace(grid))
        p_away = float(np.triu(grid, 1).sum())    # j > i
        return p_home, p_draw, p_away

    def predict(self, home, away, lam_mult=1.0, mu_mult=1.0):
        """Full prediction; lam_mult/mu_mult are where lineup, fatigue,
        travel and style adjustments plug in (multipliers on expected goals)."""
        lam, mu = self.expected_goals(home, away)
        lam, mu = lam * lam_mult, mu * mu_mult
        grid = self.score_grid(lam, mu)
        p_h, p_d, p_a = self.outcome_probs(lam, mu)
        best = np.unravel_index(np.argmax(grid), grid.shape)
        return {
            "home_xg": lam, "away_xg": mu,
            "p_home": p_h, "p_draw": p_d, "p_away": p_a,
            "grid": grid, "most_likely_score": (int(best[0]), int(best[1])),
        }
