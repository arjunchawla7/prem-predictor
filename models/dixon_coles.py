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
from scipy.special import gammaln
from scipy.stats import poisson

MAX_GOALS = 10  # scoreline grid is (0..MAX_GOALS) x (0..MAX_GOALS)


def _pois_ll(k, lam):
    """Poisson log-density, valid for non-integer k.

    Identical to poisson.logpmf on integers; the gammaln form is what lets the
    same likelihood accept an xG target (which is continuous). Strictly this is
    a quasi-likelihood for non-integer k, not a normalised density — fine here,
    since the k! term does not depend on the parameters being optimised.
    """
    return k * np.log(lam) - lam - gammaln(k + 1.0)


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
    blend_w: float = 1.0        # target = blend_w*goals + (1-blend_w)*xG.
                                # 1.0 = goals (original), 0.0 = pure xG.
    draw_boost: float = 1.0     # multiplier on the drawn cells of the score
                                # grid, fitted per refit on the training
                                # window. 1.0 = plain Dixon-Coles. See
                                # fit_draw_boost() for why it is needed and
                                # what it does NOT fix.
    draw_scale: float = 1.0     # fixed out-of-sample correction stacked on
                                # draw_boost. draw_boost is fitted IN-sample,
                                # where the model is already better calibrated
                                # than it will be on unseen fixtures, so it
                                # under-corrects on its own. Scaling the grid
                                # diagonal and scaling the final draw
                                # probability are algebraically identical
                                # (both renormalise over the same total), so
                                # the two multipliers simply compose.
    prior_strength: float = 0.0  # Gaussian prior pulling log attack/defence
                                # toward the league average. 0 = unregularised
                                # (original behaviour); see fit() for why this
                                # matters for barely-observed teams.

    def fit(self, matches, as_of=None):
        """matches: iterable of dicts with keys
           home, away (team keys), fthg, ftag, date (np.datetime64/str),
           and home_xg/away_xg when blend_w < 1.
        as_of: only matches strictly before this date are used (backtesting
           guard against leakage); weights decay relative to it.

        With blend_w < 1 the attack/defence/gamma fit runs against the blended
        target, then rho is refit alone against ACTUAL goals with everything
        else held fixed — the low-score correction describes the discreteness
        of real scorelines, which a continuous target carries no information
        about."""
        import pandas as pd
        df = pd.DataFrame(matches)
        df["date"] = pd.to_datetime(df["date"])
        if as_of is not None:
            as_of = pd.Timestamp(as_of)
            df = df[df["date"] < as_of]
        else:
            as_of = df["date"].max()
        # prior_strength note: with no prior, a team carrying one or two
        # matches in the window is not identifiable and its rating runs off to
        # the boundary. Observed live: Sunderland, one match into 2025-26, fit
        # to defence=0.0001, which priced Burnley-Sunderland at p(home)=4.5e-6
        # and put 3.6% of the season's entire log-loss on that single game. The
        # penalty is a zero-mean Gaussian on log attack/defence, so it barely
        # moves a team with 100+ matches of evidence and dominates for a team
        # with one.
        self.teams = sorted(set(df["home"]) | set(df["away"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        h = df["home"].map(idx).to_numpy()
        a = df["away"].map(idx).to_numpy()
        hg = df["fthg"].to_numpy(float)
        ag = df["ftag"].to_numpy(float)
        days = (as_of - df["date"]).dt.days.to_numpy(float)
        w = np.exp(-self.xi * days)

        if self.blend_w >= 1.0:
            th, ta = hg, ag
        else:
            # Rows with no xG fall back to their goal count, so a partially
            # covered training window degrades per-match instead of dropping.
            hx = pd.to_numeric(df.get("home_xg"), errors="coerce").to_numpy(float)
            ax = pd.to_numeric(df.get("away_xg"), errors="coerce").to_numpy(float)
            hx = np.where(np.isnan(hx), hg, hx)
            ax = np.where(np.isnan(ax), ag, ax)
            b = self.blend_w
            th, ta = b * hg + (1 - b) * hx, b * ag + (1 - b) * ax

        exact = self.blend_w >= 1.0

        def nll(p):
            atk = np.exp(p[:n])
            dfc = np.exp(p[n:2 * n])
            gamma = np.exp(p[2 * n])
            rho = p[2 * n + 1]
            lam = atk[h] * dfc[a] * gamma
            mu = atk[a] * dfc[h]
            ll = _pois_ll(th, lam) + _pois_ll(ta, mu)
            if exact:
                ll = ll + np.log(np.clip(tau(hg, ag, lam, mu, rho), 1e-10, None))
            # soft identifiability: mean(log attack) = 0
            obj = -(w * ll).sum() + 1e3 * p[:n].mean() ** 2
            if self.prior_strength:
                obj = obj + self.prior_strength * (p[:2 * n] ** 2).sum()
            return obj

        p0 = np.concatenate([np.zeros(n), np.zeros(n), [np.log(1.3), -0.05]])
        res = minimize(nll, p0, method="L-BFGS-B",
                       options={"maxiter": 400})
        p = res.x
        self.attack = {t: float(np.exp(p[idx[t]])) for t in self.teams}
        self.defence = {t: float(np.exp(p[n + idx[t]])) for t in self.teams}
        self.gamma = float(np.exp(p[2 * n]))
        self.converged = bool(res.success)

        if exact:
            self.rho = float(p[2 * n + 1])
        else:
            lam = np.exp(p[:n])[h] * np.exp(p[n:2 * n])[a] * self.gamma
            mu = np.exp(p[:n])[a] * np.exp(p[n:2 * n])[h]

            def rho_nll(r):
                return -(w * np.log(np.clip(
                    tau(hg, ag, lam, mu, r[0]), 1e-10, None))).sum()

            rres = minimize(rho_nll, [-0.05], method="L-BFGS-B",
                            bounds=[(-0.3, 0.3)])
            self.rho = float(rres.x[0])
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
        if self.draw_boost * self.draw_scale != 1.0:
            grid[np.diag_indices(MAX_GOALS + 1)] *= (self.draw_boost
                                                     * self.draw_scale)
        grid = np.clip(grid, 0, None)
        return grid / grid.sum()

    def outcome_probs_vec(self, lam, mu, draw_boost=None):
        """(n, 3) W/D/L probabilities for arrays of lam/mu.

        Same maths as score_grid + outcome_probs, evaluated for every fixture
        at once — fitting draw_boost needs the whole training set per
        objective evaluation, which is far too slow one grid at a time.
        """
        b = (self.draw_boost if draw_boost is None else draw_boost) * self.draw_scale
        g = np.arange(MAX_GOALS + 1)
        lam = np.asarray(lam, float)[:, None]
        mu = np.asarray(mu, float)[:, None]
        ph = poisson.pmf(g[None, :], lam)          # (n, G) home goals
        pa = poisson.pmf(g[None, :], mu)           # (n, G) away goals
        grid = ph[:, :, None] * pa[:, None, :]     # (n, G, G)

        # tau applies to the four low-score cells only
        l0, m0 = lam[:, 0], mu[:, 0]
        grid[:, 0, 0] *= 1 - l0 * m0 * self.rho
        grid[:, 0, 1] *= 1 + l0 * self.rho
        grid[:, 1, 0] *= 1 + m0 * self.rho
        grid[:, 1, 1] *= 1 - self.rho
        if b != 1.0:
            idx = np.arange(MAX_GOALS + 1)
            grid[:, idx, idx] *= b

        grid = np.clip(grid, 0, None)
        grid /= grid.sum(axis=(1, 2), keepdims=True)
        i, j = np.tril_indices(MAX_GOALS + 1, -1)
        p_home = grid[:, i, j].sum(axis=1)
        p_draw = np.einsum("nii->n", grid)
        return np.column_stack([p_home, p_draw, 1 - p_home - p_draw])

    def fit_draw_boost(self, matches, as_of=None, bounds=(0.8, 2.0)):
        """Fit draw_boost by maximum likelihood on observed W/D/L results.

        Why this is needed: a Poisson score grid systematically under-produces
        draws. Dixon-Coles' rho already corrects the four lowest scorelines,
        but draws at 2-2 and above get no correction at all, and the model
        treats each team's strength as a known point estimate when it is
        really an estimate with spread — both push probability away from the
        diagonal. Measured on 2025-26: mean predicted draw .234 against an
        actual .274.

        What it does NOT fix: draws still will not become the argmax pick.
        The gap between the draw probability and the leading outcome is about
        6.6 points at its narrowest, so no plausible boost closes it — and
        forcing one by inflating further makes calibration worse, not better.
        This improves the honesty of the probabilities, not the headline
        hit-rate. Those are different things and only the first is a defect.

        Fitted on the same time-decayed training window as everything else,
        so it never sees the season it is scored on.
        """
        import pandas as pd
        df = pd.DataFrame(matches)
        df["date"] = pd.to_datetime(df["date"])
        if as_of is not None:
            as_of = pd.Timestamp(as_of)
            df = df[df["date"] < as_of]
        else:
            as_of = df["date"].max()
        df = df[df["home"].isin(self.attack) & df["away"].isin(self.attack)]
        if df.empty:
            return self

        lam = np.array([self.attack[h] * self.defence[a] * self.gamma
                        for h, a in zip(df["home"], df["away"])])
        mu = np.array([self.attack[a] * self.defence[h]
                       for h, a in zip(df["home"], df["away"])])
        hg = df["fthg"].to_numpy(float)
        ag = df["ftag"].to_numpy(float)
        actual = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
        w = np.exp(-self.xi * (as_of - df["date"]).dt.days.to_numpy(float))

        # draw_scale is the out-of-sample correction that sits ON TOP of this
        # fit. If it were left applied here, the in-sample optimiser would
        # simply shrink draw_boost to cancel it out and the correction would
        # vanish. Neutralise it for the duration of the fit.
        saved_scale, self.draw_scale = self.draw_scale, 1.0
        try:
            def nll(p):
                probs = self.outcome_probs_vec(lam, mu, draw_boost=float(p[0]))
                hit = probs[np.arange(len(probs)), actual]
                return -(w * np.log(np.clip(hit, 1e-12, None))).sum()

            res = minimize(nll, [1.15], method="L-BFGS-B", bounds=[bounds])
            self.draw_boost = float(res.x[0])
        finally:
            self.draw_scale = saved_scale
        return self

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
