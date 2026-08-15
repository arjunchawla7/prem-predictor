"""Poisson count models for corners and cards — fitted independently.

Deliberately NOT derived from the goals grid. Corners and cards come from
different mechanisms than goals (territory and game state for corners,
discipline and refereeing for cards), so they get their own fit on their own
data. Sharing the goals model's structure would smuggle in the assumption that
a team good at scoring is proportionally good at winning corners.

The model is multiplicative, the same shape as Layer 1 minus the low-score
correction:

    E[count for team i vs j] = base × produce_i × concede_j × side_factor
                               (× referee_factor, cards only)

Fitted by iterative proportional fitting with exponential time-decay weights
and a gamma prior shrinking every factor toward 1. Shrinkage is not optional
here — it was the single biggest accuracy gain on the goals model, and the
same boundary problem applies to a team with three matches in the window.

Cards model yellows only. Reds average 0.058 per team per match, which is too
rare to fit and would be swamped by the prior anyway.
"""
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

XI = 0.0015          # time decay; half-life ~460 days
PRIOR = 12.0         # gamma prior strength on team factors, in "effective events"
REF_PRIOR = 25.0     # referees get seen ~25 times a season — shrink harder
ITERS = 40


def _days_between(a, b):
    return abs((b - a).days)


def _parse(d):
    if isinstance(d, datetime):
        return d
    return datetime.fromisoformat(str(d).replace(" ", "T")[:19])


class CountModel:
    """One count per team per match (corners, or yellow cards).

    rows: dicts with date, home, away, home_count, away_count and optionally
    referee. Fit once per backtest refit, cheaply — this is arithmetic, not an
    optimiser.
    """

    def __init__(self, use_referee=False, xi=XI, prior=PRIOR,
                 ref_prior=REF_PRIOR):
        self.use_referee = use_referee
        self.xi, self.prior, self.ref_prior = xi, prior, ref_prior
        self.produce = defaultdict(lambda: 1.0)   # team's own count rate
        self.concede = defaultdict(lambda: 1.0)   # count it induces in others
        self.referee = defaultdict(lambda: 1.0)
        self.base_home = self.base_away = 1.0

    # ---- fitting ----

    def fit(self, rows, as_of=None):
        rows = [r for r in rows if r.get("home_count") is not None
                and r.get("away_count") is not None]
        if not rows:
            raise ValueError("no rows with counts to fit")
        as_of = _parse(as_of) if as_of else max(_parse(r["date"]) for r in rows)
        w = [math.exp(-self.xi * _days_between(_parse(r["date"]), as_of))
             for r in rows]

        # side baselines: home and away rates differ systematically (home
        # sides win more corners, away sides collect more cards)
        tw = sum(w)
        self.base_home = sum(x * r["home_count"] for x, r in zip(w, rows)) / tw
        self.base_away = sum(x * r["away_count"] for x, r in zip(w, rows)) / tw

        teams = {r["home"] for r in rows} | {r["away"] for r in rows}
        for t in teams:
            self.produce[t] = self.concede[t] = 1.0
        refs = {r.get("referee") for r in rows if r.get("referee")}
        for rf in refs:
            self.referee[rf] = 1.0

        for _ in range(ITERS):
            self._update_factor(rows, w, "produce")
            self._update_factor(rows, w, "concede")
            if self.use_referee and refs:
                self._update_referee(rows, w)
            self._renormalise(teams)
        return self

    def _expected(self, r, skip=None):
        """Expected (home, away) counts, optionally leaving one factor out so
        the IPF step can solve for it."""
        rf = self.referee[r["referee"]] if (
            self.use_referee and r.get("referee")) else 1.0
        h = self.base_home * rf
        a = self.base_away * rf
        if skip != "produce":
            h *= self.produce[r["home"]]
            a *= self.produce[r["away"]]
        if skip != "concede":
            h *= self.concede[r["away"]]
            a *= self.concede[r["home"]]
        return h, a

    def _update_factor(self, rows, w, which):
        num = defaultdict(float)
        den = defaultdict(float)
        for x, r in zip(w, rows):
            eh, ea = self._expected(r, skip=which)
            if which == "produce":
                num[r["home"]] += x * r["home_count"]; den[r["home"]] += x * eh
                num[r["away"]] += x * r["away_count"]; den[r["away"]] += x * ea
            else:
                # a team's concede factor is set by what its OPPONENTS record
                num[r["away"]] += x * r["home_count"]; den[r["away"]] += x * eh
                num[r["home"]] += x * r["away_count"]; den[r["home"]] += x * ea
        target = self.produce if which == "produce" else self.concede
        for t in list(target):
            # gamma prior: prior extra "expected" events pull the factor to 1
            target[t] = (num[t] + self.prior) / (den[t] + self.prior)

    def _update_referee(self, rows, w):
        num = defaultdict(float)
        den = defaultdict(float)
        for x, r in zip(w, rows):
            rf = r.get("referee")
            if not rf:
                continue
            saved, self.referee[rf] = self.referee[rf], 1.0
            eh, ea = self._expected(r)
            self.referee[rf] = saved
            num[rf] += x * (r["home_count"] + r["away_count"])
            den[rf] += x * (eh + ea)
        for rf in list(self.referee):
            self.referee[rf] = ((num[rf] + self.ref_prior)
                                / (den[rf] + self.ref_prior))

    def _renormalise(self, teams):
        """Keep the factors identifiable — their geometric mean is pinned to 1
        so the level lives in base_home/base_away, not in the team factors."""
        for target in (self.produce, self.concede):
            vals = [target[t] for t in teams if target[t] > 0]
            if not vals:
                continue
            g = math.exp(sum(math.log(v) for v in vals) / len(vals))
            if g > 0:
                for t in teams:
                    target[t] /= g

    # ---- prediction ----

    def expected_counts(self, home, away, referee=None):
        """(home, away) expected counts. Unknown teams and unknown referees
        fall back to the neutral factor 1.0, which is the league average."""
        rf = self.referee[referee] if (
            self.use_referee and referee and referee in self.referee) else 1.0
        h = self.base_home * self.produce[home] * self.concede[away] * rf
        a = self.base_away * self.produce[away] * self.concede[home] * rf
        return float(h), float(a)

    def referee_factor(self, referee):
        """How this referee compares to an average one (1.0 = average).
        None when unknown, which is the honest answer before an appointment
        is published."""
        if not self.use_referee or not referee or referee not in self.referee:
            return None
        return float(self.referee[referee])


def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def total_distribution(lam, mu, top=25):
    """P(total = k). Two independent Poissons sum to Poisson(lam+mu)."""
    total = lam + mu
    return [poisson_pmf(k, total) for k in range(top + 1)]


def prob_at_least(lam, mu, threshold):
    """P(total >= threshold), from the same summed Poisson."""
    dist = total_distribution(lam, mu, top=max(60, threshold + 40))
    return float(sum(dist[threshold:]))


def count_stats(lam, mu, lines):
    """Expected totals plus P(total >= line) for each requested integer line."""
    return {
        "home": round(lam, 2), "away": round(mu, 2),
        "total": round(lam + mu, 2),
        "lines": {str(n): round(prob_at_least(lam, mu, n), 4) for n in lines},
        "distribution": [round(p, 4)
                         for p in total_distribution(lam, mu, top=20)],
    }
