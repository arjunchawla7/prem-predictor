"""Candidate situational features — measured before they are believed.

Football talk is full of team "records" ("X never lose at home leading at
half-time"). Nearly all of them are a league-wide base rate wearing a team's
shirt: *every* team wins most games they lead at half-time. A pattern only
carries information if the team's rate beats what any team does in the same
situation, by more than the sample can explain.

So nothing here is a model input. This module measures; adoption is a separate,
evidenced decision:

  1. compute the rate from the database, over whatever window the data covers
     (never from a quoted anecdote)
  2. compare it to the league-wide rate for the SAME situation
  3. test the gap against binomial noise, and flag small samples outright
  4. only then may it be backtested as a feature — and only kept if it improves
     Brier/log-loss on held-out data

Two flags matter as much as the numbers:

  small_sample   fewer than MIN_RELIABLE occurrences. Report it, never let it
                 move a prediction, however dramatic the streak sounds.
  pre_match      whether the situation is even KNOWABLE before kickoff. A
                 half-time scoreline is not. Such a pattern can never be a
                 pre-match feature no matter how strong it looks — at most it
                 is an in-play note.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest

MIN_RELIABLE = 30      # below this, a rate is reported but never trusted
MIN_USABLE = 20        # below this it is not even worth testing

# Greater London bounding box, applied to the stadium coordinates already
# stored for the travel adjustment — derives the derby set from data rather
# than a hand-kept club list that goes stale on promotion.
LONDON = dict(lat=(51.28, 51.70), lon=(-0.52, 0.32))

# A league match at least this many days after the team's previous one, mid
# season, is the far side of an international break.
BREAK_GAP_DAYS = 12


def team_match_frame(conn) -> pd.DataFrame:
    """One row per team per match: result plus the context columns situations
    are filtered on. Two rows per match, one from each side's perspective."""
    m = pd.read_sql_query(
        """SELECT m.id, m.season, m.date, m.fthg, m.ftag, m.hthg, m.htag,
                  h.name AS home, a.name AS away,
                  h.lat AS h_lat, h.lon AS h_lon, a.lat AS a_lat, a.lon AS a_lon
           FROM matches m
           JOIN teams h ON h.id = m.home_team_id
           JOIN teams a ON a.id = m.away_team_id
           ORDER BY m.date""", conn)
    m["date"] = pd.to_datetime(m["date"])

    def in_london(lat, lon):
        return (lat.between(*LONDON["lat"]) & lon.between(*LONDON["lon"]))

    m["london_derby"] = (in_london(m["h_lat"], m["h_lon"])
                         & in_london(m["a_lat"], m["a_lon"]))

    sides = []
    for is_home in (1, 0):
        gf, ga = ("fthg", "ftag") if is_home else ("ftag", "fthg")
        hgf, hga = ("hthg", "htag") if is_home else ("htag", "hthg")
        d = pd.DataFrame({
            "match_id": m["id"], "season": m["season"], "date": m["date"],
            "team": m["home"] if is_home else m["away"],
            "opp": m["away"] if is_home else m["home"],
            "is_home": is_home,
            "gf": m[gf], "ga": m[ga], "ht_gf": m[hgf], "ht_ga": m[hga],
            "london_derby": m["london_derby"],
        })
        sides.append(d)
    f = pd.concat(sides, ignore_index=True)

    f["result"] = np.where(f.gf > f.ga, "W", np.where(f.gf == f.ga, "D", "L"))
    ht = f.ht_gf - f.ht_ga
    f["ht_state"] = np.where(ht.isna(), None,
                             np.where(ht > 0, "lead",
                                      np.where(ht == 0, "level", "trail")))

    # days since that team's previous league match in the dataset
    f = f.sort_values(["team", "date"])
    f["rest_days"] = f.groupby("team")["date"].diff().dt.days
    f["after_break"] = (f["rest_days"] >= BREAK_GAP_DAYS) & (f["date"].dt.month != 8)
    return f.sort_values("date").reset_index(drop=True)


@dataclass
class Finding:
    label: str
    team: str
    outcome: str            # what is being counted, e.g. "win"
    n: int                  # team occurrences
    rate: float
    base_n: int             # league-wide occurrences in the same situation
    base_rate: float
    lift: float             # rate - base_rate
    p_value: float          # team rate vs league base rate, two-sided binomial
    window: str
    pre_match: bool
    small_sample: bool
    verdict: str

    def as_row(self):
        return {"label": self.label, "team": self.team, "outcome": self.outcome,
                "n": self.n, "rate": round(self.rate, 4),
                "base_n": self.base_n, "base_rate": round(self.base_rate, 4),
                "lift": round(self.lift, 4), "p_value": round(self.p_value, 4),
                "window": self.window, "pre_match": self.pre_match,
                "small_sample": self.small_sample, "verdict": self.verdict}


def evaluate(frame, mask, team, label, outcome="win", pre_match=True,
             alpha=0.05) -> Finding:
    """Team rate vs league-wide rate for the same situation.

    mask: boolean Series over `frame` defining the situation (team-agnostic).
    outcome: 'win', 'not_lose' or 'draw'.
    """
    sit = frame[mask]
    hit = {"win": sit.result.eq("W"),
           "not_lose": sit.result.ne("L"),
           "draw": sit.result.eq("D")}[outcome]

    own = sit.team.eq(team)
    n, k = int(own.sum()), int(hit[own].sum())
    # baseline excludes the team itself, so the comparison is "this team vs
    # everyone else" rather than a team partly compared against its own record
    base = ~own
    base_n, base_k = int(base.sum()), int(hit[base].sum())
    rate = k / n if n else float("nan")
    base_rate = base_k / base_n if base_n else float("nan")

    p = (binomtest(k, n, base_rate).pvalue
         if n and base_n and 0 < base_rate < 1 else 1.0)
    window = (f"{sit.date.min():%Y-%m-%d} to {sit.date.max():%Y-%m-%d}"
              if len(sit) else "no data")
    small = n < MIN_RELIABLE

    if n < MIN_USABLE:
        verdict = f"unusable — only {n} occurrences"
    elif not pre_match:
        verdict = "not usable pre-match — state unknown at kickoff"
    elif small:
        verdict = f"unreliable — {n} occurrences (< {MIN_RELIABLE})"
    elif p >= alpha:
        verdict = "no team-specific signal — matches the league base rate"
    else:
        verdict = "candidate — beats base rate; must still pass a backtest"

    return Finding(label, team, outcome, n, rate, base_n, base_rate,
                   rate - base_rate, float(p), window, pre_match, small,
                   verdict)


def model_control(conn, frame, mask, team, outcome="win", model=None,
                  pre_match=True):
    """The control that actually matters: does the pattern beat what the MODEL
    already expects, not just the league average?

    The league base rate is the wrong yardstick on its own. "Arsenal win 63% of
    home London derbies vs a 35% league rate" is mostly "Arsenal are good and
    several London clubs are not" — team strength leaks straight into the lift.
    Comparing the realised rate against the Dixon-Coles probability for those
    exact fixtures removes that, leaving only what the ratings fail to explain.

    Fitted on the full sample, so the control is mildly optimistic about the
    model — which is the conservative direction here: it makes a pattern harder
    to claim as signal, not easier.
    """
    from models.dixon_coles import DixonColes
    from models.backtest import load_matches, promoted_prior

    if not pre_match:
        # The model probability is a PRE-match quantity. Scoring it against a
        # situation defined by the half-time score compares it to outcomes
        # conditioned on information it never had, so the residual is
        # guaranteed large and means nothing. Refuse rather than return a
        # number that reads like evidence.
        return {"n": int((mask & frame.team.eq(team)).sum()),
                "actual": None, "model_expected": None, "residual_lift": None,
                "z": None, "explained_by_ratings": None,
                "error": "control invalid: situation is not known pre-match, "
                         "so a pre-match probability is not a valid baseline"}

    if model is None:
        model = DixonColes().fit(load_matches(conn).to_dict("records"))

    sit = frame[mask & frame.team.eq(team)]
    hit = {"win": sit.result.eq("W"), "not_lose": sit.result.ne("L"),
           "draw": sit.result.eq("D")}[outcome]

    exp = []
    for _, r in sit.iterrows():
        for t in (r.team, r.opp):
            if t not in model.attack:
                model.attack[t], model.defence[t] = promoted_prior(model)
        home, away = (r.team, r.opp) if r.is_home else (r.opp, r.team)
        p_h, p_d, p_a = model.outcome_probs(*model.expected_goals(home, away))
        p_win, p_lose = (p_h, p_a) if r.is_home else (p_a, p_h)
        exp.append({"win": p_win, "not_lose": p_win + p_d, "draw": p_d}[outcome])

    exp = np.array(exp, dtype=float)
    actual = float(hit.mean()) if len(sit) else float("nan")
    expected = float(exp.mean()) if len(exp) else float("nan")
    # sd of the mean of independent Bernoullis with differing p
    sd = float(np.sqrt((exp * (1 - exp)).sum())) / len(exp) if len(exp) else np.nan
    z = (actual - expected) / sd if sd else float("nan")
    return {"n": len(sit), "actual": round(actual, 4),
            "model_expected": round(expected, 4),
            "residual_lift": round(actual - expected, 4),
            "z": round(z, 2),
            "explained_by_ratings": bool(abs(z) < 2)}


# --- situation library -----------------------------------------------------
# (name, mask builder, pre_match, default outcome)
SITUATIONS = {
    "home_leading_at_ht": (
        lambda f: (f.is_home == 1) & (f.ht_state == "lead"),
        False, "win"),
    "home": (lambda f: f.is_home == 1, True, "win"),
    "london_derby": (lambda f: f.london_derby, True, "win"),
    "london_derby_home": (
        lambda f: f.london_derby & (f.is_home == 1), True, "win"),
    "after_international_break": (
        lambda f: f.after_break, True, "win"),
    "away_after_international_break": (
        lambda f: f.after_break & (f.is_home == 0), True, "win"),
}


def run(frame, team, situation, outcome=None) -> Finding:
    mask_fn, pre_match, default_outcome = SITUATIONS[situation]
    return evaluate(frame, mask_fn(frame), team, situation,
                    outcome or default_outcome, pre_match)


def scan(frame, situation, outcome=None, min_n=MIN_USABLE) -> pd.DataFrame:
    """Every team's finding for one situation — the honest way to look at a
    'record', because it shows the same number for all 20 clubs at once."""
    rows = [run(frame, t, situation, outcome).as_row()
            for t in sorted(frame.team.unique())]
    out = pd.DataFrame(rows)
    return out[out.n >= min_n].sort_values("lift", ascending=False)
