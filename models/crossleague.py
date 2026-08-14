"""Rate a team with no top-flight history from its Championship results.

A promoted side otherwise gets `promoted_prior` — the average of the three
weakest fitted teams — which is the same number for every promoted club
regardless of how it actually played.

Championship ratings cannot be read straight onto the top-flight scale: they
are estimated against Championship opposition, so a side that scores freely
there would look like a good Premier League team. The bridge is the clubs
appearing in BOTH divisions inside the window (Burnley, Leeds and Sunderland
went up; Ipswich, Leicester and Southampton came down). One pooled fit over
E0+E1 puts everyone on a single scale anchored by those clubs, and the pooled
ratings are then shifted so the shared clubs agree with the top-flight-only
fit.

Scope is deliberately narrow: this ONLY supplies ratings for teams the
top-flight fit has never seen. Established clubs keep the validated E0-only
ratings, so this cannot move predictions for anyone else.

Evidence, and its limits: on the 2025-26 holdout this changed exactly ONE
match of 380. Only Sunderland arrived with no top-flight history in the
window, and once they had played a week the fit had real data and the prior
stopped applying. Direction was positive (Brier .6123 -> .6117) but a single
fixture is not evidence, so treat this as better-principled rather than
demonstrated. Pooling E1 into the training set proper WAS measurable, and was
clearly worse (accuracy .4763 vs .4868, Brier .6201 vs .6123) — hence
prior-only.
"""
import numpy as np
import pandas as pd

MIN_SHARED = 5      # clubs needed in both fits to trust the scale alignment


def aligned_pooled_model(pooled_matches, e0_model, make_model, as_of=None):
    """Fit over E0+E1 and express the result on `e0_model`'s scale.

    Returns None when the two fits share too few clubs to align them, which is
    the safe outcome — callers fall back to the generic promoted prior.
    """
    df = pd.DataFrame(pooled_matches)
    if df.empty:
        return None
    if as_of is not None:
        df = df[pd.to_datetime(df["date"]) < pd.Timestamp(as_of)]
        if df.empty:
            return None

    pooled = make_model().fit(df.to_dict("records"), as_of=as_of)
    shared = [t for t in pooled.attack if t in e0_model.attack]
    if len(shared) < MIN_SHARED:
        return None

    d_atk = np.mean([np.log(e0_model.attack[t]) - np.log(pooled.attack[t])
                     for t in shared])
    d_def = np.mean([np.log(e0_model.defence[t]) - np.log(pooled.defence[t])
                     for t in shared])
    pooled.attack = {t: float(v * np.exp(d_atk))
                     for t, v in pooled.attack.items()}
    pooled.defence = {t: float(v * np.exp(d_def))
                      for t, v in pooled.defence.items()}
    pooled.shared_clubs = len(shared)
    return pooled
