"""Layer 1 configuration, shared by the live predictor and every backtest.

Single source so the numbers on /performance always describe the model that is
actually making predictions. All three were settled by the 2025-26
walk-forward pass in scripts/accuracy_pass.py:

  accuracy  .4658 -> .4868      brier .6188 -> .6142     log-loss 1.0607 -> 1.0237
"""

# Pinned rather than "every season in the matches table". 2019-20..2021-22 are
# loaded for the situational baselines, and training on all seven seasons
# scored worse than four (Brier .6155 vs .6142) even with time decay. Left
# implicit, a routine data pull silently changed the live model.
TRAIN_SEASONS = ["2223", "2324", "2425", "2526"]

# Fit against xG instead of goals. Scorelines are noisy match to match; xG
# gives steadier attack/defence ratings. Goals-xG blends all scored worse than
# pure xG, so this is 0.0 and not an intermediate weight.
BLEND_W = 0.0

# Gaussian prior pulling log attack/defence toward the league average.
# Barely moves a team with 150 matches of evidence; dominates for a team with
# one. Without it, Sunderland — one match into 2025-26 — fitted to
# defence=0.0001 and priced Burnley to win at 4.5e-6, putting 3.6% of the
# season's entire log-loss on a single fixture.
PRIOR_STRENGTH = 5.0

# Time decay per day. Re-swept AFTER the shrinkage fix (the original sweep was
# measured with that pathology still in the frame); 0.001 remained best.
XI = 0.001

# --- draw calibration -------------------------------------------------------
# A Poisson score grid under-produces draws: rho corrects only the four lowest
# scorelines, 2-2 and above get nothing, and point-estimate team strengths push
# probability off the diagonal. Measured on 2025-26, mean predicted draw was
# .234 against an actual .274.
#
# Two one-parameter corrections, which compose:
#   draw_boost   fitted per refit on the training window (in-sample, so it
#                under-corrects on its own — see fit_draw_boost)
#   DRAW_SCALE   fixed, fitted on 2023-24 + 2024-25 WALK-FORWARD output, i.e.
#                on out-of-sample predictions, which is where the real
#                shortfall shows up
#
# 2025-26 holdout:
#   current model                   acc .4868  brier .6142  logloss 1.0237  p̄(draw) .2343
#   + draw boost                    acc .4868  brier .6133  logloss 1.0222  p̄(draw) .2432
#   + boost & scale (ships now)     acc .4868  brier .6123  logloss 1.0203  p̄(draw) .2580
#
# Read that honestly: this is a CALIBRATION fix worth .0019 Brier, not an
# accuracy fix. Accuracy is unchanged and a draw is still never the argmax
# pick — the nearest a draw comes to leading is 6.6 points, and inflating far
# enough to close that makes Brier and log-loss worse. Full six-parameter
# vector scaling was also tested and DISCARDED: it fixed the draw buckets but
# dragged home/away calibration with it (acc .4684, brier .6168), because the
# home-win rate differs season to season and the correction did not transfer.
#
# Refit DRAW_SCALE (scripts/draw_pass.py) as more completed seasons land.
DRAW_SCALE = 1.0824

# Layer 2 style matchup adjustment, OFF.
#
# It used to earn its place (.4684 vs .4658 baseline). Re-run against the fixed
# Layer 1 it now costs accuracy on every metric:
#
#   Layer 1 alone      accuracy .487   brier .6142   log-loss 1.0237
#   Layer 1 + style    accuracy .466   brier .6169   log-loss 1.0273
#
# The most likely reading is that style was partly compensating for the
# unshrunk ratings, and now double-counts a correction Layer 1 already makes.
# Same rule as everything else here: it does not improve calibration, so it
# does not feed the prediction. Set True to put it back.
STYLE_ENABLED = False

# 50/50 model + market blend, published as a SEPARATE labeled output and never
# as the model's own number. Needs closing/market odds for the fixture; without
# them the blend is simply absent. Backtested on 2025-26: accuracy .4947,
# brier .6085, log-loss 1.0148 — better than the model alone (.4868 / .6142 /
# 1.0237), but it is not the model's unassisted accuracy and is not reported as
# such. The market alone scored .4947 / .6077 / 1.0118.
MARKET_BLEND_W = 0.5


# Backtest frames the performance page publishes, in display order. Shared
# with scripts/make_seed.py so the deploy bundle carries exactly these and not
# the accuracy-pass experiment frames that also land in data/backtests/.
BACKTEST_FILES = {
    "Layer 1 — season-average": "layer1_season_avg_2526.csv",
    "Layer 1 + lineup/fatigue/travel": "layer1_lineup_weighted_2526.csv",
    "Layer 1 + Layer 2 (style, currently off)": "layer2_style_2526.csv",
}


def configure(model):
    """Apply the adopted settings to a DixonColes instance."""
    model.blend_w = BLEND_W
    model.prior_strength = PRIOR_STRENGTH
    model.xi = XI
    model.draw_scale = DRAW_SCALE
    return model


def make_model():
    """Fresh, configured DixonColes whose fit() also fits the draw boost.

    The factory run_backtest expects, and what the live predictor uses — so
    the two cannot diverge.
    """
    from models.dixon_coles import DixonColes
    model = configure(DixonColes())
    plain_fit = model.fit

    def fit(matches, as_of=None):
        plain_fit(matches, as_of=as_of)
        return model.fit_draw_boost(matches, as_of=as_of)

    model.fit = fit
    return model
