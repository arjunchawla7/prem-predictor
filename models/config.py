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
    return model


def make_model():
    """Fresh, configured DixonColes — the factory run_backtest expects."""
    from models.dixon_coles import DixonColes
    return configure(DixonColes())
