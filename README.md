# Prem Predictor

Local Premier League match-prediction tool for the 2026-27 season. Python backend + SQLite + small Flask web frontend. Personal project — built for working end-to-end, not scale.

**Honest expectations:** the model backtests at **48.7%** exact-outcome accuracy on 2025-26 (uniform guessing = 33%). It does not beat the market and is not supposed to: bookmakers' closing odds score **49.5%** on the same 380 matches, which is the practical ceiling for this kind of model rather than a number to chase. The performance page keeps this in your face on purpose.

Two structural limits worth knowing up front, stated once here and not repeated below: a draw is *never* the model's top pick (the Poisson grid caps draw probability near 30% while draws are ~27% of results), and lineup/fatigue/travel weighting is measured but has never improved calibration on the 2025-26 holdout — it ships anyway, clearly labeled, for what-if use. Situational "records" are measured but never fed into a prediction — see below.

## Setup

```
py -m venv .venv
.venv\Scripts\python -m pip install pandas numpy scipy requests flask pytest
```

First-time data build (order matters):

```
.venv\Scripts\python scripts\seed_teams.py        # teams + stadium coords
.venv\Scripts\python scripts\download_history.py  # 4 seasons of results CSVs
.venv\Scripts\python scripts\load_matches.py      # -> SQLite
.venv\Scripts\python scripts\pull_understat.py    # xG, player stats, rosters (~10 min first run)
.venv\Scripts\python scripts\pull_fixtures.py     # 2026-27 fixtures + gameweeks
```

## Weekly routine

```
# a few days before the gameweek — results refresh + Mode 1 (ideal XI) predictions
.venv\Scripts\python scripts\refresh_week.py

# ~1h before kickoff, manually — Mode 2 confirmed-lineup rebuild
.venv\Scripts\python scripts\rebuild_confirmed.py        # next GW, or pass a GW number
```

Both are manual by design. The one exception is the odds refresh, which has to
run unattended to be worth anything — see below.

## Web frontend

```
.venv\Scripts\python backend\app.py    # -> http://127.0.0.1:5000
```

- **Gameweek view** — each card reads in three parts: the **result** probability as the headline (this is the prediction), the **market** as three outlined boxes in the same columns (decimal price, implied %, drift since the opening line), and the **scoreline distribution** as the top five scorelines. The single most likely scoreline is deliberately *not* stated on its own — at 11-15% it is the highest of a crowded field, not a call, and headlining it overstated the model's confidence. Plus xG, lineup mode badge (Provisional / Final / Manual) and partial-data warnings.
- **Match page** (any fixture card is a link) — the same three parts, then the full scoreline heatmap, position-grouped lineups on a pitch, teamsheet (manager, derived preferred formation, tactical traits), summer arrivals, factor list for that specific prediction, and prediction history.
- **Manual lineup (Mode 3)** — "edit XI" on any fixture: pick exactly 11 per side (or leave one side untouched). Locks the fixture to manual — automatic rebuilds skip it until you hit "reset to auto".
- **/performance** — the single detail page: headline numbers, how the model works, layer-by-layer backtest, settled experiments, calibration, provisional-vs-final drift, the scoreline audit, the 4+ goal caveat, known limitations, situational-records findings, data sources and gaps, and the data-pull log. Linked from the header of every page and from a one-line footer on the gameweek view.

The accuracy figures deliberately do **not** appear on the gameweek or match pages. They used to open the gameweek view as four stat tiles, which asked a first-time visitor to weigh the model's limitations before looking at a single prediction. Nothing was softened in moving them — the detail page carries more than it did before, not less.

## Market odds (optional)

Register free at the-odds-api.com, then `setx ODDS_API_KEY <key>` (this persists to the user environment; already-open shells won't see it). Without a key the odds pull skips gracefully and the UI says so.

Odds are pulled repeatedly rather than once, because a single early snapshot is the market's opinion from before team news. Every pull appends a row to `market_odds` (keyed by fixture + timestamp), so price history accumulates locally and the UI can show drift since the opening line. Implied probabilities and the model+market blend are derived at read time in `backend/market.py`, which means a line that moves after a prediction was generated still updates what the page shows.

**Scheduling it (Windows).** `scripts/pull_odds.py` is built to be run on a dumb hourly timer — its guards decide whether to actually spend anything, before any HTTP request is made:

```
schtasks /Create /TN "PremPredictor odds refresh" ^
  /TR "\"C:\path\to\prem-predictor\.venv\Scripts\pythonw.exe\" \"C:\path\to\prem-predictor\scripts\pull_odds.py\"" ^
  /SC HOURLY /MO 1 /F

schtasks /Query  /TN "PremPredictor odds refresh" /V /FO LIST    # inspect
schtasks /Delete /TN "PremPredictor odds refresh" /F             # remove
```

`pythonw.exe` keeps a console window from flashing every hour; the tradeoff is that stdout goes nowhere, so the record of each run lives in `pull_log` (visible on /performance). The task runs as the logged-on user and inherits `ODDS_API_KEY` from the user environment — running it while logged out would need a stored Windows password, which isn't set up.

Guards, all overridable with `--force`:

| flag | default | effect |
|---|---|---|
| `--window-hours` | 72 | only call the API when a kickoff is this close |
| `--min-interval` | 180 | minutes that must pass between snapshots |
| `--reserve` | 25 | refuse to spend the last N API credits |

`refresh_week.py` takes the opening snapshot with a widened window; `rebuild_confirmed.py` refreshes again immediately before the Mode 2 rebuild, so the final prediction is blended against a line that has seen the same confirmed XIs the model just did.

**Free-tier economics.** Cost per call is `markets × regions`, and one `/odds` call prices *every* upcoming fixture at once — so a refresh costs 2 credits (h2h × uk,eu) whether one match or ten are pending. The free tier is 500 credits/month ≈ 250 refreshes; the guards above use roughly 8 per matchweek. Set `ODDS_REGIONS=uk` to halve it. Remaining quota is read from the `x-requests-remaining` header and logged after every call.

Two things checked against the provider's docs, recorded so they aren't re-litigated: the `/historical/*` endpoints are documented as paid-only **and** cost 10× credits, but nothing here needs them since snapshots are stored locally; and in-play/live odds are *not* obviously paid-gated for this provider — `/odds` is documented as returning "upcoming and live games". Live odds are **not implemented** and would need feasibility and coverage checked with a real key first.

## How the model works

- **Layer 1 — Dixon-Coles** (`models/dixon_coles.py`): per-team attack/defence strengths + home advantage + low-score correction, maximum-likelihood fit (scipy) with exponential time-decay weighting. Full scoreline grid via Poisson; W/D/L from the grid. Fitted against **xG rather than goals**, with ratings **shrunk toward the league average** — both settled by backtest, see below. Settings live in `models/config.py`, shared by the live predictor and every backtest so the performance page always describes the model that is actually predicting.
- **Player ratings** (`models/player_ratings.py`): previous-season understat per-90s, z-scored within position group, quintile tiers 1-5. Low-minute players get a flagged provisional average, never an invented number.
- **Fatigue** (`models/fatigue.py`): decayed sum of recent league minutes → 0-100; discounts a player's rating up to −10%. League minutes only — no cup/European minutes are tracked, so fatigue is understated for teams in Europe.
- **Lineup weighting** (`models/lineup.py`): a named XI's mean effective rating vs the team's typical lineup → capped ±8% nudge on expected goals.
- **Travel/congestion** (`models/travel.py`): haversine distance (≤3% discount) + short-rest comparison (2%).
- **Layer 2 — style** (`models/style.py`): k-means style buckets from shot proxies; matchup adjustment learned from *Layer-1 residuals* (so it can't just rediscover "good beats bad"); capped ±0.15 xG. **Currently switched off** (`STYLE_ENABLED`) — it costs accuracy against the corrected Layer 1.
- **Situational records** (`models/situational.py`): measures claimed team "records" against the league base rate for the same situation, against the model's own expectation, and against binomial noise. Nothing here feeds a prediction; see below.
- **Team profile** (`models/formation.py`): preferred formation *derived* from the position codes of each side's real starting XIs over its last 30 matches (4-2-3-1, 3-5-2, …), reported with its share and runner-up shapes; plus tactical trait labels from league percentile ranks on shot volume, xG per shot, shots conceded and corners. No possession or pressing data exists in this database, so no claim is made about either.
- **Defensive ratings are a proxy**, not true defensive stats — built from team goals conceded + xG buildup, since no source here carries tackles/interceptions data.

### Validation findings (2025-26 walk-forward backtest)

Run `python scripts/accuracy_pass.py` to reproduce; full results land in `data/backtests/accuracy_pass.csv`.

| variant | accuracy | Brier | log-loss | verdict |
|---|---|---|---|---|
| Layer 1, goals target, unshrunk (original) | 46.6% | 0.6188 | 1.0607 | — |
| + rating shrinkage (PRIOR_STRENGTH=5) | 47.1% | 0.6151 | 1.0248 | kept |
| + xG fitting target (BLEND_W=0) | 47.9% | 0.6150 | 1.0247 | kept |
| both together | 48.7% | 0.6142 | 1.0237 | kept |
| **+ draw correction (ships now)** | **48.7%** | **0.6123** | **1.0203** | **kept** |
| + Layer 2 style | 46.6% | 0.6169 | 1.0273 | switched off |
| *market closing odds alone* | *49.5%* | *0.6077* | *1.0118* | *reference, not a model* |
| model + market 50/50 blend | 49.5% | 0.6085 | 1.0148 | kept, labeled separately |

Several time-decay rates, training-window lengths (up to 7 seasons), and goals/xG blend ratios were also swept and discarded — none beat the settings above. Full sweep values are in `accuracy_pass.py` if you want to rerun them.

Two things worth knowing about these numbers:

- **The biggest single gain was a bug fix, not a new feature.** Teams with only one or two matches in the training window were fitting to the parameter boundary and pricing some fixtures at absurd extremes — one early-season Sunderland fixture alone carried over 3% of the whole season's log-loss. Rating shrinkage fixes this boundary case directly.
- **Do not read 50%+ into any of this.** The market itself only manages 49.5% here. Every variant above was scored once against this holdout and kept or discarded — none were iterated against it to push the number up.

### Is 1-1 dominating the grid? (audited, no)

`python scripts/scoreline_audit.py -n 20` samples fixtures across the range of expected goals and prints the watched cells, the actual top scorelines, and the gap to second. Across 20 fixtures the top scoreline averages **12.6%** (range 9.6-15.2%) with a mean **3.7pp** gap to second, and it correctly yields to 2-0 in mismatches — ordinary Poisson behaviour, nothing running away.

Against plain Poisson on the same xG, the shipped grid lifts 1-1 by ×1.20. That decomposes exactly into the fitted draw multiplier (1.175, applied to *every* drawn scoreline, not 1-1 specifically) and the Dixon-Coles `tau(1,1) = 1-rho`, both already validated for calibration above. It changes which scoreline leads in 2 of 20 fixtures, both raw dead heats inside 0.2pp — and both happened to be promoted-team fixtures, because the rough Championship-derived ratings are what put those matches in the mismatch band where 2-0 climbs to tie 1-1. **The frequent 1-1 was a presentation problem, not a modelling one**, which is why the fix was to stop headlining the single scoreline rather than to touch the grid.

### Situational "records"

`python scripts/situational_report.py` measures a claimed pattern properly rather than trusting the anecdote. Worked example — Man Utd leading at half-time at Old Trafford, computed over 2019-2026:

| | n | rate | league (other clubs) | lift | p |
|---|---|---|---|---|---|
| win | 55 | 81.8% | 75.3% | +6.5pp | 0.35 |
| not lose | 55 | **100%** | 92.0% | +8.0pp | 0.02 |

Genuinely unbeaten in 55 — and still not usable: Liverpool are 74/74 and Leicester 26/26 on the same measure, the league base rate is already 92%, and a half-time scoreline isn't knowable before kickoff anyway. The framework also requires beating the *model's own expectation*, not just the league average: Arsenal's home London-derby record looks like +27.9pp (p=0.001) against the league rate but is +0.0pp (z=−0.3) once Arsenal's rating is accounted for. **No situational pattern currently qualifies as a model input.**

## Data sources & known gaps

- Results/shots/cards: football-data.co.uk, via its GitHub mirror (datasets/football-datasets) — same CSVs, same schema as the origin site.
- Market closing odds: football-data.co.uk direct (`scripts/pull_odds_history.py`) — the only source here carrying odds, so it's the only thing that can back a market backtest. Seasons 2019-20 through 2025-26, all 2660 matches priced.
- xG, player minutes/rosters/position codes: understat.com (internal JSON API; needs an X-Requested-With header). Note: understat's `roster_in` field is **not** minutes played and reads "0" for substitutes too — starters are identified by position code ≠ Sub, not by this field.
- Fixtures, gameweeks, confirmed lineups, squads, managers, crests and player photos: official PL Pulselive API (footballapi.pulselive.com).
- **Gaps, flagged in-app rather than hidden:** no cup/European minutes (fatigue understated for teams in Europe); no true defensive stats (proxy ratings only); promoted teams start on a weakest-3-average prior until Championship data (below) or top-flight history fills in; FPL's API is unreachable from this network's firewall.

## Repo layout

```
backend/   db.py (schema), predict.py (engine), app.py (Flask),
           market.py (overround removal, blend, odds snapshot history)
models/    config (settled settings), dixon_coles, player_ratings, fatigue,
           lineup, travel, style, formation, situational, backtest, evaluate
scripts/   data pulls, refresh_week, rebuild_confirmed, pull_odds (scheduled),
           backtests, sweeps, accuracy_pass, situational_report,
           pull_odds_history, scoreline_audit, tail_check
frontend/  index.html (gameweek), match.html (fixture detail), performance.html
data/      prem.db, raw CSVs/JSON cache, backtests/, ca_bundle.pem
tests/     model sanity tests (pytest)
```

## Tests

```
.venv\Scripts\python -m pytest tests -q
```

## Deploying (Render, free tier)

The repo ships `Procfile`, `requirements.txt`, and `render.yaml` for a one-click deploy. A `data/seed/` bundle (a copy of the DB + backtest CSVs, tracked in git unlike `data/prem.db` itself) seeds a fresh deployment so the site has real predictions immediately instead of an empty schema.

1. Push this repo to GitHub (see below if it isn't already there).
2. On [render.com](https://render.com): **New → Blueprint**, point it at the repo. Render reads `render.yaml` and creates the web service.
3. First deploy takes a few minutes (installs deps, then `seed_disk.py` copies `data/seed/prem.db` into place at build time). After that you get a public URL.

**Persistence caveat (confirmed, not hypothetical):** Render's free tier supports neither a persistent disk nor a preDeploy step, so the database lives in the container's own ephemeral filesystem — every deploy or restart resets it back to the `data/seed/` snapshot committed in the repo. Fine for browsing predictions, not for accumulating new ones between deploys. A paid "Starter" instance (~$7/mo) adds a real persistent disk if that matters.

**Updating deployed data:** there's no scheduled job on the free tier, and nothing persists between deploys anyway. To push newer data live: run `refresh_week.py` locally as usual, then `scripts\make_seed.py` to refresh `data/seed/`, commit, and push — the next deploy picks it up. (Or run `rebuild_confirmed.py` similarly before a gameweek.) This is a personal-project workaround, not a real pipeline.

### Pushing to GitHub for the first time

```
gh repo create prem-predictor --private --source=. --push

# or manually:
git remote add origin https://github.com/<you>/prem-predictor.git
git push -u origin master
```

### Championship (second-tier) data

`scripts/pull_championship.py` loads E1 results from football-data.co.uk, so a promoted side can be rated from its actual second-tier form instead of a generic weakest-3-average prior. Rows are tagged `matches.division='E1'`; `load_matches()` returns top-flight only unless asked otherwise, and the live predictor filters to E0 explicitly.

**Championship rows are never in the training pool.** That was tested and is clearly worse (accuracy .4763 vs .4868, Brier .6201 vs .6123, same 380 fixtures). They're used only to seed teams the top-flight fit has never seen, via one pooled fit rescaled onto the top-flight scale using the 25 clubs present in both.

The three clubs promoted into 2025-26 are the only honest test of this — project them from Championship data alone, then compare against what they turned out to be:

| team | projected atk/def | actual atk/def |
|---|---|---|
| Leeds | 1.135 / 1.058 | 1.048 / 1.207 |
| Burnley | 0.886 / 0.985 | 0.767 / 1.570 |
| Sunderland | 0.767 / 1.131 | 0.832 / 1.195 |
| *generic weakest-3 prior* | *0.762 / 1.664* | — |

Cross-league beats the generic prior on both axes (mean absolute error: attack .090 vs .120, defence .266 vs .340) but is **systematically optimistic about defence** — promoted teams concede more than the projection expects, badly so for Burnley. n=3 is far too few to calibrate a correction without inventing one, so the bias is documented rather than fitted out.

Known gaps this doesn't fix: no xG for Championship rows (understat doesn't cover it, so cross-league ratings are fit on a different target than top-flight ones — goals only); no lineups or player data (football-data.co.uk carries results only, so formation derivation and player ratings for a promoted side still fall back to season-average); FBref has never been used as a source for this project.

## About

Personal project. No license/topics set.

### Open question: league-level scoring (revisit ~December 2026)

The model has no term for how many goals the division scores overall. The
identifiability constraint pins mean(log attack) to a constant, so a
league-wide shift gets absorbed into relative team ratings rather than tracked
in its own right.

2025-26 exposed this. A 4+ goal side occurred in **7.6%** of matches against a
**13.9%** four-season average, and the model kept predicting **14.8%** — 56
expected, 29 seen, a 4-sigma overshoot, worst in mismatches (19.1% predicted
vs 7.9% actual). `python scripts/tail_check.py` reproduces it.

What it is not:

- **Not a thin tail.** The model already over-predicts 6, 7 and 8+ goal
  matches, so a negative binomial would push it further the wrong way. That
  idea was tested on the numbers and dropped.
- **Not fixable by time decay.** `tail_check.py --sweep` shows forgetting old
  matches eight times faster (half-life 693d to 87d) recovers only a third of
  the gap, and within a season the estimate drifts slightly *up* as
  low-scoring evidence accumulates, because rating spread widens faster than
  the level falls.

**Deliberately not fixed on one season.** Once 2026-27 has ~150 matches
played, re-run `scripts/tail_check.py` with `TARGET` set to `2627`. If the low
rate persists, the candidate fix is a per-season (or decayed rolling)
league-level term multiplying both expected-goal figures, fitted alongside the
ratings — backtested to the same Brier/log-loss standard as everything else.
If 2025-26 was a one-off, leave it alone. The match page carries a caveat on
the 4+ figure and /performance#tail explains it to readers meanwhile.
