# Prem Predictor

Local Premier League match-prediction tool for the 2026-27 season.
Python backend + SQLite + small Flask web frontend. Personal project —
built for working end-to-end, not scale.

**Honest expectations:** the model backtests at **48.7%** exact-outcome
accuracy on 2025-26 (uniform guessing = 33%). It does not beat the market and
is not supposed to: bookmakers' closing odds score **49.5%** on the same 380
matches, which is the practical ceiling for this kind of model rather than a
number to chase. The performance page keeps this in your face on purpose.

Two structural limits worth knowing before reading any accuracy figure:
a draw is *never* the model's top pick (a Poisson grid caps draw probability
near 30% while draws are ~27% of results), and situational "records" are
measured but never fed into a prediction — see `models/situational.py`.

## Setup

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install pandas numpy scipy requests flask pytest
```

First-time data build (order matters):

```powershell
.venv\Scripts\python scripts\seed_teams.py        # teams + stadium coords
.venv\Scripts\python scripts\download_history.py  # 4 seasons of results CSVs
.venv\Scripts\python scripts\load_matches.py      # -> SQLite
.venv\Scripts\python scripts\pull_understat.py    # xG, player stats, rosters (~10 min first run)
.venv\Scripts\python scripts\pull_fixtures.py     # 2026-27 fixtures + gameweeks
```

## Weekly routine

```powershell
# a few days before the gameweek — results refresh + Mode 1 (ideal XI) predictions
.venv\Scripts\python scripts\refresh_week.py

# ~1h before kickoff, manually — Mode 2 confirmed-lineup rebuild
.venv\Scripts\python scripts\rebuild_confirmed.py        # next GW, or pass a GW number
```

No OS scheduling is set up — both are manual by design.

## Web frontend

```powershell
.venv\Scripts\python backend\app.py    # -> http://127.0.0.1:5000
```

- **Gameweek view** — W/D/L bars, top scorelines, xG, lineup mode badge
  (Provisional / Final / Manual), starting XI with rating tier + fatigue,
  market odds when available, partial-data warnings.
- **Match page** (`match page →` on any fixture) — scoreline heatmap,
  position-grouped lineups with player photos, teamsheet (manager, derived
  preferred formation, tactical traits), summer arrivals, prediction history.
- **Manual lineup (Mode 3)** — "edit XI" on any fixture: pick exactly 11 per
  side (or leave one side untouched). Locks the fixture to manual — automatic
  rebuilds skip it until you hit "reset to auto".
- **/performance** — layer-by-layer backtest results, calibration table,
  provisional-vs-final drift tracking, data-pull log, known limitations.

## Market odds (optional)

Register free at the-odds-api.com, then `setx ODDS_API_KEY <key>`.
Without a key the odds pull skips gracefully and the UI says so.

## How the model works

- **Layer 1 — Dixon-Coles** (`models/dixon_coles.py`): per-team attack/defence
  strengths + home advantage + low-score correction, maximum-likelihood fit
  (scipy) with exponential time-decay weighting. Full scoreline grid via
  Poisson; W/D/L from the grid. Math commented in the file. Fitted against
  **xG rather than goals**, with the ratings **shrunk toward the league
  average** — both settled by backtest, see below. Settings live in
  `models/config.py`, shared by the live predictor and every backtest so the
  performance page always describes the model that is actually predicting.
- **Player ratings** (`models/player_ratings.py`): previous-season understat
  per-90s, z-scored within position group, quintile tiers 1-5. Low-minute
  players get a flagged provisional average, never an invented number.
- **Fatigue** (`models/fatigue.py`): decayed sum of recent league minutes →
  0-100; discounts a player's rating up to −10%.
- **Lineup weighting** (`models/lineup.py`): a named XI's mean effective
  rating vs the team's typical lineup → capped ±8% nudge on expected goals.
- **Travel/congestion** (`models/travel.py`): haversine distance (≤3%
  discount) + short-rest comparison (2%).
- **Layer 2 — style** (`models/style.py`): k-means style buckets from shot
  proxies; matchup adjustment learned from *Layer-1 residuals* (so it can't
  just rediscover "good beats bad"); capped ±0.15 xG. **Currently switched
  off** (`STYLE_ENABLED`) — it costs accuracy against the corrected Layer 1.
- **Situational records** (`models/situational.py`): measures claimed team
  "records" against the league base rate for the same situation, against the
  model's own expectation, and against binomial noise. Nothing here feeds a
  prediction; see below.
- **Team profile** (`models/formation.py`): preferred formation *derived*
  from the position codes of each side's real starting XIs over its last 30
  matches (4-2-3-1, 3-5-2, …), reported with its share and the runner-up
  shapes; plus tactical trait labels from league percentile ranks on shot
  volume, xG per shot, shots conceded and corners. No possession or pressing
  data exists in this database, so no claim is made about either.

### Validation findings (2025-26 walk-forward backtest)

Run `python scripts/accuracy_pass.py` to reproduce; results land in
`data/backtests/accuracy_pass.csv`.

| variant | accuracy | Brier | log-loss | verdict |
|---|---|---|---|---|
| Layer 1, goals target, unshrunk (was shipping) | 46.6% | 0.6188 | 1.0607 | — |
| + rating shrinkage (`PRIOR_STRENGTH=5`) | 47.1% | 0.6151 | 1.0248 | **kept** |
| + xG fitting target (`BLEND_W=0`) | 47.9% | 0.6150 | 1.0247 | **kept** |
| both together | 48.7% | 0.6142 | 1.0237 | **kept** |
| **+ draw correction (ships now)** | **48.7%** | **0.6123** | **1.0203** | **kept** |
| goals/xG blends (0.25 / 0.5 / 0.75) | 47.1–47.6% | 0.6149–0.6168 | 1.0250–1.0314 | discarded |
| time decay xi ∈ {.0003,.0005,.0018,.003,.005} | 46.3–47.1% | 0.6183–0.6241 | 1.0576–1.0710 | discarded — 0.001 already best |
| 7 training seasons instead of 4 | 46.8% | 0.6211 | 1.0590 | discarded |
| + Layer 2 style | 46.6% | 0.6169 | 1.0273 | **switched off** |
| *market closing odds alone* | *49.5%* | *0.6077* | *1.0118* | *reference, not a model* |
| model + market 50/50 blend | 49.5% | 0.6085 | 1.0148 | kept, **labeled separately** |

Notes on what these numbers mean:

- **The single biggest win was a bug, not a feature.** Teams with one or two
  matches in the training window were unidentifiable and fitted to the
  parameter boundary. Sunderland, one match into 2025-26, reached
  `defence=0.0001` and priced Burnley to win at 4.5e-6 — that one fixture
  carried 3.6% of the season's entire log-loss. `promoted_prior` only covered
  teams with *zero* history. Shrinkage fixes the 1-2 match case.
- **The xi sweep was re-run after that fix**, because the original sweep was
  measured with the pathology still in the frame. 0.001 survived.
- **Lineup/fatigue/travel remains unproven** (48.7% either way) and is still
  labeled as such on the fixture page.
- **Do not read 50%+ into any of this.** The market itself manages 49.5% here.
  Variants were scored once against this holdout and kept or discarded; none
  were iterated against it to push the number up.

### Situational "records"

`python scripts/situational_report.py` measures a claimed pattern properly.
Worked example — Man Utd leading at half-time at Old Trafford, computed over
2019-2026, not quoted from anecdote:

| | n | rate | league (other clubs) | lift | p |
|---|---|---|---|---|---|
| win | 55 | 81.8% | 75.3% | +6.5pp | 0.35 |
| not lose | 55 | **100%** | 92.0% | +8.0pp | 0.02 |

Genuinely unbeaten in 55 — and still not usable: Liverpool are 74/74 and
Leicester 26/26 on the same measure, the league base rate is already 92%, and
a half-time scoreline is not knowable before kickoff. The framework also
requires beating the *model's own expectation*, not just the league average:
Arsenal's home London-derby record looks like +27.9pp (p=0.001) against the
league rate but is +0.0pp (z=−0.3) once Arsenal's rating is accounted for.
**No situational pattern currently qualifies as a model input.**

## Data sources & known gaps

- Results/shots/cards: football-data.co.uk **via its GitHub mirror**
  (`datasets/football-datasets`) — the origin site *was* blocked as "Gambling"
  by the FortiGate firewall on this network. Same CSVs, same schema.
- Market closing odds: football-data.co.uk **direct** (`scripts/pull_odds_history.py`).
  The origin is reachable again, and it is the only source here carrying odds —
  the GitHub mirror strips those columns, and the-odds-api serves only
  *upcoming* fixtures, so neither can back a market backtest. Seasons
  2019-20..2025-26, all 2660 matches priced.
- xG, player minutes/rosters/position codes: understat.com (internal JSON
  API; needs `X-Requested-With` header). Note: understat's `roster_in` is
  **not** a minute and reads "0" for substitutes too — starters are the
  entries whose position code isn't `Sub`. Getting this wrong silently
  produces wrong starting XIs (it did, until `backfill_slots.py` repaired
  every stored appearance).
- Fixtures, gameweeks, confirmed lineups, squads, managers, crests and
  player photos: official PL Pulselive API (`footballapi.pulselive.com`).
- **Gaps (flagged in-app, not faked):** no cup/European minutes (fatigue
  understated for European teams); no true defensive stats (proxy ratings);
  promoted teams start on a weakest-3 prior; FPL API is FortiGuard-blocked.

### Network note (this machine)

All HTTPS is TLS-inspected by a FortiGate firewall. `scripts/net.py` gives
every puller a session that trusts `data/ca_bundle.pem` (certifi + the
FortiGate CA, exported from the live chain) and relaxes OpenSSL's
`VERIFY_X509_STRICT` (the FortiGate CA lacks an Authority Key Identifier).
On a normal network everything still works — the bundle is a superset of
certifi. Regenerate the bundle if the firewall's CA rotates.

## Repo layout

```
backend/   db.py (schema), predict.py (engine), app.py (Flask)
models/    config (settled settings), dixon_coles, player_ratings, fatigue,
           lineup, travel, style, formation, situational, backtest, evaluate
scripts/   data pulls, refresh_week, rebuild_confirmed, backtests, sweeps,
           accuracy_pass, situational_report, pull_odds_history
frontend/  index.html (gameweek), performance.html
data/      prem.db, raw CSVs/JSON cache, backtests/, ca_bundle.pem
tests/     model sanity tests (pytest)
```

## Tests

```powershell
.venv\Scripts\python -m pytest tests -q
```

## Deploying (Render, free tier)

The repo ships `Procfile`, `requirements.txt`, and `render.yaml` for a
one-click deploy. A `data/seed/` bundle (a copy of the DB + backtest CSVs,
tracked in git unlike `data/prem.db` itself) seeds a fresh deployment so the
site has real predictions immediately instead of an empty schema.

1. Push this repo to GitHub (see below if it isn't already there).
2. On [render.com](https://render.com): **New → Blueprint**, point it at the
   repo. Render reads `render.yaml` and creates the web service.
3. First deploy takes a few minutes (installs deps, then `seed_disk.py`
   copies `data/seed/prem.db` into place at build time). After that you get
   a public URL.

**Persistence caveat (confirmed, not hypothetical):** Render's free tier
supports neither a persistent disk nor a preDeploy step, so the database
lives in the container's own ephemeral filesystem — every deploy or restart
resets it back to the `data/seed/` snapshot committed in the repo. Fine for
browsing predictions, not for accumulating new ones between deploys. A paid
"Starter" instance (~$7/mo) adds a real persistent disk if that matters.

**Updating deployed data:** there's no scheduled job on the free tier, and
nothing persists between deploys anyway. To push newer data live: run
`refresh_week.py` locally as usual, then `scripts\make_seed.py` to refresh
`data/seed/`, commit, and push — the next deploy picks it up. (Or run
`rebuild_confirmed.py` similarly before a gameweek.) This is a
personal-project workaround, not a real pipeline.

### Pushing to GitHub for the first time

```powershell
gh repo create prem-predictor --private --source=. --push
# or manually:
git remote add origin https://github.com/<you>/prem-predictor.git
git push -u origin master
```

### Championship (second-tier) data

`scripts/pull_championship.py` loads E1 results from football-data.co.uk, so a
promoted side can be rated from its actual second-tier form instead of a
generic "average of the three weakest teams" prior. Rows are tagged
`matches.division='E1'`; `load_matches()` returns top-flight only unless asked
otherwise, and the live predictor filters to `E0` explicitly.

**Championship rows are never in the training pool.** That was tested and is
clearly worse (accuracy .4763 vs .4868, Brier .6201 vs .6123, same 380
fixtures). They are used only to seed teams the top-flight fit has never seen,
via one pooled fit rescaled onto the top-flight scale using the 25 clubs
present in both.

How good is the projection? The clubs that went up in 2025-26 are the only
honest test — project them from Championship data alone, then compare against
what they turned out to be:

| team | projected atk/def | actual atk/def |
|---|---|---|
| Leeds | 1.135 / 1.058 | 1.048 / 1.207 |
| Burnley | 0.886 / 0.985 | 0.767 / 1.570 |
| Sunderland | 0.767 / 1.131 | 0.832 / 1.195 |
| *generic weakest-3 prior* | *0.762 / 1.664* | — |

Cross-league beats the generic prior on both axes (mean abs error: attack .090
vs .120, defence .266 vs .340) but is **systematically optimistic about
defence** — promoted teams concede more than the projection expects, badly so
for Burnley. n=3, which is far too few to calibrate a correction without
inventing one, so the bias is documented rather than fitted out.

Known gaps, none of which this fixes:

- **No xG.** Understat covers six top-flight leagues and 404s on the
  Championship. E1 rows are goals-only, while the live model fits on xG — so
  cross-league ratings come from a different target than top-flight ones.
- **No lineups or player data.** football-data.co.uk carries results only.
  Formation derivation and player ratings for a promoted side still have
  nothing to work from, so the "not enough lineup data" fallback remains.
- **No FBref.** This project has never scraped FBref, and FBref now sits
  behind a Cloudflare bot challenge that returns 403 even for `robots.txt`.
