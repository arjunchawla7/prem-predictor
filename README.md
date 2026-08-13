# Prem Predictor

Local Premier League match-prediction tool for the 2026-27 season.
Python backend + SQLite + small Flask web frontend. Personal project —
built for working end-to-end, not scale.

**Honest expectations:** the model backtests at ~47% exact-outcome accuracy
(uniform guessing = 33%, bookmakers ≈ 53-55%) with decent calibration. It
does not beat the market and is not supposed to. The performance page keeps
this in your face on purpose.

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
  Poisson; W/D/L from the grid. Math commented in the file.
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
  just rediscover "good beats bad"); capped ±0.15 xG.
- **Team profile** (`models/formation.py`): preferred formation *derived*
  from the position codes of each side's real starting XIs over its last 30
  matches (4-2-3-1, 3-5-2, …), reported with its share and the runner-up
  shapes; plus tactical trait labels from league percentile ranks on shot
  volume, xG per shot, shots conceded and corners. No possession or pressing
  data exists in this database, so no claim is made about either.

### Validation findings (2025-26 walk-forward backtest)

| variant | accuracy | Brier | log-loss |
|---|---|---|---|
| Layer 1 season-average | 46.6% | 0.6188 | 1.0569 |
| + lineup/fatigue/travel | 46.6% | 0.6186 | 1.0566 |
| + Layer 2 style | 46.8% | 0.6184 | 1.0563 |

**Neither addition meaningfully improved calibration.** Both are kept in the
code (and useful for what-if lineup analysis), but they are not currently
earning their keep — re-check on 2026-27 data as it accumulates.

## Data sources & known gaps

- Results/shots/cards: football-data.co.uk **via its GitHub mirror**
  (`datasets/football-datasets`) — the origin site is blocked as "Gambling"
  by the FortiGate firewall on this network. Same CSVs, same schema.
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
models/    dixon_coles, player_ratings, fatigue, lineup, travel, style, backtest
scripts/   data pulls, refresh_week, rebuild_confirmed, backtests, sweeps
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
   copies `data/seed/prem.db` onto the persistent disk). After that you get
   a public URL.

**Persistence caveat:** the free plan may not include a persistent disk —
check when the service is created. If it doesn't, the database resets to
the seed bundle on every deploy/restart (fine for browsing, not for
accumulating new predictions). A paid "Starter" instance keeps the disk.

**Updating deployed data:** there's no scheduled job on the free tier. To
push newer data live: run `refresh_week.py` locally as usual, then
`scripts\make_seed.py` to refresh `data/seed/`, commit, and push — the next
deploy picks it up. (Or run `rebuild_confirmed.py` similarly before a
gameweek.) This is a personal-project workaround, not a real pipeline.

### Pushing to GitHub for the first time

```powershell
gh repo create prem-predictor --private --source=. --push
# or manually:
git remote add origin https://github.com/<you>/prem-predictor.git
git push -u origin master
```
