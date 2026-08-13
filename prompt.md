Build a Premier League match prediction tool — a Python backend + simple web frontend, 

run locally, for the current 2026-27 season. This is a personal project, not production 

infra — prioritize working end-to-end over scalability or polish.



\## Goal

Predict match outcome probabilities (home win / draw / away win) and a scoreline 

probability grid for upcoming Prem matchweek fixtures, combining squad strength, player 

fatigue, travel, and tactical style matchups. Realistic accuracy target: 50-55% on exact 

outcome, well-calibrated probabilities — do NOT chase higher accuracy or imply the model 

is more certain than it is.



\## Data pipeline (build first, validate before modeling)

1\. Pull historical EPL match data from football-data.co.uk (free CSVs, no auth needed) — 

&#x20;  last 4 seasons: full-time score, half-time score, shots, shots on target, corners, 

&#x20;  cards, referee.

2\. Scrape or find a free source for xG per match if reasonably accessible (understat.com); 

&#x20;  if too brittle to scrape reliably, fall back to goals-only and note xG was skipped.

3\. Pull player-level data for the last 1-2 seasons (appearances, minutes, goals, assists, 

&#x20;  position, per-90 proxy if available — FBref or similar). If no clean free source exists 

&#x20;  for a stat, skip it and note the gap rather than fabricating numbers.

4\. Store everything in a local SQLite database: matches, teams, team\_match\_stats, players, 

&#x20;  player\_match\_minutes, fixtures (with venue coordinates for travel calc).

5\. Write a script to pull the CURRENT gameweek's fixtures.



\## Player rating system

\- Rate each player from previous-season data: minutes, goals/assists per 90 for 

&#x20; attackers/midfielders, defensive proxy (tackles, interceptions, clean sheet 

&#x20; involvement) for defenders — weight by position, don't compare across positions 

&#x20; directly.

\- Bucket into a simple tier system (e.g. S/A/B/C or 1-5) per position group. Document 

&#x20; the formula clearly — this is a heuristic, not an official rating.

\- New signings / low-minute players get a flagged "provisional" fallback rating 

&#x20; (league/position average) rather than a fabricated confident number.



\## Player fatigue tracking

\- Per-player fatigue score (0-100), updating from: minutes played last match, minutes 

&#x20; in last 7/14 days, days of rest, midweek cup/European involvement.

\- Simple decay model: fatigue accumulates with minutes/match density, recovers with rest.

\- Discount a player's effective rating when heavily fatigued, feeding into the 

&#x20; lineup-weighted team rating below.



\## Travel \& fixture congestion

\- Away travel distance from venue coordinates (straight-line is fine).

\- Rest days since each team's last match, flagging congested fixture runs.

\- Apply a small, documented discount to the traveling/congested team's effective rating 

&#x20; — keep modest, this is a secondary effect.



\## Layer 1 — Squad strength model (core engine)

\- Dixon-Coles model: attack/defense rating per team, fit via maximum likelihood 

&#x20; (scipy.optimize), home advantage parameter, time-decay weighting for recent matches.

\- Weight team rating toward the ACTUAL lineup being used for that gameweek's prediction 

&#x20; (see two-stage workflow below) via the player rating + fatigue system, not a flat 

&#x20; season average.

\- Generate a full scoreline probability matrix via Poisson from the resulting expected 

&#x20; goals; derive W/D/L probabilities and most-likely scoreline.



\## Layer 2 — Style adjustment (build only after Layer 1 + fatigue/travel are validated)

\- Compute proxy tactical metrics per team (directness, shots-conceded pattern, 

&#x20; possession-adjacent stats if available).

\- Cluster teams into 3-4 rough style buckets (possession-heavy, direct/counter, 

&#x20; low-block/defensive, high-press).

\- Build a style-vs-style historical matchup adjustment, CONTROLLING for underlying 

&#x20; quality gap first (residualize against Layer 1's predictions) — don't let it just 

&#x20; rediscover "bad teams lose anyway."

\- Apply as a small adjustment to expected goals, not a replacement for Layer 1.



\## Per-gameweek lineup workflow — THREE modes



The system must support three ways a prediction's lineup gets set, selectable per 

fixture:



\*\*Mode 1 — Ideal/probable XI (default, automatic)\*\*

\- A few days before the gameweek, before official lineups are out: build each team's 

&#x20; most-used lineup from recent matches (by minutes played), adjusted for known 

&#x20; injuries/suspensions and current fatigue. Tag predictions from this mode "provisional."



\*\*Mode 2 — Confirmed XI (automatic rebuild near kickoff)\*\*

\- A script that fetches actual confirmed starting lineups close to kickoff (official 

&#x20; sources land \~1hr pre-match; use a reliable team-news source — e.g. Sky Sports team 

&#x20; news — for earlier "likely" confirmation if available; if no reliable free source 

&#x20; exists for this, say so rather than guessing).

\- Compare confirmed XI against the ideal XI used in Mode 1; flag meaningful differences 

&#x20; (e.g. "first-choice striker rested").

\- Regenerate the prediction and store as "final" — keep both provisional and final in 

&#x20; the database rather than overwriting, so they can be compared later.



\*\*Mode 3 — Manual lineup (user override)\*\*

\- A frontend input where I can manually select/edit the starting XI myself for either 

&#x20; team, for any fixture, at any time — e.g. if I want to test "what if City rest 

&#x20; Haaland" or I have my own read on who's starting before it's confirmed anywhere.

\- Predictions generated from a manual lineup are tagged "manual" and don't get 

&#x20; overwritten by Mode 1 or Mode 2 automatic rebuilds unless I explicitly reset the 

&#x20; fixture back to automatic mode.

\- This should reuse the same player rating/fatigue engine as the automatic modes — 

&#x20; just swapping which 11 players feed into it.



\*\*Frontend requirements for this\*\*

\- Each fixture clearly labeled with its current mode: Provisional / Final / Manual.

\- A simple lineup editor (dropdown or searchable list of the squad) for Mode 3.

\- When a confirmed lineup differs from the ideal XI in a way that meaningfully changed 

&#x20; the prediction, show a short note with the before/after expected goals shift.

\- Track and display, over time, how much provisional vs. final predictions actually 

&#x20; differ on average — tells you if the two-stage process earns its complexity.



\## Market benchmark

\- Pull closing odds per fixture from a free odds API if available (e.g. the-odds-api.com 

&#x20; free tier). Convert to implied probabilities and display alongside the model's own 

&#x20; prediction — not to copy the bookmaker, but as a built-in sanity check for when the 

&#x20; model's badly off.



\## Validation (required, do not skip)

\- Backtest Layer 1 alone (season-average ratings, no lineup/fatigue weighting) on the 

&#x20; most recent complete season — time-series split only, never shuffled.

\- Report outcome accuracy %, Brier score/log-loss, and a calibration plot.

\- Backtest again with lineup-weighted rating + fatigue/travel added; report whether this 

&#x20; improved calibration. Note: this can only be backtested where historical lineup data 

&#x20; is available — if incomplete, say so rather than faking it.

\- Backtest again with Layer 2 (style) added; same explicit report on whether it helped.

\- If any layer doesn't improve Brier score/log-loss, keep it in the code but clearly 

&#x20; flag the finding — I want to know what's actually earning its keep, not just ship 

&#x20; everything and assume it's helping.



\## Failure handling

\- If fixture, lineup, or odds data is missing/stale for a gameweek, show that clearly 

&#x20; on the frontend ("prediction based on partial data") rather than silently filling gaps 

&#x20; with guesses.

\- Log data pull failures to a file so week-to-week issues are traceable.

\- If any data source turns out unreliable/rate-limited/broken during the build, stop 

&#x20; and tell me rather than silently substituting fake/synthetic data.



\## Testing

\- Basic sanity-check unit tests: scoreline grid probabilities sum to \~1, a 

&#x20; significantly higher-rated team gets a higher win probability in a synthetic test 

&#x20; fixture, isolating home advantage actually increases home win probability.



\## Web frontend

\- Simple local web app (Flask or FastAPI + basic HTML/JS, no heavy framework needed).

\- Gameweek view: fixtures with W/D/L probability bars, top scoreline probabilities, 

&#x20; expected goals per team, lineup mode label (Provisional/Final/Manual), starting XI 

&#x20; with player tier + fatigue level, travel/congestion flags, market odds comparison.

\- Manual lineup editor per fixture (Mode 3).

\- Model-performance page: backtest accuracy/calibration results, layer by layer, and 

&#x20; the model's own historical accuracy shown prominently near predictions (e.g. "this 

&#x20; type of prediction has been right \~52% of the time historically") so it never implies 

&#x20; more confidence than it's earned.



\## Weekly refresh process

\- One command that: pulls results from the past week, updates team ratings and player 

&#x20; fatigue, pulls next gameweek's fixtures, and runs Mode 1 (ideal XI) predictions.

\- A separate command for the Mode 2 rebuild (confirmed lineups) to be run manually by 

&#x20; me closer to kickoff — do not set up automated OS-level scheduling/cron, I'll trigger 

&#x20; these myself.



\## Constraints

\- Use SQLite, not a heavier database.

\- Comment the Dixon-Coles math and rating/fatigue formulas clearly — I want to 

&#x20; understand this, not just run it.

\- Structure as a proper repo: /data, /models, /backend, /frontend, README explaining 

&#x20; how to run it, refresh it weekly, and use the manual lineup override.



Work through this in order: data pipeline → player ratings + fatigue → Layer 1 (ideal-XI 

mode only, season-average baseline) → validation checkpoint → travel/congestion → Layer 2 

→ validation → Mode 2 (confirmed lineup rebuild) → Mode 3 (manual override) → market 

odds → frontend → testing. Check in with me after the first Layer 1 validation 

(season-average vs lineup-weighted) before building further, since if lineup-weighting 

doesn't actually improve calibration, that changes how much effort the rest deserves.

