"""SQLite schema + connection helper.

One local file: data/prem.db. Tables:
  teams                 canonical team list + stadium coordinates (travel calc)
  matches               one row per played match (result, xG when available)
  team_match_stats      per-team per-match shots/corners/cards etc.
  players               player identity + position group
  player_match_minutes  per-player per-match minutes/goals/assists
  fixtures              upcoming (and past) scheduled fixtures for the current
                        season, with lineup mode state per fixture
  predictions           every generated prediction, tagged provisional /
                        final / manual — never overwritten, so provisional vs
                        final can be compared later
  pull_log              data-pull failures, for week-to-week traceability
"""
import sqlite3
from pathlib import Path

import os

# PREM_DATA_DIR / PREM_DB_PATH let a deployment point at a persistent-disk
# mount (e.g. Render) instead of the repo-relative default used locally.
DATA_DIR = Path(os.environ.get(
    "PREM_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data")))
DB_PATH = Path(os.environ.get("PREM_DB_PATH", str(DATA_DIR / "prem.db")))

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,          -- football-data.co.uk spelling
    understat_name TEXT,                -- understat spelling
    lat REAL, lon REAL                  -- stadium coordinates
);

CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY,
    season TEXT NOT NULL,               -- '2223' .. '2627'
    date TEXT NOT NULL,                 -- ISO yyyy-mm-dd
    home_team_id INTEGER NOT NULL REFERENCES teams(id),
    away_team_id INTEGER NOT NULL REFERENCES teams(id),
    fthg INTEGER NOT NULL, ftag INTEGER NOT NULL,
    ftr TEXT NOT NULL,                  -- H/D/A
    hthg INTEGER, htag INTEGER, htr TEXT,
    referee TEXT,
    understat_id INTEGER,               -- NULL until xG matched
    home_xg REAL, away_xg REAL,         -- NULL when xG unavailable
    UNIQUE(season, date, home_team_id, away_team_id)
);

CREATE TABLE IF NOT EXISTS team_match_stats (
    match_id INTEGER NOT NULL REFERENCES matches(id),
    team_id INTEGER NOT NULL REFERENCES teams(id),
    is_home INTEGER NOT NULL,
    shots INTEGER, shots_on_target INTEGER, corners INTEGER,
    fouls INTEGER, yellows INTEGER, reds INTEGER,
    PRIMARY KEY (match_id, team_id)
);

CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY,
    understat_id INTEGER UNIQUE,
    name TEXT NOT NULL,
    team_id INTEGER REFERENCES teams(id),   -- current/most recent team
    position TEXT                            -- GK / DEF / MID / FWD
);

CREATE TABLE IF NOT EXISTS player_match_minutes (
    match_id INTEGER NOT NULL REFERENCES matches(id),
    player_id INTEGER NOT NULL REFERENCES players(id),
    team_id INTEGER NOT NULL REFERENCES teams(id),
    minutes INTEGER NOT NULL,
    goals INTEGER DEFAULT 0, assists INTEGER DEFAULT 0,
    started INTEGER DEFAULT 0,
    PRIMARY KEY (match_id, player_id)
);

CREATE TABLE IF NOT EXISTS player_season_stats (
    player_id INTEGER NOT NULL REFERENCES players(id),
    season TEXT NOT NULL,               -- '2425', '2526', ...
    team_id INTEGER REFERENCES teams(id),
    position TEXT,                      -- GK / DEF / MID / FWD (dominant)
    minutes INTEGER, games INTEGER,
    goals INTEGER, assists INTEGER,
    xg REAL, xa REAL, npxg REAL,
    shots INTEGER, key_passes INTEGER,
    xg_buildup REAL, xg_chain REAL,
    PRIMARY KEY (player_id, season)
);

CREATE TABLE IF NOT EXISTS fixtures (
    id INTEGER PRIMARY KEY,
    season TEXT NOT NULL,
    gameweek INTEGER,
    date TEXT,                          -- ISO datetime (kickoff, UK time)
    home_team_id INTEGER NOT NULL REFERENCES teams(id),
    away_team_id INTEGER NOT NULL REFERENCES teams(id),
    understat_id INTEGER UNIQUE,
    pulse_id INTEGER UNIQUE,            -- premierleague.com fixture id
    status TEXT DEFAULT 'scheduled',    -- scheduled / played
    lineup_mode TEXT DEFAULT 'auto',    -- auto / manual: manual blocks
                                        -- automatic prediction rebuilds
    UNIQUE(season, home_team_id, away_team_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY,
    fixture_id INTEGER NOT NULL REFERENCES fixtures(id),
    created_at TEXT NOT NULL,
    tag TEXT NOT NULL,                  -- provisional / final / manual
    home_xg REAL NOT NULL, away_xg REAL NOT NULL,
    p_home REAL NOT NULL, p_draw REAL NOT NULL, p_away REAL NOT NULL,
    score_grid TEXT NOT NULL,           -- JSON [[p(0-0),p(0-1)..]..] home rows
    home_lineup TEXT, away_lineup TEXT, -- JSON list of player ids (or NULL =
                                        -- season-average, no lineup weighting)
    notes TEXT,                         -- e.g. lineup-diff explanation
    partial_data INTEGER DEFAULT 0      -- 1 = prediction from incomplete data
);

CREATE TABLE IF NOT EXISTS managers (
    id INTEGER PRIMARY KEY,
    team_id INTEGER NOT NULL REFERENCES teams(id),
    pulse_id INTEGER UNIQUE,
    opta_code TEXT,                     -- 'man51018', keys the photo CDN
    name TEXT NOT NULL,
    role TEXT,                          -- Manager / Head Coach
    nationality TEXT,
    dob TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS transfers (
    id INTEGER PRIMARY KEY,
    player_id INTEGER NOT NULL REFERENCES players(id),
    detected_at TEXT NOT NULL,
    from_team_id INTEGER REFERENCES teams(id),   -- NULL = new to the league
    to_team_id INTEGER REFERENCES teams(id)      -- NULL = left the league
);

CREATE TABLE IF NOT EXISTS market_odds (
    fixture_id INTEGER NOT NULL REFERENCES fixtures(id),
    ts TEXT NOT NULL,                   -- when pulled
    bookmaker TEXT,
    odds_home REAL, odds_draw REAL, odds_away REAL,
    PRIMARY KEY (fixture_id, ts)
);

CREATE TABLE IF NOT EXISTS pull_log (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    source TEXT NOT NULL,
    ok INTEGER NOT NULL,
    detail TEXT
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    # additive migrations for DBs created before a column existed
    def add_col(table, col, decl):
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
    add_col("fixtures", "pulse_id", "INTEGER")
    add_col("teams", "crest_url", "TEXT")
    add_col("teams", "pulse_id", "INTEGER")
    add_col("players", "pulse_id", "INTEGER")
    add_col("players", "shirt_num", "INTEGER")
    add_col("players", "in_current_squad", "INTEGER DEFAULT 0")
    add_col("players", "detail_pos", "TEXT")   # e.g. ST / RW / CDM / CB / LB
    add_col("players", "opta_code", "TEXT")    # 'p154561', keys the photo CDN
    # understat slot code for that appearance (GK/DC/DL/DMC/AMC/FW/...),
    # which is what formation shapes are derived from
    add_col("player_match_minutes", "slot_pos", "TEXT")
    # market closing odds per played match (scripts/pull_odds_history.py);
    # market_odds holds pre-match quotes for UPCOMING fixtures instead.
    for c in ("odds_home", "odds_draw", "odds_away"):
        add_col("matches", c, "REAL")
    # what actually fed each stored prediction (JSON list), so an old row can
    # still be read back correctly after the model configuration changes
    add_col("predictions", "factors", "TEXT")
    # model+market blend, kept in its own columns so it can never be mistaken
    # for the model's own probabilities
    for c in ("blend_home", "blend_draw", "blend_away"):
        add_col("predictions", c, "REAL")
    return conn


def log_pull(conn: sqlite3.Connection, source: str, ok: bool, detail: str = ""):
    from datetime import datetime, timezone
    conn.execute("INSERT INTO pull_log (ts, source, ok, detail) VALUES (?,?,?,?)",
                 (datetime.now(timezone.utc).isoformat(), source, int(ok), detail))
    conn.commit()
