"""Seed the teams table: canonical names, understat name mapping, stadium
coordinates (approximate, for straight-line travel distance only).

Teams promoted for a future season that aren't listed here get inserted
dynamically by the loaders with NULL coordinates; predictions involving them
flag travel data as partial rather than guessing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.db import connect

# name (football-data.co.uk spelling), understat spelling, stadium lat, lon
TEAMS = [
    ("Arsenal",          "Arsenal",                 51.5549, -0.1084),
    ("Aston Villa",      "Aston Villa",             52.5092, -1.8847),
    ("Bournemouth",      "Bournemouth",             50.7352, -1.8382),
    ("Brentford",        "Brentford",               51.4907, -0.2889),
    ("Brighton",         "Brighton",                50.8616, -0.0837),
    ("Burnley",          "Burnley",                 53.7890, -2.2302),
    ("Chelsea",          "Chelsea",                 51.4817, -0.1910),
    ("Crystal Palace",   "Crystal Palace",          51.3983, -0.0855),
    ("Everton",          "Everton",                 53.4280, -2.9992),  # Hill Dickinson Stadium (2025-)
    ("Fulham",           "Fulham",                  51.4749, -0.2217),
    ("Ipswich",          "Ipswich",                 52.0550,  1.1447),
    ("Leeds",            "Leeds",                   53.7778, -1.5721),
    ("Leicester",        "Leicester",               52.6204, -1.1422),
    ("Liverpool",        "Liverpool",               53.4308, -2.9608),
    ("Luton",            "Luton",                   51.8842, -0.4316),
    ("Man City",         "Manchester City",         53.4831, -2.2004),
    ("Man United",       "Manchester United",       53.4631, -2.2913),
    ("Newcastle",        "Newcastle United",        54.9756, -1.6217),
    ("Nott'm Forest",    "Nottingham Forest",       52.9399, -1.1329),
    ("Sheffield United", "Sheffield United",        53.3703, -1.4708),
    ("Southampton",      "Southampton",             50.9058, -1.3911),
    ("Sunderland",       "Sunderland",              54.9146, -1.3882),
    ("Tottenham",        "Tottenham",               51.6043, -0.0664),
    ("West Ham",         "West Ham",                51.5386,  0.0166),
    ("Wolves",           "Wolverhampton Wanderers", 52.5903, -2.1302),
]


def main():
    conn = connect()
    for name, us_name, lat, lon in TEAMS:
        conn.execute(
            """INSERT INTO teams (name, understat_name, lat, lon)
               VALUES (?,?,?,?)
               ON CONFLICT(name) DO UPDATE SET
                 understat_name=excluded.understat_name,
                 lat=excluded.lat, lon=excluded.lon""",
            (name, us_name, lat, lon))
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
    print(f"teams: {n}")


if __name__ == "__main__":
    main()
