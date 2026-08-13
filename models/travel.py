"""Travel + fixture-congestion adjustments. Deliberately small effects.

Travel: straight-line (haversine) distance from the away team's stadium to
the venue. Discount on the AWAY side's expected goals:
  0.5% per 200 km, capped at 3%  ->  mult = 1 - min(0.03, dist_km/200 * 0.005)
London-to-London derbies ≈ 0; Bournemouth to Newcastle (~500 km) ≈ 1.2%.

Congestion: rest days since each team's last league match. If a team has
< 4 full rest days AND at least 2 fewer than its opponent, its expected
goals take a 2% discount. (Cup/European midweek games are not in the DB —
same known gap as fatigue.py — so congestion can be understated.)

If stadium coordinates are missing (newly promoted team not in the seed
list), no adjustment is applied and `partial` is returned True so the
prediction is labelled "partial data" instead of silently guessing.
"""
import math

TRAVEL_RATE = 0.005 / 200.0   # per km
TRAVEL_CAP = 0.03
SHORT_REST_DAYS = 4
REST_GAP = 2
CONGESTION_DISCOUNT = 0.02


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def travel_multiplier(away_coords, venue_coords):
    """Returns (multiplier_on_away_xg, distance_km_or_None, partial)."""
    if None in (away_coords + venue_coords):
        return 1.0, None, True
    d = haversine_km(*away_coords, *venue_coords)
    return 1.0 - min(TRAVEL_CAP, d * TRAVEL_RATE), d, False


def rest_days(conn, team_id: int, as_of_date: str):
    row = conn.execute(
        """SELECT julianday(?) - julianday(MAX(m.date)) AS d FROM matches m
           WHERE (m.home_team_id = ? OR m.away_team_id = ?) AND m.date < ?""",
        (as_of_date, team_id, team_id, as_of_date)).fetchone()
    return row["d"]


def congestion_multipliers(home_rest, away_rest):
    """(home_mult, away_mult, flags) from rest-day comparison."""
    hm = am = 1.0
    flags = []
    if home_rest is not None and away_rest is not None:
        if home_rest < SHORT_REST_DAYS and away_rest - home_rest >= REST_GAP:
            hm = 1.0 - CONGESTION_DISCOUNT
            flags.append(f"home short rest ({home_rest:.0f}d vs {away_rest:.0f}d)")
        if away_rest < SHORT_REST_DAYS and home_rest - away_rest >= REST_GAP:
            am = 1.0 - CONGESTION_DISCOUNT
            flags.append(f"away short rest ({away_rest:.0f}d vs {home_rest:.0f}d)")
    return hm, am, flags
