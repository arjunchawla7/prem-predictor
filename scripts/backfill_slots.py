"""Backfill player_match_minutes.slot_pos AND repair .started from the
cached understat match JSON (data/raw/understat/match_*.json).

slot_pos is the per-appearance position code (GK, DC, DL, DMC, AMC, FW, ...)
that formation shapes are derived from.

The started repair matters: the original loader treated roster_in == "0" as
"started", but roster_in is not a minute and reads "0" for substitutes too,
so the stored starting XIs were partly wrong. The correct rule is that bench
players carry position 'Sub' and everyone else started.

Reads only local cache files, no network. Safe to re-run.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.db import connect

CACHE = ROOT / "data" / "raw" / "understat"


def main():
    conn = connect()
    by_us = {r["understat_id"]: r["id"] for r in conn.execute(
        "SELECT id, understat_id FROM matches WHERE understat_id IS NOT NULL")}
    players = {r["understat_id"]: r["id"] for r in conn.execute(
        "SELECT id, understat_id FROM players WHERE understat_id IS NOT NULL")}

    updated, files = 0, 0
    for f in CACHE.glob("match_*.json"):
        us_mid = int(f.stem.split("_")[1])
        mid = by_us.get(us_mid)
        if mid is None:
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        files += 1
        for side in ("h", "a"):
            for entry in data["rosters"][side].values():
                pid = players.get(int(entry["player_id"]))
                if pid is None:
                    continue
                pos_code = entry.get("position")
                cur = conn.execute(
                    "UPDATE player_match_minutes SET slot_pos=?, started=? "
                    "WHERE match_id=? AND player_id=?",
                    (pos_code, 0 if pos_code == "Sub" else 1, mid, pid))
                updated += cur.rowcount
        if files % 200 == 0:
            conn.commit()
            print(f"  {files} matches…")
    conn.commit()
    print(f"slot_pos backfilled: {updated} appearances from {files} matches")


if __name__ == "__main__":
    main()
