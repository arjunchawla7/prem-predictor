"""Merge duplicate player rows created by the pre-fix name matcher
(squad members whose accented names — Ø, ł, ß … — failed to match their
understat row and were inserted as brand-new players).

A dup pair is: one row with pulse_id + no understat_id (from squad sync),
one row with understat_id + no pulse_id, same normalised name. The
understat row absorbs the pulse fields; transfers rows are repointed;
the dup is deleted. Safe to re-run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend.db import connect
from pull_squads import norm


def main():
    conn = connect()
    dups = conn.execute(
        "SELECT * FROM players WHERE pulse_id IS NOT NULL "
        "AND understat_id IS NULL").fetchall()
    keepers = conn.execute(
        "SELECT * FROM players WHERE understat_id IS NOT NULL").fetchall()
    by_norm = {}
    for k in keepers:
        by_norm.setdefault(norm(k["name"]), []).append(k)

    merged = 0
    for d in dups:
        cand = by_norm.get(norm(d["name"]), [])
        if len(cand) != 1:
            # mononym fallback: understat 'Gabriel' == PL 'Gabriel Magalhães'
            toks = set(norm(d["name"]).split())
            cand = [k for k in keepers
                    if set(norm(k["name"]).split()) < toks]
            if len(cand) != 1:
                continue
        k = cand[0]
        conn.execute(
            """UPDATE players SET pulse_id=?, shirt_num=?, team_id=?,
                 in_current_squad=?, position=COALESCE(position, ?)
               WHERE id=?""",
            (d["pulse_id"], d["shirt_num"], d["team_id"],
             d["in_current_squad"], d["position"], k["id"]))
        conn.execute("UPDATE transfers SET player_id=? WHERE player_id=?",
                     (k["id"], d["id"]))
        # false "new to league" transfer for the dup becomes noise — drop it
        conn.execute(
            """DELETE FROM transfers WHERE player_id=? AND from_team_id IS NULL
               AND to_team_id=?""", (k["id"], d["team_id"]))
        conn.execute("DELETE FROM players WHERE id=?", (d["id"],))
        merged += 1
    # also drop stale "departed" rows for players who are in a current squad
    n = conn.execute(
        """DELETE FROM transfers WHERE to_team_id IS NULL AND player_id IN
             (SELECT id FROM players WHERE in_current_squad=1)""").rowcount
    conn.commit()
    print(f"merged {merged} duplicate players; removed {n} stale departure rows")


if __name__ == "__main__":
    main()
