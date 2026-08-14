"""Run candidate situational 'records' through models.situational.

Default run is the Man Utd half-time-lead-at-Old-Trafford claim, reported the
only way it means anything: alongside the same number for every other club.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.db import connect
from models.situational import (MIN_RELIABLE, model_control, run, scan,
                                team_match_frame, SITUATIONS)

pd.set_option("display.width", 200)


def show(f, team, situation, outcome=None):
    r = run(f, team, situation, outcome)
    print(f"\n=== {team} — {situation} ({r.outcome}) ===")
    print(f"  window          {r.window}")
    print(f"  {team+' rate':<15} {r.n:>4} occurrences, "
          f"{r.rate:.1%} {r.outcome}")
    print(f"  {'league rate':<15} {r.base_n:>4} occurrences, "
          f"{r.base_rate:.1%} {r.outcome}   (all other clubs)")
    print(f"  lift            {r.lift:+.1%}   p={r.p_value:.3f}")
    print(f"  pre-match usable: {r.pre_match}   "
          f"small sample (<{MIN_RELIABLE}): {r.small_sample}")
    print(f"  VERDICT: {r.verdict}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="Man United")
    ap.add_argument("--situation", default="home_leading_at_ht",
                    choices=sorted(SITUATIONS))
    ap.add_argument("--outcome", default=None,
                    choices=[None, "win", "not_lose", "draw"])
    ap.add_argument("--all-situations", action="store_true")
    args = ap.parse_args()

    conn = connect()
    f = team_match_frame(conn)
    print(f"dataset: {f.match_id.nunique()} matches, "
          f"{f.date.min():%Y-%m-%d} to {f.date.max():%Y-%m-%d}, "
          f"{f.team.nunique()} clubs")

    sits = sorted(SITUATIONS) if args.all_situations else [args.situation]
    for s in sits:
        r = show(f, args.team, s, args.outcome)
        outcome = args.outcome or SITUATIONS[s][2]
        mask_fn, pre_match, _ = SITUATIONS[s]
        ctrl = model_control(conn, f, mask_fn(f), args.team, outcome,
                             pre_match=pre_match)
        print(f"  model-expectation control: {ctrl}")
        board = scan(f, s, outcome)
        if len(board):
            print(f"  --- every club, same situation "
                  f"(this is what makes it a base rate or a signal) ---")
            print(board[["team", "n", "rate", "base_rate", "lift",
                         "p_value"]].to_string(index=False))


if __name__ == "__main__":
    main()
