"""Download historical EPL match CSVs into data/raw/.

Primary source football-data.co.uk is blocked on this network (FortiGuard
"Gambling" category), so we pull the same CSVs from the maintained GitHub
mirror `datasets/football-datasets`, which scrapes football-data.co.uk daily
and keeps its exact schema (FTHG, FTAG, FTR, shots, corners, cards, referee).

Seasons: the 4 complete seasons 2022-23 .. 2025-26, plus the current 2026-27
file when it appears (404 before the season's first results are posted —
reported, not fatal).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from net import session

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
COMPLETE_SEASONS = ["2223", "2324", "2425", "2526"]
CURRENT_SEASON = "2627"
MIRROR = ("https://raw.githubusercontent.com/datasets/football-datasets/"
          "main/datasets/premier-league/season-{s}.csv")

HTTP = session()


def fetch(url: str, out: Path, required: bool) -> bool:
    try:
        r = HTTP.get(url, timeout=30)
        r.raise_for_status()
        out.write_bytes(r.content)
        print(f"OK   {out.name}: {r.text.count(chr(10))} lines")
        return True
    except Exception as e:
        print(f"{'FAIL' if required else 'SKIP'} {out.name}: {e}")
        return not required


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    ok = True
    for s in COMPLETE_SEASONS:
        ok &= fetch(MIRROR.format(s=s), RAW / f"E0_{s}.csv", required=True)
    fetch(MIRROR.format(s=CURRENT_SEASON), RAW / f"E0_{CURRENT_SEASON}.csv",
          required=False)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
