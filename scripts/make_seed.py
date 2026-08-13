"""Build the deploy seed bundle: a copy of the current local database and
backtest CSVs, tracked in git (unlike data/prem.db itself) so a fresh
deployment starts with real data instead of an empty schema.

Run this locally whenever you want the deployed site to pick up newer data
on its next deploy — it does not touch the live disk on Render, only the
repo files that seed_disk.py copies from on first boot.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.db import DATA_DIR

SEED = ROOT / "data" / "seed"


def main():
    SEED.mkdir(parents=True, exist_ok=True)
    db = DATA_DIR / "prem.db"
    if not db.exists():
        print("no local data/prem.db found — run the data pipeline first")
        sys.exit(1)
    shutil.copy2(db, SEED / "prem.db")
    print(f"copied {db} -> {SEED / 'prem.db'} "
          f"({db.stat().st_size / 1e6:.1f} MB)")

    bt_src = DATA_DIR / "backtests"
    bt_dst = SEED / "backtests"
    if bt_src.exists():
        bt_dst.mkdir(exist_ok=True)
        n = 0
        for f in bt_src.glob("*.csv"):
            shutil.copy2(f, bt_dst / f.name)
            n += 1
        print(f"copied {n} backtest CSVs")


if __name__ == "__main__":
    main()
