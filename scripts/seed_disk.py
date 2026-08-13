"""Deploy-time bootstrap: if the persistent disk has no database yet, copy
the seed bundle (data/seed/) onto it. Never overwrites an existing DB, so
redeploys don't clobber data collected since the last seed refresh.

Runs as Render's preDeployCommand (see render.yaml) — harmless to run
locally too, where DATA_DIR is the same as the seed source and this is a
no-op.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.db import DATA_DIR, DB_PATH

SEED = ROOT / "data" / "seed"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seed_db = SEED / "prem.db"
    if DB_PATH.exists():
        print(f"{DB_PATH} already present — leaving it alone")
    elif seed_db.exists():
        shutil.copy2(seed_db, DB_PATH)
        print(f"seeded {DB_PATH} from {seed_db}")
    else:
        print("no seed database found — starting with an empty schema")

    bt_dst = DATA_DIR / "backtests"
    bt_src = SEED / "backtests"
    if bt_src.exists():
        bt_dst.mkdir(parents=True, exist_ok=True)
        for f in bt_src.glob("*.csv"):
            dst = bt_dst / f.name
            if not dst.exists():
                shutil.copy2(f, dst)


if __name__ == "__main__":
    main()
