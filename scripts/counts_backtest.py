"""Do the corners and cards models beat guessing the average? (Tier 2)

Same walk-forward discipline as everything else: step through 2025-26 in date
order, refit weekly on matches played before that date only, never shuffled.

Count models need different reference points than a three-way result model.
Two baselines are reported for every measure, because "the model is close to
right on average" is nearly free for a quantity whose league mean barely
moves:

  league mean    predict the training-window average for every match
  team form      predict the two sides' own season-to-date averages, added

A model that cannot beat both of these has learned nothing worth shipping.

Scored on:
  MAE / RMSE     on the match total
  Brier/log-loss on a threshold event, against its base rate
  calibration    predicted vs actual by bucket

Cards are run twice — with and without the referee factor — since the referee
is only known close to kickoff, so the gain has to be worth the dependency.

  .venv\\Scripts\\python scripts\\counts_backtest.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.db import connect
from models.config import TRAIN_SEASONS
from models.counts import CountModel, prob_at_least

TARGET = "2526"
CORNER_LINE = 10        # P(total corners >= 10)
CARD_LINE = 4           # P(total yellow cards >= 4)


def load_counts(conn, kind):
    """One row per match with the home/away count for `kind`, plus referee."""
    col = {"corners": "corners", "cards": "yellows"}[kind]
    return pd.read_sql_query(
        f"""SELECT m.id, m.season, m.date, m.referee,
                   h.name AS home, a.name AS away,
                   hs.{col} AS home_count, aws.{col} AS away_count
            FROM matches m
            JOIN teams h ON h.id = m.home_team_id
            JOIN teams a ON a.id = m.away_team_id
            JOIN team_match_stats hs ON hs.match_id = m.id AND hs.is_home = 1
            JOIN team_match_stats aws ON aws.match_id = m.id AND aws.is_home = 0
            WHERE COALESCE(m.division, 'E0') = 'E0'
              AND hs.{col} IS NOT NULL AND aws.{col} IS NOT NULL
            ORDER BY m.date""", conn)


def walk(df, target_season, use_referee, refit_every=7, min_train=380):
    target = df[df["season"] == target_season].sort_values("date")
    rows, model, last_fit = [], None, None
    # season-to-date team averages, for the "team form" baseline
    seen_for = defaultdict(list)
    seen_against = defaultdict(list)

    for _, r in target.iterrows():
        date = pd.Timestamp(r["date"])
        if model is None or (date - last_fit).days >= refit_every:
            train = df[pd.to_datetime(df["date"]) < date]
            if len(train) < min_train:
                continue
            model = CountModel(use_referee=use_referee).fit(
                train.to_dict("records"), as_of=date)
            last_fit = date
            train_mean = float(train["home_count"].mean()
                               + train["away_count"].mean())

        lam, mu = model.expected_counts(r["home"], r["away"],
                                        r.get("referee"))
        actual = float(r["home_count"] + r["away_count"])

        # team-form baseline: each side's own average so far this season,
        # for-and-against averaged; falls back to the league mean early on
        def side_avg(team):
            f, a = seen_for[team], seen_against[team]
            if not f:
                return train_mean / 2
            return (np.mean(f) + np.mean(a)) / 2
        form = side_avg(r["home"]) + side_avg(r["away"])

        rows.append({
            "date": r["date"], "home": r["home"], "away": r["away"],
            "referee": r.get("referee"),
            "pred_home": lam, "pred_away": mu, "pred_total": lam + mu,
            "league_mean": train_mean, "form": form,
            "actual": actual,
            "ref_factor": model.referee_factor(r.get("referee")) or 1.0,
        })
        seen_for[r["home"]].append(r["home_count"])
        seen_against[r["home"]].append(r["away_count"])
        seen_for[r["away"]].append(r["away_count"])
        seen_against[r["away"]].append(r["home_count"])
    return pd.DataFrame(rows)


def threshold_scores(bt, line, lam_col="pred_home", mu_col="pred_away"):
    p = np.array([prob_at_least(a, b, line)
                  for a, b in zip(bt[lam_col], bt[mu_col])])
    y = (bt["actual"] >= line).astype(int).to_numpy()
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    base = float(y.mean())
    # the same event scored by a model that always predicts the base rate
    bb = np.full(len(y), base)
    return {
        "line": line, "base_rate": base, "mean_pred": float(p.mean()),
        "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
        "majority": float(max(base, 1 - base)),
        "brier": float(np.mean((pc - y) ** 2)),
        "brier_base": float(np.mean((bb - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "logloss_base": float(-np.mean(y * np.log(base)
                                       + (1 - y) * np.log(1 - base))),
        "p": p, "y": y,
    }


def calibration(p, y, bins=5):
    edges = np.linspace(p.min(), p.max() + 1e-9, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() >= 5:
            out.append((f"{edges[i]:.2f}-{edges[i+1]:.2f}", int(m.sum()),
                        float(p[m].mean()), float(y[m].mean())))
    return out


def err(pred, actual):
    d = np.asarray(pred, float) - np.asarray(actual, float)
    return float(np.mean(np.abs(d))), float(np.sqrt(np.mean(d ** 2)))


def report(name, bt, line):
    print(f"\n{'=' * 62}\n{name}  (n={len(bt)}, walk-forward on {TARGET})")
    mae, rmse = err(bt.pred_total, bt.actual)
    lmae, lrmse = err(bt.league_mean, bt.actual)
    fmae, frmse = err(bt.form, bt.actual)
    print(f"  actual mean total   {bt.actual.mean():.2f}"
          f"   model mean {bt.pred_total.mean():.2f}")
    print(f"  MAE   model {mae:.3f}   league mean {lmae:.3f} "
          f"({mae - lmae:+.3f})   team form {fmae:.3f} ({mae - fmae:+.3f})")
    print(f"  RMSE  model {rmse:.3f}   league mean {lrmse:.3f} "
          f"({rmse - lrmse:+.3f})   team form {frmse:.3f} ({rmse - frmse:+.3f})")

    t = threshold_scores(bt, line)
    print(f"\n  threshold event: total >= {line}"
          f"   (happened {t['base_rate']:.1%} of matches)")
    print(f"    mean prediction   {t['mean_pred']:.1%}"
          f"   (gap {t['mean_pred'] - t['base_rate']:+.1%})")
    print(f"    accuracy @0.5     {t['accuracy']:.1%}"
          f"   vs always-majority {t['majority']:.1%}"
          f"   ({t['accuracy'] - t['majority']:+.1%})")
    print(f"    Brier             {t['brier']:.4f}"
          f"   vs base rate {t['brier_base']:.4f}"
          f"   ({t['brier'] - t['brier_base']:+.4f})")
    print(f"    log-loss          {t['logloss']:.4f}"
          f"   vs base rate {t['logloss_base']:.4f}"
          f"   ({t['logloss'] - t['logloss_base']:+.4f})")
    cal = calibration(t["p"], t["y"])
    if cal:
        print("    calibration:")
        for b, n, pp, aa in cal:
            flag = "" if abs(pp - aa) < 0.08 else "   <-- off"
            print(f"      {b}  n={n:>3}  predicted {pp:.1%}  "
                  f"actual {aa:.1%}{flag}")
    return {"mae": mae, "rmse": rmse, "lmae": lmae, "fmae": fmae, **{
        k: v for k, v in t.items() if k not in ("p", "y")}}


def main():
    conn = connect()
    results = {}

    corners = load_counts(conn, "corners")
    corners = corners[corners["season"].isin(TRAIN_SEASONS)]
    bt_c = walk(corners, TARGET, use_referee=False)
    results["corners"] = report("CORNERS", bt_c, CORNER_LINE)
    bt_c.to_csv(ROOT / "data" / "backtests" / "counts_corners.csv", index=False)

    cards = load_counts(conn, "cards")
    cards = cards[cards["season"].isin(TRAIN_SEASONS)]
    bt_n = walk(cards, TARGET, use_referee=False)
    results["cards_no_ref"] = report("CARDS — no referee factor", bt_n, CARD_LINE)
    bt_r = walk(cards, TARGET, use_referee=True)
    results["cards_ref"] = report("CARDS — with referee factor", bt_r, CARD_LINE)
    bt_r.to_csv(ROOT / "data" / "backtests" / "counts_cards.csv", index=False)

    print(f"\n{'=' * 62}\nDoes the referee factor earn its place?")
    a, b = results["cards_no_ref"], results["cards_ref"]
    print(f"  MAE       {a['mae']:.3f} -> {b['mae']:.3f} "
          f"({b['mae'] - a['mae']:+.3f})")
    print(f"  Brier     {a['brier']:.4f} -> {b['brier']:.4f} "
          f"({b['brier'] - a['brier']:+.4f})")
    print(f"  log-loss  {a['logloss']:.4f} -> {b['logloss']:.4f} "
          f"({b['logloss'] - a['logloss']:+.4f})")
    spread = bt_r["ref_factor"]
    print(f"  referee factors seen: {spread.min():.3f} to {spread.max():.3f} "
          f"(1.0 = average referee)")

    print(f"\n{'=' * 62}\nHow noisy is each quantity? (spread of the actual "
          f"total around its own mean)")
    for label, bt in (("corners", bt_c), ("cards", bt_r)):
        sd = bt.actual.std()
        print(f"  {label:<8} mean {bt.actual.mean():.2f}  sd {sd:.2f}  "
              f"sd/mean {sd / bt.actual.mean():.2f}")


if __name__ == "__main__":
    main()


def sweep():
    """Sensitivity check, run explicitly with --sweep.

    Both models over-predict, and the goals model has a documented version of
    the same problem: the league level drifts between seasons and a slow decay
    keeps expecting the old one. This asks whether faster forgetting closes the
    gap, or whether the models simply have nothing to say.

    EXPLORATORY. The shipped defaults are not chosen from this table — picking
    the best row of a holdout sweep is how you manufacture a result that does
    not survive contact with next season.
    """
    conn = connect()
    for kind, line in (("corners", CORNER_LINE), ("cards", CARD_LINE)):
        df = load_counts(conn, kind)
        df = df[df["season"].isin(TRAIN_SEASONS)]
        print(f"\n=== {kind}: decay sensitivity (line >= {line})")
        print(f"{'xi':>8} {'half-life':>10} {'mean pred':>10} {'MAE':>8}"
              f" {'vs mean':>8} {'Brier':>8} {'vs base':>9}")
        for xi in (0.0015, 0.003, 0.006, 0.012, 0.02):
            rows, model, last_fit, train_mean = [], None, None, None
            target = df[df["season"] == TARGET].sort_values("date")
            for _, r in target.iterrows():
                date = pd.Timestamp(r["date"])
                if model is None or (date - last_fit).days >= 7:
                    train = df[pd.to_datetime(df["date"]) < date]
                    if len(train) < 380:
                        continue
                    model = CountModel(use_referee=False, xi=xi).fit(
                        train.to_dict("records"), as_of=date)
                    last_fit = date
                    train_mean = float(train["home_count"].mean()
                                       + train["away_count"].mean())
                lam, mu = model.expected_counts(r["home"], r["away"])
                rows.append({"pred": lam + mu, "lam": lam, "mu": mu,
                             "league_mean": train_mean,
                             "actual": float(r["home_count"] + r["away_count"])})
            b = pd.DataFrame(rows)
            mae, _ = err(b.pred, b.actual)
            lmae, _ = err(b.league_mean, b.actual)
            p = np.array([prob_at_least(a, c, line)
                          for a, c in zip(b.lam, b.mu)])
            y = (b.actual >= line).astype(int).to_numpy()
            base = float(y.mean())
            brier = float(np.mean((np.clip(p, 1e-12, 1 - 1e-12) - y) ** 2))
            brier_base = float(np.mean((np.full(len(y), base) - y) ** 2))
            print(f"{xi:>8} {0.693/xi:>9.0f}d {b.pred.mean():>10.2f}"
                  f" {mae:>8.3f} {mae - lmae:>+8.3f} {brier:>8.4f}"
                  f" {brier - brier_base:>+9.4f}")
