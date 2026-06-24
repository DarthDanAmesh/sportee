"""
nested_compare_lr_xgb.py
=========================
Rigorous comparison of LogisticRegression vs XGBoost on the same 19-feature
set, using NESTED time-based validation: hyperparameters are selected only
on each fold's inner train/val split, then evaluated ONCE on that fold's
outer test split, which the search never saw. This is repeated across 5
expanding-window outer folds, giving 5 independent test estimates per
model instead of 1 — enough to ask whether the two models' scores are
actually distinguishable from noise, not just whether one point estimate
beat another.

WHY THIS EXISTS:
A single train/val/test split with a 243-combination XGBoost grid search
(as run previously) picks the best of 243 noisy validation estimates, then
reports that as if it were an unbiased test score. The single test score
(75.27%) had a validation->test drop of 1.25pp (vs ~0.1-0.7pp typical for
the much smaller LR grid) and a CI that overlapped LR's CI by ~0.7pp.
Both are signatures of a search overfitting the validation split rather
than a genuine model improvement. This script removes that ambiguity by
re-running selection independently per fold and looking at the SPREAD of
test scores, not a single number.

OUTER FOLDS (5, expanding window, all time-correct — no shuffling):
  Fold 1: train < 2017-01-01, test  2017-01-01 to 2017-12-31
  Fold 2: train < 2018-01-01, test  2018-01-01 to 2018-12-31
  Fold 3: train < 2019-01-01, test  2019-01-01 to 2019-12-31
  Fold 4: train < 2020-01-01, test  2020-01-01 to 2020-12-31
  Fold 5: train < 2021-01-01, test  2021-01-01 to 2021-12-31
(Adjust OUTER_FOLD_BOUNDARIES below if your data's date range differs —
 the script will skip folds with insufficient data on either side.)

INNER SPLIT (within each outer fold's training data):
  Same 80/20 fit/val carve already used in train_model.py, used ONLY to
  pick hyperparameters + confidence threshold. Never touches outer test.

Usage:
    python nested_compare_lr_xgb.py
"""
from __future__ import annotations

import itertools
import math
import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

DB_FILE = "predictor.db"

# ── Outer fold boundaries (expanding window) ────────────────────────
# Each tuple is (train_cutoff_exclusive, test_start_inclusive, test_end_exclusive).
# Train = everything before train_cutoff. Test = [test_start, test_end).
OUTER_FOLD_BOUNDARIES = [
    ("2017-01-01", "2017-01-01", "2018-01-01"),
    ("2018-01-01", "2018-01-01", "2019-01-01"),
    ("2019-01-01", "2019-01-01", "2020-01-01"),
    ("2020-01-01", "2020-01-01", "2021-01-01"),
    ("2021-01-01", "2021-01-01", "2022-01-01"),
]

MIN_CONFIDENCE_CANDIDATES = [0.50, 0.55, 0.60, 0.65, 0.70]
MIN_BETS_FRACTION = 0.10
MIN_TRAIN_ROWS = 60
MIN_TEAM_MATCHES = 50

LR_C_CANDIDATES = [0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

# Trimmed XGBoost grid (36 combos instead of 243) — kept near the region
# that won in the single-split run (n_estimators=50, max_depth=6,
# learning_rate=0.3) while still covering enough of the space to be a fair
# search, not a confirmation of the prior winner.
XGB_PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [3, 6],
    "learning_rate": [0.1, 0.3],
    "subsample": [0.9, 1.0],
    "colsample_bytree": [0.9, 1.0],
}
XGB_FIXED_KWARGS = {"eval_metric": "mlogloss", "random_state": 42}


# ──────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION (19 features — identical to train_model.py
#  Step A: n_matches removed, no Step B diffs, so this matches the
#  same feature set the original XGBoost experiment used)
# ──────────────────────────────────────────────────────────────
def get_market_features(odds_h, odds_d, odds_a) -> Tuple[float, float, float, float]:
    if odds_h and odds_d and odds_a and odds_h > 1.0 and odds_d > 1.0 and odds_a > 1.0:
        raw_h, raw_d, raw_a = 1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a
        total = raw_h + raw_d + raw_a
        prob_h, prob_d, prob_a = raw_h / total, raw_d / total, raw_a / total
        entropy = 0.0
        for p in [prob_h, prob_d, prob_a]:
            if p > 0:
                entropy -= p * math.log2(p)
        market_confidence = 1.0 - (entropy / math.log2(3))
        return prob_h, prob_d, prob_a, market_confidence
    return 1 / 3, 1 / 3, 1 / 3, 0.0


def build_team_index(rows) -> Dict[str, List[dict]]:
    index: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        d, home, away = r["match_date"], r["home_team"], r["away_team"]
        hs, as_ = r["home_score"], r["away_score"]
        home_res = "W" if hs > as_ else ("L" if hs < as_ else "D")
        away_res = "L" if hs > as_ else ("W" if hs < as_ else "D")
        index[home].append({"date": d, "scored": hs, "conceded": as_, "result": home_res,
                             "pts": 3 if home_res == "W" else (1 if home_res == "D" else 0)})
        index[away].append({"date": d, "scored": as_, "conceded": hs, "result": away_res,
                             "pts": 3 if away_res == "W" else (1 if away_res == "D" else 0)})
    for team in index:
        index[team].sort(key=lambda x: x["date"])
    return index


def get_team_features(team: str, before_date: str, index: Dict[str, List[dict]]) -> dict:
    history = [m for m in index.get(team, []) if m["date"] < before_date]
    if not history:
        return {"win_rate": 0.0, "attack": 0.0, "defense": 0.0, "form_points": 0.5,
                "points": 0.0, "ppg": 0.0, "recent_attack": 0.0, "recent_defense": 0.0}
    n = len(history)
    wins = sum(1 for m in history if m["result"] == "W")
    goals_scored = sum(m["scored"] for m in history)
    goals_conceded = sum(m["conceded"] for m in history)
    pts = sum(m["pts"] for m in history)
    recent = history[-10:]
    results = [m["result"] for m in recent]
    vals = {"W": 1.0, "D": 0.5, "L": 0.0}
    weights = list(range(1, len(results) + 1))
    form_points = sum(vals.get(r, 0.0) * w for r, w in zip(results, weights)) / sum(weights)
    n_recent = len(recent)
    recent_attack = sum(m["scored"] for m in recent) / n_recent
    recent_defense = sum(m["conceded"] for m in recent) / n_recent
    return {"win_rate": wins / n, "attack": goals_scored / n, "defense": goals_conceded / n,
            "form_points": form_points, "points": pts, "ppg": pts / n,
            "recent_attack": recent_attack, "recent_defense": recent_defense}


def build_odds_lookup(rows) -> Dict[Tuple[str, str, str], Tuple]:
    lookup = {}
    for r in rows:
        key = (r["match_date"], r["home_team"], r["away_team"])
        odds_h = r["odds_h"] if "odds_h" in r.keys() else None
        odds_d = r["odds_d"] if "odds_d" in r.keys() else None
        odds_a = r["odds_a"] if "odds_a" in r.keys() else None
        lookup[key] = (odds_h, odds_d, odds_a)
    return lookup


def extract_features(home, away, match_date, index, odds_lookup) -> np.ndarray:
    hf = get_team_features(home, match_date, index)
    af = get_team_features(away, match_date, index)
    odds_h, odds_d, odds_a = odds_lookup.get((match_date, home, away), (None, None, None))
    prob_h, prob_d, prob_a, market_conf = get_market_features(odds_h, odds_d, odds_a)
    points_diff = hf["points"] - af["points"]
    return np.array([
        hf["win_rate"], af["win_rate"], hf["attack"], hf["defense"], af["attack"], af["defense"],
        hf["form_points"], af["form_points"], prob_h, prob_d, prob_a, market_conf, points_diff,
        hf["ppg"], af["ppg"], hf["recent_attack"], hf["recent_defense"],
        af["recent_attack"], af["recent_defense"],
    ], dtype=np.float32)


# ──────────────────────────────────────────────────────────────
#  THRESHOLD SEARCH (shared by both models)
# ──────────────────────────────────────────────────────────────
def best_threshold(probs: np.ndarray, y_true: np.ndarray,
                    candidates: List[float], min_fraction: float) -> dict:
    n_total = len(y_true)
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    best = {"min_confidence": candidates[0], "accuracy": 0.0, "n_bets": 0, "coverage": 0.0}
    for mc in candidates:
        mask = max_probs >= mc
        n_bets = int(mask.sum())
        if n_bets < n_total * min_fraction:
            continue
        acc = accuracy_score(y_true[mask], preds[mask])
        if acc > best["accuracy"]:
            best = {"min_confidence": mc, "accuracy": round(float(acc), 4),
                    "n_bets": n_bets, "coverage": round(n_bets / n_total, 4)}
    return best


def gated_test_accuracy(probs_test, y_test, min_confidence) -> Optional[dict]:
    max_probs = probs_test.max(axis=1)
    preds = probs_test.argmax(axis=1)
    mask = max_probs >= min_confidence
    n_bets = int(mask.sum())
    if n_bets == 0:
        return None
    acc = accuracy_score(y_test[mask], preds[mask])
    se = math.sqrt(acc * (1 - acc) / n_bets) if n_bets > 0 else float("nan")
    return {"accuracy": float(acc), "n_bets": n_bets, "coverage": n_bets / len(y_test),
            "ci_low": acc - 1.96 * se, "ci_high": acc + 1.96 * se}


# ──────────────────────────────────────────────────────────────
#  INNER SEARCH: LOGISTIC REGRESSION
# ──────────────────────────────────────────────────────────────
def inner_search_lr(X_fit, y_fit, X_val, y_val) -> dict:
    best_overall = {"C": LR_C_CANDIDATES[0], "min_confidence": MIN_CONFIDENCE_CANDIDATES[0], "accuracy": 0.0}
    for c in LR_C_CANDIDATES:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                LogisticRegression(C=c, max_iter=2000, solver="lbfgs",
                                    class_weight="balanced", random_state=42),
                method="isotonic", cv=5)),
        ])
        pipe.fit(X_fit, y_fit)
        val_probs = pipe.predict_proba(X_val)
        thresh = best_threshold(val_probs, y_val, MIN_CONFIDENCE_CANDIDATES, MIN_BETS_FRACTION)
        if thresh["accuracy"] > best_overall["accuracy"]:
            best_overall = {"C": c, "min_confidence": thresh["min_confidence"], "accuracy": thresh["accuracy"]}
    return best_overall


def fit_lr_final(X_train, y_train, C) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            LogisticRegression(C=C, max_iter=2000, solver="lbfgs",
                                class_weight="balanced", random_state=42),
            method="isotonic", cv=5)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ──────────────────────────────────────────────────────────────
#  INNER SEARCH: XGBOOST
# ──────────────────────────────────────────────────────────────
def inner_search_xgb(X_fit, y_fit, X_val, y_val) -> dict:
    best_overall = {"params": None, "min_confidence": MIN_CONFIDENCE_CANDIDATES[0], "accuracy": 0.0}
    keys = list(XGB_PARAM_GRID.keys())
    for combo in itertools.product(*[XGB_PARAM_GRID[k] for k in keys]):
        params = dict(zip(keys, combo))
        clf_kwargs = {**XGB_FIXED_KWARGS, **params}
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(XGBClassifier(**clf_kwargs), method="isotonic", cv=5)),
        ])
        pipe.fit(X_fit, y_fit)
        val_probs = pipe.predict_proba(X_val)
        thresh = best_threshold(val_probs, y_val, MIN_CONFIDENCE_CANDIDATES, MIN_BETS_FRACTION)
        if thresh["accuracy"] > best_overall["accuracy"]:
            best_overall = {"params": params, "min_confidence": thresh["min_confidence"], "accuracy": thresh["accuracy"]}
    return best_overall


def fit_xgb_final(X_train, y_train, params) -> Pipeline:
    clf_kwargs = {**XGB_FIXED_KWARGS, **params}
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(XGBClassifier(**clf_kwargs), method="isotonic", cv=5)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ──────────────────────────────────────────────────────────────
#  MAIN: NESTED COMPARISON ACROSS OUTER FOLDS
# ──────────────────────────────────────────────────────────────
def main():
    if not XGBOOST_AVAILABLE:
        print("xgboost is not installed in this environment. Install it (pip install xgboost) and re-run.")
        return

    print("=" * 70)
    print("  NESTED LR vs XGBOOST COMPARISON (5 expanding-window outer folds)")
    print("=" * 70)

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    all_rows = conn.execute(
        "SELECT match_date, home_team, away_team, home_score, away_score, odds_h, odds_d, odds_a "
        "FROM matches ORDER BY match_date"
    ).fetchall()
    conn.close()
    print(f"\nLoaded {len(all_rows):,} matches.")

    index = build_team_index(all_rows)
    odds_lookup = build_odds_lookup(all_rows)
    valid_teams = {team for team, m in index.items() if len(m) >= MIN_TEAM_MATCHES}
    print(f"Teams with >= {MIN_TEAM_MATCHES} matches: {len(valid_teams)}")

    print("\nExtracting features once for the full dataset...")
    X, y, dates = [], [], []
    label_map = {"H": 0, "D": 1, "A": 2}
    for r in all_rows:
        home, away = r["home_team"], r["away_team"]
        if home not in valid_teams or away not in valid_teams:
            continue
        hs, as_ = r["home_score"], r["away_score"]
        outcome = "H" if hs > as_ else ("A" if as_ > hs else "D")
        feats = extract_features(home, away, r["match_date"], index, odds_lookup)
        X.append(feats)
        y.append(label_map[outcome])
        dates.append(r["match_date"])
    X = np.array(X)
    y = np.array(y)
    dates = np.array(dates)
    print(f"Feature matrix: {X.shape}")

    lr_results = []
    xgb_results = []

    for fold_i, (train_cutoff, test_start, test_end) in enumerate(OUTER_FOLD_BOUNDARIES, 1):
        train_mask = dates < train_cutoff
        test_mask = (dates >= test_start) & (dates < test_end)
        X_train_full, y_train_full = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        print(f"\n{'─'*70}")
        print(f"OUTER FOLD {fold_i}: train < {train_cutoff}  |  test [{test_start}, {test_end})")
        print(f"{'─'*70}")
        print(f"  Train: {len(X_train_full):,}   Test: {len(X_test):,}")

        if len(X_train_full) < MIN_TRAIN_ROWS or len(X_test) < 50:
            print("  Skipping fold: insufficient train or test data for this date range.")
            continue

        n_train_full = len(X_train_full)
        val_size = max(int(n_train_full * 0.20), 30)
        split_idx = n_train_full - val_size
        X_fit, y_fit = X_train_full[:split_idx], y_train_full[:split_idx]
        X_val, y_val = X_train_full[split_idx:], y_train_full[split_idx:]

        # ---- LR: inner search, refit, OUTER test (touched once) ----
        lr_best = inner_search_lr(X_fit, y_fit, X_val, y_val)
        lr_pipe = fit_lr_final(X_train_full, y_train_full, lr_best["C"])
        lr_probs_test = lr_pipe.predict_proba(X_test)
        lr_test = gated_test_accuracy(lr_probs_test, y_test, lr_best["min_confidence"])
        print(f"\n  LR   inner-selected: C={lr_best['C']}  min_conf={lr_best['min_confidence']}  "
              f"(inner val acc={lr_best['accuracy']:.4f})")
        if lr_test:
            print(f"  LR   OUTER TEST: acc={lr_test['accuracy']:.4f}  n={lr_test['n_bets']}  "
                  f"cov={lr_test['coverage']:.1%}  CI=[{lr_test['ci_low']:.4f}, {lr_test['ci_high']:.4f}]")
            lr_results.append(lr_test)
        else:
            print("  LR   OUTER TEST: no predictions cleared the confidence threshold.")

        # ---- XGB: inner search, refit, OUTER test (touched once) ----
        xgb_best = inner_search_xgb(X_fit, y_fit, X_val, y_val)
        xgb_pipe = fit_xgb_final(X_train_full, y_train_full, xgb_best["params"])
        xgb_probs_test = xgb_pipe.predict_proba(X_test)
        xgb_test = gated_test_accuracy(xgb_probs_test, y_test, xgb_best["min_confidence"])
        print(f"\n  XGB  inner-selected: {xgb_best['params']}  min_conf={xgb_best['min_confidence']}  "
              f"(inner val acc={xgb_best['accuracy']:.4f})")
        if xgb_test:
            print(f"  XGB  OUTER TEST: acc={xgb_test['accuracy']:.4f}  n={xgb_test['n_bets']}  "
                  f"cov={xgb_test['coverage']:.1%}  CI=[{xgb_test['ci_low']:.4f}, {xgb_test['ci_high']:.4f}]")
            xgb_results.append(xgb_test)
        else:
            print("  XGB  OUTER TEST: no predictions cleared the confidence threshold.")

    # ── Summary across folds ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY ACROSS OUTER FOLDS")
    print("=" * 70)

    def summarize(name, results):
        if not results:
            print(f"\n  {name}: no valid folds.")
            return
        accs = [r["accuracy"] for r in results]
        covs = [r["coverage"] for r in results]
        print(f"\n  {name}  (n={len(results)} folds)")
        print(f"    Per-fold accuracy: {[round(a, 4) for a in accs]}")
        print(f"    Mean accuracy:     {np.mean(accs):.4f}")
        print(f"    Std across folds:  {np.std(accs):.4f}")
        print(f"    Mean coverage:     {np.mean(covs):.1%}")

    summarize("Logistic Regression", lr_results)
    summarize("XGBoost", xgb_results)

    if lr_results and xgb_results:
        lr_accs = np.array([r["accuracy"] for r in lr_results])
        xgb_accs = np.array([r["accuracy"] for r in xgb_results])
        diff = xgb_accs.mean() - lr_accs.mean() if len(lr_accs) == len(xgb_accs) else None
        print(f"\n  Mean(XGB) - Mean(LR) = {diff:+.4f}" if diff is not None else
              "\n  Different fold counts between models — compare per-fold output above directly.")
        print("\n  Read this as a real comparison only if the per-fold accuracies for XGB are")
        print("  consistently above LR's across most/all folds. If they crisscross (XGB wins")
        print("  some folds, LR wins others, within roughly each other's CI), the single-split")
        print("  75.3% result was very likely a lucky draw from the 243-combination search.")


if __name__ == "__main__":
    main()