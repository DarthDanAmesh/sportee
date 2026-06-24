"""
train_xgboost.py
================
Trains an XGBoost classifier on the 19-feature set (no diff features).
Uses time-correct feature extraction, validation-based hyperparameter tuning,
calibration, and confidence gating. The test set is touched only once at the end.

To switch to LightGBM, change the MODEL_TYPE variable below.
"""
from __future__ import annotations

import itertools
import json
import math
import pickle
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------
# MODEL SELECTION: choose 'xgboost' or 'lightgbm'
# --------------------------------------------------------------------
MODEL_TYPE = 'xgboost'   # change to 'lightgbm' if desired

if MODEL_TYPE == 'xgboost':
    from xgboost import XGBClassifier
    BASE_ESTIMATOR_CLASS = XGBClassifier
    # Hyperparameter grid for XGBoost
    PARAM_GRID = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.05, 0.1, 0.3],
        'subsample': [0.7, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.9, 1.0],
    }
    # additional fixed kwargs for the classifier
    FIXED_KWARGS = {
        'eval_metric': 'mlogloss',
        'random_state': 42,
    }
elif MODEL_TYPE == 'lightgbm':
    from lightgbm import LGBMClassifier
    BASE_ESTIMATOR_CLASS = LGBMClassifier
    PARAM_GRID = {
        'n_estimators': [50, 100, 200],
        'num_leaves': [31, 63, 127],
        'learning_rate': [0.05, 0.1, 0.3],
        'subsample': [0.7, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.9, 1.0],
    }
    FIXED_KWARGS = {
        'random_state': 42,
        'verbose': -1,
    }
else:
    raise ValueError("MODEL_TYPE must be 'xgboost' or 'lightgbm'")

# --------------------------------------------------------------------
# Constants (keep the same as in the original train_model.py)
# --------------------------------------------------------------------
DB_FILE = "predictor.db"
MODEL_FILE = "model.pkl"
CONFIG_FILE = "model_config.json"

MIN_CONFIDENCE_CANDIDATES = [0.50, 0.55, 0.60, 0.65, 0.70]
MIN_BETS_FRACTION = 0.10          # gated predictions must cover at least 10% of validation
TRAIN_CUTOFF = "2018-07-01"

MIN_TRAIN_ROWS = 60
MIN_PER_CLASS = 10
MIN_TEAM_MATCHES = 50            # filter out teams with insufficient history

# --------------------------------------------------------------------
# FEATURE EXTRACTION (19 features, no diff features)
# This is the same as the "Step A" version (home_n_matches/away_n_matches
# removed, and no Step B diff features). 19 features total.
# --------------------------------------------------------------------
def get_market_features(odds_h, odds_d, odds_a) -> Tuple[float, float, float, float]:
    """Convert odds to implied probabilities and market confidence (1 - entropy)."""
    if odds_h and odds_d and odds_a and odds_h > 1.0 and odds_d > 1.0 and odds_a > 1.0:
        raw_h, raw_d, raw_a = 1.0/odds_h, 1.0/odds_d, 1.0/odds_a
        total = raw_h + raw_d + raw_a
        prob_h, prob_d, prob_a = raw_h/total, raw_d/total, raw_a/total

        entropy = 0.0
        for p in [prob_h, prob_d, prob_a]:
            if p > 0:
                entropy -= p * math.log2(p)
        market_confidence = 1.0 - (entropy / math.log2(3))
        return prob_h, prob_d, prob_a, market_confidence
    return 1/3, 1/3, 1/3, 0.0

def build_team_index(rows) -> Dict[str, List[dict]]:
    index: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        d, home, away = r["match_date"], r["home_team"], r["away_team"]
        hs, as_ = r["home_score"], r["away_score"]
        home_res = "W" if hs > as_ else ("L" if hs < as_ else "D")
        away_res = "L" if hs > as_ else ("W" if hs < as_ else "D")

        index[home].append({"date": d, "venue": "home", "scored": hs, "conceded": as_,
                            "result": home_res, "pts": 3 if home_res == "W" else (1 if home_res == "D" else 0)})
        index[away].append({"date": d, "venue": "away", "scored": as_, "conceded": hs,
                            "result": away_res, "pts": 3 if away_res == "W" else (1 if away_res == "D" else 0)})
    for team in index:
        index[team].sort(key=lambda x: x["date"])
    return index

def get_team_features(team: str, before_date: str, index: Dict[str, List[dict]]) -> dict:
    history = [m for m in index.get(team, []) if m["date"] < before_date]
    if not history:
        return {
            "win_rate": 0.0, "attack": 0.0, "defense": 0.0, "form_points": 0.5,
            "points": 0.0, "n_matches": 0, "ppg": 0.0,
            "recent_attack": 0.0, "recent_defense": 0.0,
        }
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

    return {
        "win_rate": wins / n,
        "attack": goals_scored / n,
        "defense": goals_conceded / n,
        "form_points": form_points,
        "points": pts,
        "n_matches": n,
        "ppg": pts / n,
        "recent_attack": recent_attack,
        "recent_defense": recent_defense,
    }

def build_odds_lookup(rows) -> Dict[Tuple[str, str, str], Tuple[Optional[float], Optional[float], Optional[float]]]:
    lookup = {}
    for r in rows:
        key = (r["match_date"], r["home_team"], r["away_team"])
        odds_h = r["odds_h"] if "odds_h" in r.keys() else None
        odds_d = r["odds_d"] if "odds_d" in r.keys() else None
        odds_a = r["odds_a"] if "odds_a" in r.keys() else None
        lookup[key] = (odds_h, odds_d, odds_a)
    return lookup

def extract_features(home: str, away: str, match_date: str,
                     index: Dict[str, List[dict]],
                     odds_lookup: Dict[Tuple[str, str, str], Tuple]) -> np.ndarray:
    hf = get_team_features(home, match_date, index)
    af = get_team_features(away, match_date, index)

    odds_h, odds_d, odds_a = odds_lookup.get((match_date, home, away), (None, None, None))
    prob_h, prob_d, prob_a, market_conf = get_market_features(odds_h, odds_d, odds_a)
    points_diff = hf["points"] - af["points"]

    # 19 features (no n_matches, no diff features)
    return np.array([
        hf["win_rate"], af["win_rate"],
        hf["attack"], hf["defense"],
        af["attack"], af["defense"],
        hf["form_points"], af["form_points"],
        prob_h, prob_d, prob_a,
        market_conf,
        points_diff,
        # home_n_matches / away_n_matches REMOVED
        hf["ppg"], af["ppg"],
        hf["recent_attack"], hf["recent_defense"],
        af["recent_attack"], af["recent_defense"],
    ], dtype=np.float32)

FEATURE_NAMES = [
    "home_win_rate", "away_win_rate",
    "home_attack", "home_defense", "away_attack", "away_defense",
    "home_form_points", "away_form_points",
    "prob_h", "prob_d", "prob_a", "market_confidence",
    "points_difference",
    "home_ppg", "away_ppg",
    "home_recent_attack", "home_recent_defense",
    "away_recent_attack", "away_recent_defense",
]

# --------------------------------------------------------------------
# CONFIDENCE THRESHOLD GRID SEARCH (unchanged)
# --------------------------------------------------------------------
def grid_search_thresholds(main_probs: np.ndarray, y_true: np.ndarray,
                           min_conf_candidates: List[float], min_fraction: float,
                           verbose: bool = True) -> dict:
    n_total = len(y_true)
    max_probs = main_probs.max(axis=1)
    main_preds = main_probs.argmax(axis=1)

    best = {"min_confidence": min_conf_candidates[0], "accuracy": 0.0, "n_bets": 0, "coverage": 0.0}
    results = []

    for mc in min_conf_candidates:
        mask = max_probs >= mc
        n_bets = int(mask.sum())
        if n_bets < n_total * min_fraction:
            continue
        acc = accuracy_score(y_true[mask], main_preds[mask])
        coverage = n_bets / n_total
        results.append({"min_confidence": mc, "accuracy": round(float(acc), 4),
                        "n_bets": n_bets, "coverage": round(coverage, 4)})
        if verbose:
            print(f"  min_conf={mc:.2f}  bets={n_bets}/{n_total} ({coverage:.1%})  acc={acc:.4f}")
        if acc > best["accuracy"]:
            best = {"min_confidence": mc, "accuracy": round(float(acc), 4),
                    "n_bets": n_bets, "coverage": round(coverage, 4)}
    return {"best": best, "grid": results}

# --------------------------------------------------------------------
# HYPERPARAMETER + THRESHOLD GRID SEARCH FOR BOOSTED TREES
# --------------------------------------------------------------------
def grid_search_boost(X_fit: np.ndarray, y_fit: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      param_grid: dict,
                      min_conf_candidates: List[float],
                      min_fraction: float) -> dict:
    print("\n" + "─" * 60)
    print(f"{MODEL_TYPE.upper()} HYPERPARAMETER + THRESHOLD GRID SEARCH")
    print("─" * 60)

    overall_best = {
        "params": None,
        "min_confidence": min_conf_candidates[0],
        "accuracy": 0.0,
        "n_bets": 0,
        "coverage": 0.0
    }
    results = []

    # Flatten the grid
    keys = list(param_grid.keys())
    for combo in itertools.product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        print(f"\nTrying: {params}")

        # Build classifier with these parameters
        clf_kwargs = {**FIXED_KWARGS, **params}
        base_clf = BASE_ESTIMATOR_CLASS(**clf_kwargs)

        calibrated_clf = CalibratedClassifierCV(base_clf, method='isotonic', cv=5)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', calibrated_clf)
        ])
        pipeline.fit(X_fit, y_fit)

        val_probs = pipeline.predict_proba(X_val)
        val_preds_all = val_probs.argmax(axis=1)
        val_acc_all = accuracy_score(y_val, val_preds_all)
        val_ll_all = log_loss(y_val, val_probs, labels=[0, 1, 2])

        print(f"  Overall val accuracy: {val_acc_all:.4f}  log_loss: {val_ll_all:.4f}")
        thresh_result = grid_search_thresholds(val_probs, y_val, min_conf_candidates, min_fraction)
        best_for_params = thresh_result["best"]

        results.append({
            "params": params,
            "val_accuracy_all": round(float(val_acc_all), 4),
            "val_log_loss_all": round(float(val_ll_all), 4),
            "best_min_confidence": best_for_params["min_confidence"],
            "best_gated_accuracy": best_for_params["accuracy"],
            "best_gated_n_bets": best_for_params["n_bets"],
            "best_gated_coverage": best_for_params["coverage"],
        })

        if best_for_params["accuracy"] > overall_best["accuracy"]:
            overall_best = {
                "params": params,
                "min_confidence": best_for_params["min_confidence"],
                "accuracy": best_for_params["accuracy"],
                "n_bets": best_for_params["n_bets"],
                "coverage": best_for_params["coverage"],
            }

    # Print summary
    print("\n" + "─" * 60)
    print(f"{MODEL_TYPE.upper()} GRID SUMMARY (sorted by gated validation accuracy)")
    print("─" * 60)
    for row in sorted(results, key=lambda r: r["best_gated_accuracy"], reverse=True):
        p = row["params"]
        # Truncated display – each param printed
        param_str = "  ".join(f"{k}={v}" for k, v in p.items())
        print(f"  {param_str:60s}  gated_acc={row['best_gated_accuracy']:.4f}  cov={row['best_gated_coverage']:.1%}")

    print(f"\nSelected: {overall_best['params']}  min_confidence={overall_best['min_confidence']}  "
          f"(val gated accuracy={overall_best['accuracy']:.4f})")
    return {"best": overall_best, "grid": results}

# --------------------------------------------------------------------
# MAIN TRAINING ROUTINE
# --------------------------------------------------------------------
def train():
    print("=" * 60)
    print(f"  MATCH PREDICTOR — TRAINING WITH {MODEL_TYPE.upper()}")
    print("=" * 60)

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    all_rows = conn.execute(
        "SELECT match_date, home_team, away_team, home_score, away_score, odds_h, odds_d, odds_a "
        "FROM matches ORDER BY match_date"
    ).fetchall()
    conn.close()
    print(f"\nLoaded {len(all_rows):,} matches from {DB_FILE}")

    print("Building time-correct team index...")
    index = build_team_index(all_rows)

    print("Building O(1) odds lookup...")
    odds_lookup = build_odds_lookup(all_rows)

    valid_teams = {team for team, matches in index.items() if len(matches) >= MIN_TEAM_MATCHES}
    print(f"Teams indexed: {len(index)} | Teams with >= {MIN_TEAM_MATCHES} matches: {len(valid_teams)}")

    print("\nExtracting features (19 features, time-correct)...")
    X, y, dates = [], [], []
    label_map = {"H": 0, "D": 1, "A": 2}
    skipped = 0

    for r in all_rows:
        home, away = r["home_team"], r["away_team"]
        if home not in valid_teams or away not in valid_teams:
            skipped += 1
            continue
        hs, as_ = r["home_score"], r["away_score"]
        match_date = r["match_date"]
        outcome = "H" if hs > as_ else ("A" if as_ > hs else "D")

        feats = extract_features(home, away, match_date, index, odds_lookup)
        X.append(feats)
        y.append(label_map[outcome])
        dates.append(match_date)

    X = np.array(X)
    y = np.array(y)
    dates = np.array(dates)
    print(f"Feature matrix: {X.shape}  (skipped {skipped} matches due to team filter)")

    if len(X) == 0:
        print("\n❌ No usable rows. Add more match history or lower MIN_TEAM_MATCHES.")
        sys.exit(1)

    # Time-based split
    train_mask = dates < TRAIN_CUTOFF
    test_mask = dates >= TRAIN_CUTOFF
    X_train_full, y_train_full = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"\nTime split at {TRAIN_CUTOFF}: Train={len(X_train_full):,} | Test={len(X_test):,}")

    if len(X_train_full) < MIN_TRAIN_ROWS:
        print(f"\n❌ Only {len(X_train_full)} training rows. Add more data.")
        sys.exit(1)

    # Carve validation split (last 20% of train)
    n_train_full = len(X_train_full)
    val_size = max(int(n_train_full * 0.20), 30)
    split_idx = n_train_full - val_size
    X_fit, y_fit = X_train_full[:split_idx], y_train_full[:split_idx]
    X_val, y_val = X_train_full[split_idx:], y_train_full[split_idx:]
    print(f"Fit: {len(X_fit):,} | Validation: {len(X_val):,}")

    # Perform grid search for model hyperparameters + threshold
    grid_result = grid_search_boost(
        X_fit, y_fit, X_val, y_val,
        PARAM_GRID,
        MIN_CONFIDENCE_CANDIDATES,
        MIN_BETS_FRACTION
    )
    best = grid_result["best"]
    best_params = best["params"]

    if best["n_bets"] == 0:
        print(f"\n⚠️  No (params, threshold) combo cleared coverage floor. Using default params and min_confidence=0.50.")
        best_params = {k: PARAM_GRID[k][0] for k in PARAM_GRID}   # first combo
        best = {"params": best_params, "min_confidence": 0.50, "accuracy": None, "n_bets": 0, "coverage": 0.0}

    # Refit on full training with selected hyperparameters
    print(f"\nRefitting on full training set with selected params: {best_params}...")
    clf_kwargs = {**FIXED_KWARGS, **best_params}
    base_clf = BASE_ESTIMATOR_CLASS(**clf_kwargs)
    pipeline_final = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', CalibratedClassifierCV(base_clf, method='isotonic', cv=5))
    ])
    pipeline_final.fit(X_train_full, y_train_full)

    # Final test evaluation
    print("\n" + "─" * 60)
    print("TEST SET EVALUATION")
    print("─" * 60)
    probs_test = pipeline_final.predict_proba(X_test)
    preds_all = probs_test.argmax(axis=1)
    acc_all = accuracy_score(y_test, preds_all)
    ll = log_loss(y_test, probs_test, labels=[0, 1, 2])
    print(f"All matches: accuracy={acc_all:.4f}  log_loss={ll:.4f}")

    mask = probs_test.max(axis=1) >= best["min_confidence"]
    n_bets_test = int(mask.sum())
    acc_thresh_test = accuracy_score(y_test[mask], preds_all[mask]) if n_bets_test > 0 else None

    if n_bets_test > 0:
        print(f"\nGated (min_conf={best['min_confidence']}): {acc_thresh_test:.4f} on {n_bets_test}/{len(y_test)} matches")
        se = math.sqrt(acc_thresh_test * (1 - acc_thresh_test) / n_bets_test)
        print(f"  Approx. 95% CI: [{acc_thresh_test - 1.96*se:.4f}, {acc_thresh_test + 1.96*se:.4f}]  (n={n_bets_test})")

    # Save model and config
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(pipeline_final, f)

    config = {
        "model_type": MODEL_TYPE,
        "best_params": best_params,
        "best_min_confidence": best["min_confidence"],
        "grid_search_results": grid_result["grid"],
        "feature_names": FEATURE_NAMES,
        "label_map": {"H": 0, "D": 1, "A": 2},
        "label_names": ["H", "D", "A"],
        "train_cutoff": TRAIN_CUTOFF,
        "train_size": int(len(X_train_full)),
        "test_size": int(len(X_test)),
        "test_accuracy_all": round(float(acc_all), 4),
        "test_accuracy_thresh": round(float(acc_thresh_test), 4) if acc_thresh_test else None,
        "test_log_loss": round(float(ll), 4),
        "test_n_bets": n_bets_test,
        "trained_at": datetime.now().isoformat(),
        "n_features": len(FEATURE_NAMES),
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE  ({MODEL_TYPE.upper()})")
    print(f"  Test accuracy (gated): {acc_thresh_test:.1%}" if acc_thresh_test else "  Test accuracy: N/A")
    print("=" * 60)

if __name__ == "__main__":
    train()