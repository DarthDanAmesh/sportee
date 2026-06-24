"""
train_model.py
==============
Trains a time-correct ML classifier replicating the reference Experiment 3 (~74% accuracy).
  - 19-feature set (team strength, form, market odds, points-per-game, match counts, recency)
  - MAIN classifier: LogisticRegression (C tuned via validation grid search, balanced) + isotonic calibration
  - Uncertainty model REMOVED (single-stage confidence gating only)
  - Filters training data to teams with >= 50 matches for stable statistics
  - Time-correct feature extraction (no lookahead)

CHANGE LOG (regularization):
  - Added a C (inverse regularization strength) grid search, run on the SAME
    validation split already used for confidence-threshold selection.
  - For each candidate C, fits a calibrated pipeline on the fit split, then
    reuses grid_search_thresholds() to find that C's best (min_confidence,
    accuracy) pair on validation. The C/threshold combo with the highest
    validation accuracy wins.
  - The test set is touched exactly once, at the very end, using the winning
    C and threshold — never used to pick C, so this is a real comparison,
    not test-set tuning dressed up as one.
  - model_config.json now also records best_C and the full C grid, so you
    can see how sensitive results were to C across runs/datasets.

CHANGE LOG (feature selection — Step A):
  - REMOVED home_n_matches / away_n_matches. inspect_features.py measured
    these as near-zero coefficient in ALL three classes (max |coef| = 0.0195
    for home, 0.0254 for away, on real ~185k-match data) and as the most
    collinear pair in the correlation matrix among the new features
    (r=0.80 home_n_matches <-> away_n_matches). No theoretical reason for
    them to add signal once strength/form/odds are already present, and
    measurement confirmed they weren't. 21 -> 19 features.
  - get_team_features() still computes n_matches internally (cheap, may be
    useful for other diagnostics later) — it's just no longer emitted into
    the feature vector consumed by the classifier.
  - This is a REMOVAL-ONLY change, isolated from any other feature
    experiment, specifically so any accuracy delta can be attributed to
    this change alone and not confused with a simultaneous addition.

CHANGE LOG (feature engineering — Step B):
  - ADDED six explicit relative-strength (difference) features:
    attack_diff, defense_diff, form_diff, ppg_diff, recent_attack_diff,
    recent_defense_diff. All defined as (home - away), with NO sign
    flipping for the defense pair — a positive defense_diff means the
    home team conceded MORE goals/game than the away team. Every diff
    uses the same convention so there's nothing to remember per-feature;
    the linear model is free to learn a negative coefficient where that
    fits (e.g. for defense_diff, since conceding more is bad).
  - Rationale: logistic regression is handed home/away values separately
    today and has to learn the comparison implicitly via its own
    coefficients on two correlated columns. Difference features hand it
    the comparison directly. This is additive only — none of the 19
    Step-A features were removed or changed. 19 -> 25 features.
  - This is an ADDITION-ONLY change, isolated from the Step-A removal,
    so any accuracy delta here is attributable to the diffs alone.
  - Per the explicit constraint given for this work: no new classifier,
    no draw-specific tuning, no symmetry features were added alongside
    this — these are six new columns into the existing single classifier,
    nothing else in the pipeline changed.
"""
from __future__ import annotations

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DB_FILE = "predictor.db"
MODEL_FILE = "model.pkl"
CONFIG_FILE = "model_config.json"

# ── Threshold grid ─────────────────────────────────────────────
MIN_CONFIDENCE_CANDIDATES = [0.50, 0.55, 0.60, 0.65, 0.70]
MIN_BETS_FRACTION = 0.10   # must predict on at least 10% of the validation set
TRAIN_CUTOFF = "2018-07-01"

# ── Regularization grid (NEW) ───────────────────────────────────
# C is inverse regularization strength: smaller C = stronger regularization.
# Kept on a log scale, centered on the original hardcoded value (1.0) so the
# old behavior is one candidate among several, not discarded.
C_CANDIDATES = [0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

MIN_TRAIN_ROWS = 60
MIN_PER_CLASS = 10
MIN_TEAM_MATCHES = 50  # Filter out noisy teams with insufficient history


# ──────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION  (13 features, time-correct)
# ──────────────────────────────────────────────────────────────

def get_market_features(odds_h, odds_d, odds_a) -> Tuple[float, float, float, float]:
    """Convert odds to implied probabilities and market confidence (1 - entropy)."""
    if odds_h and odds_d and odds_a and odds_h > 1.0 and odds_d > 1.0 and odds_a > 1.0:
        raw_h, raw_d, raw_a = 1.0/odds_h, 1.0/odds_d, 1.0/odds_a
        total = raw_h + raw_d + raw_a
        prob_h, prob_d, prob_a = raw_h/total, raw_d/total, raw_a/total

        entropy = 0.0
        for p in [prob_h, prob_d, prob_a]:
            if p > 0: entropy -= p * math.log2(p)
        market_confidence = 1.0 - (entropy / math.log2(3))
        return prob_h, prob_d, prob_a, market_confidence
    return 1/3, 1/3, 1/3, 0.0  # Default to uniform if no odds


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
    for team in index: index[team].sort(key=lambda x: x["date"])
    return index


def get_team_features(team: str, before_date: str, index: Dict[str, List[dict]]) -> dict:
    history = [m for m in index.get(team, []) if m["date"] < before_date]
    if not history:
        return {
            "win_rate": 0.0, "attack": 0.0, "defense": 0.0, "form_points": 0.5, "points": 0.0,
            "n_matches": 0, "ppg": 0.0, "recent_attack": 0.0, "recent_defense": 0.0,
        }

    n = len(history)
    wins = sum(1 for m in history if m["result"] == "W")
    goals_scored = sum(m["scored"] for m in history)
    goals_conceded = sum(m["conceded"] for m in history)
    pts = sum(m["pts"] for m in history)

    # Last-10-match window, reused for form, recent attack/defense — already
    # time-correct (history was filtered to before_date above, no lookahead).
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
    """
    O(1) replacement for the O(N) per-row scan that used to live inside
    extract_features(). Built once per training run; extract_features()
    does a single dict lookup instead of scanning all_rows on every call.
    Previously: extract_features() looped over all_rows (~185k rows) for
    EVERY one of the ~185k training rows -> ~34 billion comparisons.
    """
    lookup: Dict[Tuple[str, str, str], Tuple[Optional[float], Optional[float], Optional[float]]] = {}
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

    # Step B: explicit relative-strength (difference) features. Consistent
    # sign convention throughout: every diff is (home - away), including
    # defense_diff (goals conceded), so a positive defense_diff means the
    # home team conceded MORE than the away team (i.e. worse defense) —
    # the model is linear and can learn whichever sign of coefficient fits;
    # what matters is that ALL six diffs use the same home-minus-away
    # convention, so nobody has to remember which ones are flipped.
    attack_diff = hf["attack"] - af["attack"]
    defense_diff = hf["defense"] - af["defense"]
    form_diff = hf["form_points"] - af["form_points"]
    ppg_diff = hf["ppg"] - af["ppg"]
    recent_attack_diff = hf["recent_attack"] - af["recent_attack"]
    recent_defense_diff = hf["recent_defense"] - af["recent_defense"]

    return np.array([
        hf["win_rate"], af["win_rate"],
        hf["attack"], hf["defense"],
        af["attack"], af["defense"],
        hf["form_points"], af["form_points"],
        prob_h, prob_d, prob_a,
        market_conf,
        points_diff,
        # home_n_matches / away_n_matches REMOVED (Step A — measured
        # near-zero coefficient in all 3 classes; see CHANGE LOG).
        hf["ppg"], af["ppg"],
        hf["recent_attack"], hf["recent_defense"],
        af["recent_attack"], af["recent_defense"],
        # Step B additions (all home - away, see note above):
        attack_diff, defense_diff, form_diff,
        ppg_diff, recent_attack_diff, recent_defense_diff,
    ], dtype=np.float32)


FEATURE_NAMES = [
    "home_win_rate", "away_win_rate",
    "home_attack", "home_defense", "away_attack", "away_defense",
    "home_form_points", "away_form_points",
    "prob_h", "prob_d", "prob_a", "market_confidence",
    "points_difference",
    # home_n_matches / away_n_matches REMOVED (Step A, see comment above).
    "home_ppg", "away_ppg",
    "home_recent_attack", "home_recent_defense",
    "away_recent_attack", "away_recent_defense",
    # Step B additions: explicit relative-strength features, all (home - away):
    "attack_diff", "defense_diff", "form_diff",
    "ppg_diff", "recent_attack_diff", "recent_defense_diff",
]


# ──────────────────────────────────────────────────────────────
#  GRID SEARCH  (confidence only)
# ──────────────────────────────────────────────────────────────
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
        if n_bets < n_total * min_fraction: continue

        acc = accuracy_score(y_true[mask], main_preds[mask])
        coverage = n_bets / n_total
        results.append({"min_confidence": mc, "accuracy": round(float(acc), 4), "n_bets": n_bets, "coverage": round(coverage, 4)})
        if verbose:
            print(f"  min_conf={mc:.2f}  bets={n_bets}/{n_total} ({coverage:.1%})  acc={acc:.4f}")

        if acc > best["accuracy"]:
            best = {"min_confidence": mc, "accuracy": round(float(acc), 4), "n_bets": n_bets, "coverage": round(coverage, 4)}

    return {"best": best, "grid": results}


# ──────────────────────────────────────────────────────────────
#  REGULARIZATION GRID SEARCH  (NEW)
# ──────────────────────────────────────────────────────────────
def grid_search_regularization(X_fit: np.ndarray, y_fit: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               c_candidates: List[float],
                               min_conf_candidates: List[float],
                               min_fraction: float) -> dict:
    """
    For each candidate C: fit a calibrated pipeline on (X_fit, y_fit), then run
    the existing confidence-threshold grid search on (X_val, y_val) to find
    that C's best achievable gated accuracy. Both C and min_confidence are
    selected together from the SAME validation split — the test set is never
    touched here. This directly answers "is C=1.0 costing us accuracy?"
    without tuning on the data we report results on.
    """
    print("\n" + "─" * 60)
    print(f"REGULARIZATION GRID SEARCH  (C candidates: {c_candidates})")
    print("─" * 60)

    overall_best = {"C": c_candidates[0], "min_confidence": min_conf_candidates[0],
                     "accuracy": 0.0, "n_bets": 0, "coverage": 0.0}
    per_c_results = []

    for c in c_candidates:
        base_clf = LogisticRegression(C=c, max_iter=2000, solver='lbfgs',
                                      class_weight='balanced', random_state=42)
        calibrated_clf = CalibratedClassifierCV(base_clf, method='isotonic', cv=5)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', calibrated_clf)
        ])
        pipeline.fit(X_fit, y_fit)

        val_probs = pipeline.predict_proba(X_val)
        # Also record plain (ungated) validation accuracy/log_loss for context —
        # useful for noticing if a C overfits (high gated acc, poor overall acc).
        val_preds_all = val_probs.argmax(axis=1)
        val_acc_all = accuracy_score(y_val, val_preds_all)
        val_ll_all = log_loss(y_val, val_probs, labels=[0, 1, 2])

        print(f"\nC={c}:  (val overall accuracy={val_acc_all:.4f}  log_loss={val_ll_all:.4f})")
        thresh_result = grid_search_thresholds(val_probs, y_val, min_conf_candidates, min_fraction)
        best_for_c = thresh_result["best"]

        per_c_results.append({
            "C": c,
            "val_accuracy_all": round(float(val_acc_all), 4),
            "val_log_loss_all": round(float(val_ll_all), 4),
            "best_min_confidence": best_for_c["min_confidence"],
            "best_gated_accuracy": best_for_c["accuracy"],
            "best_gated_n_bets": best_for_c["n_bets"],
            "best_gated_coverage": best_for_c["coverage"],
        })

        if best_for_c["accuracy"] > overall_best["accuracy"]:
            overall_best = {
                "C": c,
                "min_confidence": best_for_c["min_confidence"],
                "accuracy": best_for_c["accuracy"],
                "n_bets": best_for_c["n_bets"],
                "coverage": best_for_c["coverage"],
            }

    print("\n" + "─" * 60)
    print("REGULARIZATION GRID SUMMARY (sorted by gated validation accuracy)")
    print("─" * 60)
    print(f"  {'C':>8} {'val_acc_all':>12} {'val_logloss':>12} {'min_conf':>9} {'gated_acc':>10} {'coverage':>9}")
    for row in sorted(per_c_results, key=lambda r: r["best_gated_accuracy"], reverse=True):
        print(f"  {row['C']:>8} {row['val_accuracy_all']:>12.4f} {row['val_log_loss_all']:>12.4f} "
              f"{row['best_min_confidence']:>9.2f} {row['best_gated_accuracy']:>10.4f} {row['best_gated_coverage']:>9.1%}")

    spread = max(r["best_gated_accuracy"] for r in per_c_results) - min(r["best_gated_accuracy"] for r in per_c_results)
    print(f"\n  Spread across C values (gated accuracy): {spread:.4f}")
    if spread < 0.01:
        print("  → C has negligible effect here (<1pt). Regularization is NOT the explanation for any small gap.")
    else:
        print("  → C has a non-trivial effect. Worth keeping the tuned value.")

    print(f"\n  Selected: C={overall_best['C']}  min_confidence={overall_best['min_confidence']}  "
          f"(val gated accuracy={overall_best['accuracy']:.4f})")

    return {"best": overall_best, "grid": per_c_results}


# ──────────────────────────────────────────────────────────────
#  MAIN TRAINING ROUTINE
# ──────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  MATCH PREDICTOR — ML TRAINING (Experiment 3 Formula)")
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

    # Filter to stable teams
    valid_teams = {team for team, matches in index.items() if len(matches) >= MIN_TEAM_MATCHES}
    print(f"Teams indexed: {len(index)} | Teams with >= {MIN_TEAM_MATCHES} matches: {len(valid_teams)}")

    print("\nExtracting features (21 features, time-correct)...")
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

    # ── Time-based train/test split ────────────────────────────
    train_mask = dates < TRAIN_CUTOFF
    test_mask  = dates >= TRAIN_CUTOFF
    X_train_full, y_train_full = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"\nTime split at {TRAIN_CUTOFF}: Train={len(X_train_full):,} | Test={len(X_test):,}")

    if len(X_train_full) < MIN_TRAIN_ROWS:
        print(f"\n❌ Only {len(X_train_full)} training rows. Add more data.")
        sys.exit(1)

    # ── Carve validation split ─────────────────────────────────
    n_train_full = len(X_train_full)
    val_size = max(int(n_train_full * 0.20), 30)
    split_idx = n_train_full - val_size
    X_fit, y_fit = X_train_full[:split_idx], y_train_full[:split_idx]
    X_val, y_val = X_train_full[split_idx:], y_train_full[split_idx:]
    print(f"Fit: {len(X_fit):,} | Validation: {len(X_val):,}")

    # ── Regularization (C) + threshold grid search on VALIDATION (NEW) ──
    # This replaces the old hardcoded C=1.0 with a value selected the same
    # way min_confidence already was: by validation performance, never test.
    reg_result = grid_search_regularization(
        X_fit, y_fit, X_val, y_val, C_CANDIDATES, MIN_CONFIDENCE_CANDIDATES, MIN_BETS_FRACTION
    )
    best = reg_result["best"]
    best_C = best["C"]

    if best["n_bets"] == 0:
        print(f"\n⚠️  No (C, threshold) combination cleared coverage floor. Defaulting to C=1.0, min_confidence=0.50.")
        best_C = 1.0
        best = {"C": 1.0, "min_confidence": 0.50, "accuracy": None, "n_bets": 0, "coverage": 0.0}

    # ── Refit on FULL train using the selected C ────────────────
    print(f"\nRefitting on full training set with selected C={best_C}...")
    pipeline_final = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', CalibratedClassifierCV(LogisticRegression(C=best_C, max_iter=2000, solver='lbfgs',
                                                          class_weight='balanced', random_state=42),
                                       method='isotonic', cv=5))
    ])
    pipeline_final.fit(X_train_full, y_train_full)

    # ── Final TEST evaluation ──────────────────────────────────
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
        print(f"\nGated (C={best_C}, min_conf={best['min_confidence']}): {acc_thresh_test:.4f} on {n_bets_test}/{len(y_test)} matches")
        se = math.sqrt(acc_thresh_test * (1 - acc_thresh_test) / n_bets_test)
        print(f"  Approx. 95% CI: [{acc_thresh_test - 1.96*se:.4f}, {acc_thresh_test + 1.96*se:.4f}]  (n={n_bets_test})")

    # ── Save models ─────────────────────────────────────────────
    with open(MODEL_FILE, "wb") as f: pickle.dump(pipeline_final, f)

    config = {
        "min_confidence": best["min_confidence"],
        "best_C": best_C,
        "C_grid": reg_result["grid"],
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
    with open(CONFIG_FILE, "w") as f: json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Selected C: {best_C}")
    print(f"  Test accuracy (gated): {acc_thresh_test:.1%}" if acc_thresh_test else "  Test accuracy: N/A")
    print("=" * 60)

if __name__ == "__main__":
    train()