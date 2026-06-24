"""
inspect_features.py
====================
Standalone diagnostic for the 21-feature ML model. Run AFTER train_model.py
has produced model.pkl / model_config.json. Does not retrain anything and
does not touch train/val/test selection — purely descriptive.

Two outputs:
  1. Feature correlation matrix (flags |r| > 0.7 pairs — likely redundant
     features, per the "are PPG/win_rate/points_difference all the same
     signal?" question).
  2. Coefficient audit: averages LogisticRegression coef_ across all
     CalibratedClassifierCV folds (NOT just fold [0], which would only show
     one of five CV folds and isn't representative), per class (H/D/A),
     ranked by |coefficient| on STANDARDIZED features (so magnitudes are
     comparable across features with different scales).

Usage:
    python inspect_features.py
"""
from __future__ import annotations

import json
import pickle
import sqlite3
from typing import Dict, List

import numpy as np

from train_model import (
    DB_FILE, MODEL_FILE, CONFIG_FILE,
    FEATURE_NAMES, MIN_TEAM_MATCHES,
    build_team_index, build_odds_lookup, extract_features,
)

CORRELATION_FLAG_THRESHOLD = 0.7  # |r| above this gets flagged as likely-redundant


def load_full_feature_matrix() -> tuple[np.ndarray, List[str]]:
    """Rebuild the exact same feature matrix train_model.py used (same
    team filter, same time-correct extraction), for correlation analysis."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    all_rows = conn.execute(
        "SELECT match_date, home_team, away_team, home_score, away_score, odds_h, odds_d, odds_a "
        "FROM matches ORDER BY match_date"
    ).fetchall()
    conn.close()

    index = build_team_index(all_rows)
    odds_lookup = build_odds_lookup(all_rows)
    valid_teams = {team for team, matches in index.items() if len(matches) >= MIN_TEAM_MATCHES}

    X = []
    for r in all_rows:
        home, away = r["home_team"], r["away_team"]
        if home not in valid_teams or away not in valid_teams:
            continue
        feats = extract_features(home, away, r["match_date"], index, odds_lookup)
        X.append(feats)

    return np.array(X), FEATURE_NAMES


def print_correlation_matrix(X: np.ndarray, names: List[str]) -> None:
    print("=" * 70)
    print("  FEATURE CORRELATION MATRIX")
    print("=" * 70)
    corr = np.corrcoef(X, rowvar=False)
    n = len(names)

    # Full matrix, abbreviated column headers to keep it printable
    short = [f"{i:02d}" for i in range(n)]
    print("\nFeature index key:")
    for i, name in enumerate(names):
        print(f"  {i:02d} = {name}")

    print("\n      " + " ".join(f"{s:>6}" for s in short))
    for i in range(n):
        row = " ".join(f"{corr[i, j]:6.2f}" for j in range(n))
        print(f"  {short[i]} {row}")

    print(f"\n{'─' * 70}")
    print(f"  PAIRS WITH |correlation| > {CORRELATION_FLAG_THRESHOLD}  (likely redundant)")
    print(f"{'─' * 70}")
    flagged = []
    for i in range(n):
        for j in range(i + 1, n):
            r = corr[i, j]
            if abs(r) > CORRELATION_FLAG_THRESHOLD:
                flagged.append((abs(r), names[i], names[j], r))
    if not flagged:
        print(f"  None. No feature pair exceeds |r| > {CORRELATION_FLAG_THRESHOLD}.")
    else:
        for _, a, b, r in sorted(flagged, reverse=True):
            print(f"  {a:24s} <-> {b:24s}   r = {r:+.3f}")


def print_coefficient_audit() -> None:
    print("\n" + "=" * 70)
    print("  COEFFICIENT AUDIT  (averaged across all CalibratedClassifierCV folds)")
    print("=" * 70)

    try:
        with open(MODEL_FILE, "rb") as f:
            pipeline_final = pickle.load(f)
    except FileNotFoundError:
        print(f"  No {MODEL_FILE} found. Run train_model.py first.")
        return

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    cal_clf = pipeline_final.named_steps["clf"]
    label_names = config.get("label_names", ["H", "D", "A"])

    # Average coef_ across every CV fold's fitted estimator, not just fold 0 —
    # fold 0 alone is an arbitrary 1/5 slice of the training data and isn't
    # representative of the final ensemble's actual behavior.
    fold_coefs = []
    for cc in cal_clf.calibrated_classifiers_:
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if est is None:
            print("  Could not find inner estimator on this sklearn version's "
                  "CalibratedClassifierCV — attribute name may have changed.")
            return
        fold_coefs.append(est.coef_)

    avg_coefs = np.mean(fold_coefs, axis=0)  # shape (n_classes, n_features)
    n_folds = len(fold_coefs)
    print(f"\n  Averaged over {n_folds} CV folds. Coefficients are on STANDARDIZED")
    print("  features (after the pipeline's StandardScaler), so magnitudes ARE")
    print("  comparable across features — a feature near 0 here genuinely")
    print("  contributes little to that class's log-odds.\n")

    classes_ = getattr(cal_clf, "classes_", list(range(len(label_names))))
    for class_idx, class_label in zip(classes_, [label_names[c] for c in classes_]):
        print(f"\n  Class: {class_label}")
        print(f"  {'-' * 50}")
        coefs_for_class = avg_coefs[class_idx]
        ranked = sorted(zip(FEATURE_NAMES, coefs_for_class), key=lambda x: abs(x[1]), reverse=True)
        for name, coef in ranked:
            bar_len = min(int(abs(coef) * 10), 30)
            bar = ("+" if coef >= 0 else "-") * bar_len
            print(f"    {name:24s} {coef:+.4f}  {bar}")

    print(f"\n{'─' * 70}")
    print("  NEAR-ZERO FEATURES (|avg coef| < 0.02 in ALL three classes)")
    print(f"{'─' * 70}")
    near_zero = []
    for f_idx, name in enumerate(FEATURE_NAMES):
        max_abs = max(abs(avg_coefs[c_idx, f_idx]) for c_idx in range(len(classes_)))
        if max_abs < 0.02:
            near_zero.append((name, max_abs))
    if not near_zero:
        print("  None — every feature has a non-trivial coefficient in at least one class.")
    else:
        for name, max_abs in near_zero:
            print(f"  {name:24s}  (max |coef| across classes = {max_abs:.4f})")


def main():
    X, names = load_full_feature_matrix()
    print(f"Loaded feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features\n")
    print_correlation_matrix(X, names)
    print_coefficient_audit()


if __name__ == "__main__":
    main()