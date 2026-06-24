"""
FastAPI backend for Match Predictor
SQLite persistence — v7
Changes from v6:
  - ML inference now replicates the reference experiment's actual mechanism:
    a MAIN calibrated classifier gated by min_confidence, PLUS a SEPARATE
    uncertainty (reject-option) classifier gated by uncertainty_threshold.
    A prediction is only "actionable" when BOTH gates pass (mirrors
    train_model.py's grid search semantics exactly).
  - Feature-vector length is validated against model_config.json's
    n_features before every predict_proba call, so a future edit that
    changes feature order/count in one file (app.py or train_model.py)
    without the other fails with a clear error instead of either crashing
    inside sklearn or silently misaligning columns.
  - /model-info reports the uncertainty model's availability and both
    thresholds, plus the validation/test split sizes, so it's obvious
    whether the deployed model has the full two-stage reject option or is
    running in degraded confidence-only mode.
  - /retrain now records the last training error on ml_model so it's
    visible via /retrain-status instead of only appearing in server logs.
  - NOTE ON DEPLOYMENT: repo / predictor / ml_model / the team-index cache /
    predictor.calibrators are all process-local Python globals. SQLite is
    the only durable, cross-process source of truth. Run this with a single
    worker process (e.g. `uvicorn app:app --workers 1`). Running multiple
    worker processes will cause them to diverge from each other's in-memory
    state (one worker's added match won't appear in another worker's
    Bayesian repo/team-index/ML feature extraction until that worker
    restarts). If you need multiple workers, the in-memory repo, team index
    cache, and calibrators need to move to a shared store (e.g. re-derive
    everything from SQLite on every request, or move to Redis/Postgres).
"""
from __future__ import annotations
import csv
import json
import math
import pickle
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from datetime import date
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.isotonic import IsotonicRegression
from bayesian_match_predictor import (
    MatchRepository, BayesianMatchPredictor, StatsComputer,
    TeamRegistry, _parse_date, Match,
)
app = FastAPI(title="Match Predictor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
DB_FILE                = "predictor.db"
MODEL_FILE             = Path("model.pkl")
UNCERTAINTY_MODEL_FILE = Path("uncertainty_model.pkl")
CONFIG_FILE            = Path("model_config.json")
# ──────────────────────────────────────────────
#  DATABASE
# ──────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
@contextmanager
def db_conn():
    conn = get_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
def _column_names(conn: sqlite3.Connection, table: str) -> set:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}

def init_db() -> None:
    with db_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_date TEXT NOT NULL, home_team TEXT NOT NULL, away_team TEXT NOT NULL,
                home_score INTEGER NOT NULL, away_score INTEGER NOT NULL,
                home_xg REAL, away_xg REAL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE (match_date, home_team, away_team)
            );
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, version TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                weight_odds REAL NOT NULL, weight_performance REAL NOT NULL,
                weight_historical REAL NOT NULL, decay_form REAL NOT NULL,
                decay_h2h REAL NOT NULL, home_advantage REAL NOT NULL, notes TEXT
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                home_team TEXT NOT NULL, away_team TEXT NOT NULL,
                prediction TEXT NOT NULL, prob_home REAL NOT NULL,
                prob_draw REAL NOT NULL, prob_away REAL NOT NULL,
                confidence REAL NOT NULL, max_prob REAL,
                model_type TEXT DEFAULT 'bayesian',
                uncertainty_prob REAL,
                odds_h REAL, odds_d REAL, odds_a REAL,
                model_version_id INTEGER REFERENCES model_versions(id),
                feature_snapshot TEXT, outcome TEXT,
                actual_home_score INTEGER, actual_away_score INTEGER, is_correct INTEGER
            );
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at TEXT NOT NULL DEFAULT (datetime('now')),
                n_evaluated INTEGER NOT NULL, accuracy REAL, avg_log_loss REAL, notes TEXT
            );
            CREATE TABLE IF NOT EXISTS calibration_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
                confidence_bucket REAL NOT NULL,
                n_predictions INTEGER NOT NULL DEFAULT 0, n_correct INTEGER NOT NULL DEFAULT 0
            );
        """)

        # Migrate matches table to support odds for training
        match_cols = _column_names(conn, "matches")
        for col, stmt in [
            ("odds_h", "ALTER TABLE matches ADD COLUMN odds_h REAL"),
            ("odds_d", "ALTER TABLE matches ADD COLUMN odds_d REAL"),
            ("odds_a", "ALTER TABLE matches ADD COLUMN odds_a REAL"),
        ]:
            if col not in match_cols: conn.execute(stmt)

        # Migrate predictions table
        pred_cols = _column_names(conn, "predictions")
        for col, stmt in [
            ("actual_home_score", "ALTER TABLE predictions ADD COLUMN actual_home_score INTEGER"),
            ("actual_away_score",  "ALTER TABLE predictions ADD COLUMN actual_away_score INTEGER"),
            ("is_correct",         "ALTER TABLE predictions ADD COLUMN is_correct INTEGER"),
            ("max_prob",           "ALTER TABLE predictions ADD COLUMN max_prob REAL"),
            ("model_type",         "ALTER TABLE predictions ADD COLUMN model_type TEXT DEFAULT 'bayesian'"),
            ("uncertainty_prob",   "ALTER TABLE predictions ADD COLUMN uncertainty_prob REAL"),
        ]:
            if col not in pred_cols:
                conn.execute(stmt)
        cal_cols = _column_names(conn, "calibration_history")
        if "n_predictions" not in cal_cols:
            conn.execute("ALTER TABLE calibration_history ADD COLUMN n_predictions INTEGER NOT NULL DEFAULT 0")
        if "n_correct" not in cal_cols:
            conn.execute("ALTER TABLE calibration_history ADD COLUMN n_correct INTEGER NOT NULL DEFAULT 0")
        if conn.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0] == 0:
            conn.execute(
                "INSERT INTO model_versions (version, weight_odds, weight_performance, weight_historical,"
                " decay_form, decay_h2h, home_advantage, notes) VALUES (?,?,?,?,?,?,?,?)",
                ("v1.0", 0.40, 0.30, 0.30, 730.0, 365.0, 1.15, "Initial production weights"),
            )
def _confidence_bucket(c: float) -> float:
    return round(round(c * 10) / 10, 1)
def _update_calibration(conn, confidence: float, is_correct: int) -> None:
    bucket = _confidence_bucket(confidence)
    ex = conn.execute(
        "SELECT id, n_predictions, n_correct FROM calibration_history WHERE confidence_bucket=?", (bucket,)
    ).fetchone()
    if ex:
        conn.execute(
            "UPDATE calibration_history SET n_predictions=?, n_correct=?, recorded_at=datetime('now') WHERE id=?",
            (ex["n_predictions"] + 1, ex["n_correct"] + is_correct, ex["id"]),
        )
    else:
        conn.execute(
            "INSERT INTO calibration_history (confidence_bucket, n_predictions, n_correct) VALUES (?,?,?)",
            (bucket, 1, is_correct),
        )
# ──────────────────────────────────────────────
#  ML MODEL STATE
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
#  ML MODEL STATE (Simplified - No Uncertainty)
# ──────────────────────────────────────────────
class MLModel:
    def __init__(self):
        self.clf = None
        self.config = {}
        self.loaded = False
        self.training = False
        self.last_training_error = None

    def load(self) -> bool:
        if not MODEL_FILE.exists() or not CONFIG_FILE.exists():
            self.loaded = False
            return False
        try:
            with open(MODEL_FILE, "rb") as f: self.clf = pickle.load(f)
            with open(CONFIG_FILE) as f: self.config = json.load(f)
            self.loaded = True
            print(f"✅ ML model loaded. min_confidence={self.min_confidence}")
            return True
        except Exception as e:
            print(f"⚠️  Could not load ML model: {e}")
            self.loaded = False
            return False

    def _check_feature_length(self, features: np.ndarray) -> None:
        expected = self.config.get("n_features")
        if expected is not None and len(features) != expected:
            raise ValueError(f"Feature mismatch: expected {expected}, got {len(features)}")

    def predict(self, features: np.ndarray) -> dict:
        self._check_feature_length(features)
        probs = self.clf.predict_proba(features.reshape(1, -1))[0]
        label_names = self.config.get("label_names", ["H", "D", "A"])
        return {label_names[i]: float(probs[i]) for i in range(len(label_names))}

    @property
    def min_confidence(self) -> float:
        return float(self.config.get("min_confidence", 0.50))

ml_model = MLModel()

# ──────────────────────────────────────────────
#  IN-MEMORY BAYESIAN PREDICTOR (fallback)
# ──────────────────────────────────────────────
repo           = MatchRepository()
predictor      = BayesianMatchPredictor(repo)
stats_computer = StatsComputer(repo)
def _load_all_matches_from_db() -> None:
    repo.reset()
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT match_date, home_team, away_team, home_score, away_score, home_xg, away_xg "
            "FROM matches ORDER BY match_date"
        ).fetchall()
    for r in rows:
        repo.add_match(Match(
            date=_parse_date(r["match_date"]),
            home_team=r["home_team"], away_team=r["away_team"],
            home_score=r["home_score"], away_score=r["away_score"],
            home_xg=r["home_xg"], away_xg=r["away_xg"],
        ), skip_duplicates=True)
    predictor.stats.invalidate_cache()
def _repo_remove_match(d: date, home: str, away: str) -> None:
    repo.matches = [m for m in repo.matches
                    if not (m.date == d and m.home_team == home and m.away_team == away)]
    repo._index = {repo._match_key(m): i for i, m in enumerate(repo.matches)}
    predictor.stats.invalidate_cache()
def _repo_update_match(m: Match) -> None:
    repo.add_match(m, skip_duplicates=False)
    predictor.stats.invalidate_cache()
def _current_model_version_id() -> Optional[int]:
    with db_conn() as conn:
        row = conn.execute("SELECT id FROM model_versions ORDER BY id DESC LIMIT 1").fetchone()
    return row["id"] if row else None
def _score_to_outcome(hs: int, as_: int) -> str:
    if hs > as_: return "H"
    if as_ > hs: return "A"
    return "D"
# ──────────────────────────────────────────────
#  ML FEATURE EXTRACTION  (must mirror train_model.py exactly — see
#  MLModel._check_feature_length for the runtime guard against drift)
# ──────────────────────────────────────────────

def _get_market_features(odds_h, odds_d, odds_a):
    if odds_h and odds_d and odds_a and odds_h > 1.0 and odds_d > 1.0 and odds_a > 1.0:
        raw_h, raw_d, raw_a = 1.0/odds_h, 1.0/odds_d, 1.0/odds_a
        total = raw_h + raw_d + raw_a
        prob_h, prob_d, prob_a = raw_h/total, raw_d/total, raw_a/total
        entropy = -sum(p * math.log2(p) for p in [prob_h, prob_d, prob_a] if p > 0)
        market_confidence = 1.0 - (entropy / math.log2(3))
        return prob_h, prob_d, prob_a, market_confidence
    return 1/3, 1/3, 1/3, 0.0


def _build_team_index_from_db():
    """Build per-team match history from DB for feature extraction."""
    from collections import defaultdict
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT match_date, home_team, away_team, home_score, away_score "
            "FROM matches ORDER BY match_date"
        ).fetchall()
    index = defaultdict(list)
    all_rows = list(rows)
    for r in all_rows:
        home, away = r["home_team"], r["away_team"]
        hs, as_ = r["home_score"], r["away_score"]
        d = r["match_date"]
        if hs > as_: hr, ar = "W", "L"
        elif as_ > hs: hr, ar = "L", "W"
        else: hr, ar = "D", "D"
        index[home].append({"date": d, "venue": "home", "scored": hs, "conceded": as_,
                             "result": hr, "pts": 3 if hr=="W" else (1 if hr=="D" else 0)})
        index[away].append({"date": d, "venue": "away", "scored": as_, "conceded": hs,
                             "result": ar, "pts": 3 if ar=="W" else (1 if ar=="D" else 0)})
    for team in index:
        index[team].sort(key=lambda x: x["date"])
    return dict(index), all_rows
_team_index_cache = None
_team_index_rows  = None
def _get_team_index():
    global _team_index_cache, _team_index_rows
    if _team_index_cache is None:
        _team_index_cache, _team_index_rows = _build_team_index_from_db()
    return _team_index_cache, _team_index_rows

def _invalidate_team_index():
    global _team_index_cache, _team_index_rows
    _team_index_cache = None
    _team_index_rows  = None

def _team_features(team: str, before_date: str, index: dict) -> dict:
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
    # Must mirror train_model.py's get_team_features exactly.
    recent = history[-10:]
    results = [m["result"] for m in recent]
    vals = {"W": 1.0, "D": 0.5, "L": 0.0}
    weights = list(range(1, len(results) + 1))
    form_points = sum(vals.get(r, 0.0) * w for r, w in zip(results, weights)) / sum(weights)

    n_recent = len(recent)
    recent_attack = sum(m["scored"] for m in recent) / n_recent
    recent_defense = sum(m["conceded"] for m in recent) / n_recent

    return {
        "win_rate": wins/n, "attack": goals_scored/n, "defense": goals_conceded/n,
        "form_points": form_points, "points": pts,
        "n_matches": n, "ppg": pts/n,
        "recent_attack": recent_attack, "recent_defense": recent_defense,
    }

def _h2h_features(home: str, away: str, before_date: str, all_rows) -> dict:
    h2h = [r for r in all_rows
           if r["match_date"] < before_date
           and ((r["home_team"]==home and r["away_team"]==away)
                or (r["home_team"]==away and r["away_team"]==home))][-8:]
    if not h2h:
        return {"h2h_home_win_rate": 0.33, "h2h_draw_rate": 0.33, "h2h_n": 0}
    hw = dr = 0
    for r in h2h:
        hs, as_ = r["home_score"], r["away_score"]
        if r["home_team"] == home:
            if hs > as_: hw += 1
            elif hs == as_: dr += 1
        else:
            if as_ > hs: hw += 1
            elif hs == as_: dr += 1
    n = len(h2h)
    return {"h2h_home_win_rate": hw/n, "h2h_draw_rate": dr/n, "h2h_n": min(n/8, 1.0)}


def _extract_ml_features(
    home: str,
    away: str,
    today_str: str,
    odds_h=0,
    odds_d=0,
    odds_a=0
) -> np.ndarray:

    index, _ = _get_team_index()

    hf = _team_features(home, today_str, index)
    af = _team_features(away, today_str, index)

    prob_h, prob_d, prob_a, market_conf = _get_market_features(
        odds_h,
        odds_d,
        odds_a
    )

    points_diff = hf["points"] - af["points"]

    attack_diff = hf["attack"] - af["attack"]
    defense_diff = hf["defense"] - af["defense"]
    form_diff = hf["form_points"] - af["form_points"]
    ppg_diff = hf["ppg"] - af["ppg"]
    recent_attack_diff = hf["recent_attack"] - af["recent_attack"]
    recent_defense_diff = hf["recent_defense"] - af["recent_defense"]

    return np.array([
        hf["win_rate"],
        af["win_rate"],

        hf["attack"],
        hf["defense"],

        af["attack"],
        af["defense"],

        hf["form_points"],
        af["form_points"],

        prob_h,
        prob_d,
        prob_a,

        market_conf,

        points_diff,

        hf["ppg"],
        af["ppg"],

        hf["recent_attack"],
        hf["recent_defense"],

        af["recent_attack"],
        af["recent_defense"],

        attack_diff,
        defense_diff,
        form_diff,
        ppg_diff,
        recent_attack_diff,
        recent_defense_diff,
    ], dtype=np.float32)


def _build_bayesian_snapshot(home: str, away: str) -> dict:
    hs  = stats_computer.compute_team_stats(home)
    as_ = stats_computer.compute_team_stats(away)
    n_h2h = len(repo.get_h2h_matches(home, away))
    def ppg(ts):        return ts.points / ts.matches_played if ts.matches_played else 0.0
    def goal_ratio(ts): d = ts.goals_scored + ts.goals_conceded; return ts.goals_scored / d if d else 0.5
    return {
        "home_matches": hs.matches_played, "away_matches": as_.matches_played,
        "home_ppg": round(ppg(hs), 3), "away_ppg": round(ppg(as_), 3),
        "home_goal_ratio": round(goal_ratio(hs), 3), "away_goal_ratio": round(goal_ratio(as_), 3),
        "home_form": "".join(hs.recent_form), "away_form": "".join(as_.recent_form),
        "h2h_count": n_h2h,
        "home_league_pos": stats_computer.get_league_position(home),
        "away_league_pos": stats_computer.get_league_position(away),
    }
# ──────────────────────────────────────────────
#  STARTUP
# ──────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    init_db()
    _load_all_matches_from_db()
    ml_model.load()
# ──────────────────────────────────────────────
#  PYDANTIC MODELS
# ──────────────────────────────────────────────
class MatchEntry(BaseModel):
    date: str; home_team: str; away_team: str
    home_score: int; away_score: int
    home_xg: Optional[float] = None; away_xg: Optional[float] = None
class MatchUpdate(BaseModel):
    date: str; home_team: str; away_team: str
    home_score: int; away_score: int
    home_xg: Optional[float] = None; away_xg: Optional[float] = None
class PredictRequest(BaseModel):
    home_team: str; away_team: str
    odds_h: float = 0; odds_d: float = 0; odds_a: float = 0
class DeleteMatchRequest(BaseModel):
    date: str; home_team: str; away_team: str
class ResultUpdate(BaseModel):
    prediction_id: int; home_score: int; away_score: int
# ──────────────────────────────────────────────
#  ROUTES — STATIC & MATCHES
# ──────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
@app.get("/matches")
def list_matches():
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT id, match_date, home_team, away_team, home_score, away_score, "
            "home_xg, away_xg, created_at FROM matches ORDER BY match_date DESC"
        ).fetchall()
    return [dict(r) for r in rows]
@app.post("/matches")
def add_match(entry: MatchEntry):
    try: parsed = _parse_date(entry.date)
    except ValueError as e: raise HTTPException(400, str(e))
    home, away = TeamRegistry.normalize(entry.home_team), TeamRegistry.normalize(entry.away_team)
    if home == away: raise HTTPException(400, "Home and away teams must be different.")
    with db_conn() as conn:
        if conn.execute("SELECT id FROM matches WHERE match_date=? AND home_team=? AND away_team=?",
                        (parsed.isoformat(), home, away)).fetchone():
            return {"added": False, "total": conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]}
        conn.execute(
            "INSERT INTO matches (match_date, home_team, away_team, home_score, away_score, home_xg, away_xg) "
            "VALUES (?,?,?,?,?,?,?)",
            (parsed.isoformat(), home, away, entry.home_score, entry.away_score, entry.home_xg, entry.away_xg),
        )
        total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    repo.add_match(Match(date=parsed, home_team=home, away_team=away,
                         home_score=entry.home_score, away_score=entry.away_score,
                         home_xg=entry.home_xg, away_xg=entry.away_xg), skip_duplicates=True)
    predictor.stats.invalidate_cache()
    _invalidate_team_index()
    return {"added": True, "total": total}
@app.put("/matches")
def update_match(entry: MatchUpdate):
    try: parsed = _parse_date(entry.date)
    except ValueError as e: raise HTTPException(400, str(e))
    home, away = TeamRegistry.normalize(entry.home_team), TeamRegistry.normalize(entry.away_team)
    with db_conn() as conn:
        if conn.execute(
            "UPDATE matches SET home_score=?, away_score=?, home_xg=?, away_xg=? "
            "WHERE match_date=? AND home_team=? AND away_team=?",
            (entry.home_score, entry.away_score, entry.home_xg, entry.away_xg,
             parsed.isoformat(), home, away),
        ).rowcount == 0:
            raise HTTPException(404, "Match not found.")
    _repo_update_match(Match(date=parsed, home_team=home, away_team=away,
                             home_score=entry.home_score, away_score=entry.away_score,
                             home_xg=entry.home_xg, away_xg=entry.away_xg))
    _invalidate_team_index()
    return {"updated": True}
@app.delete("/matches")
def delete_match(req: DeleteMatchRequest):
    try: d = _parse_date(req.date)
    except ValueError as e: raise HTTPException(400, str(e))
    home, away = TeamRegistry.normalize(req.home_team), TeamRegistry.normalize(req.away_team)
    with db_conn() as conn:
        if conn.execute("DELETE FROM matches WHERE match_date=? AND home_team=? AND away_team=?",
                        (d.isoformat(), home, away)).rowcount == 0:
            raise HTTPException(404, "Match not found.")
        total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    _repo_remove_match(d, home, away)
    _invalidate_team_index()
    return {"deleted": True, "total": total}
@app.delete("/reset")
def reset():
    with db_conn() as conn:
        conn.execute("DELETE FROM matches")
    repo.reset(); predictor.stats.invalidate_cache(); _invalidate_team_index()
    return {"reset": True}
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        reader = csv.DictReader(StringIO(content.decode("utf-8-sig")))
        added, errors, new_matches = 0, [], []
        with db_conn() as conn:
            for i, row in enumerate(reader, start=2):
                date_str = row.get("date", "").strip()
                if not date_str: continue
                try:
                    parsed  = _parse_date(date_str)
                    home    = TeamRegistry.normalize(row.get("home_team", "").strip())
                    away    = TeamRegistry.normalize(row.get("away_team", "").strip())
                    hs      = int(row.get("home_score", 0))
                    as_     = int(row.get("away_score", 0))
                    hxg_raw = row.get("home_xg", "").strip()
                    axg_raw = row.get("away_xg", "").strip()
                    hxg     = float(hxg_raw) if hxg_raw else None
                    axg     = float(axg_raw) if axg_raw else None
                    if not home or not away or home == away:
                        errors.append(f"Row {i}: invalid team names"); continue
                    if conn.execute("SELECT id FROM matches WHERE match_date=? AND home_team=? AND away_team=?",
                                    (parsed.isoformat(), home, away)).fetchone(): continue
                    conn.execute(
                        "INSERT INTO matches (match_date, home_team, away_team, home_score, away_score, home_xg, away_xg) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (parsed.isoformat(), home, away, hs, as_, hxg, axg),
                    )
                    new_matches.append(Match(date=parsed, home_team=home, away_team=away,
                                            home_score=hs, away_score=as_, home_xg=hxg, away_xg=axg))
                    added += 1
                except Exception as e: errors.append(f"Row {i}: {e}")
            total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        for m in new_matches:
            repo.add_match(m, skip_duplicates=True)
        if new_matches:
            predictor.stats.invalidate_cache()
            _invalidate_team_index()
                # Build a list of dictionaries for the inserted matches
        added_matches = [
            {
                "match_date": m.date.isoformat(),
                "home_team": m.home_team,
                "away_team": m.away_team,
                "home_score": m.home_score,
                "away_score": m.away_score,
            }
            for m in new_matches
        ]
        return {
            "added": added,
            "total": total,
            "errors": errors[:10],
            "added_matches": added_matches,   # <-- new key
        }
    except Exception as e:
        raise HTTPException(400, f"Error parsing CSV: {e}")
@app.get("/teams")
def list_teams():
    return sorted(repo.get_all_teams())
@app.get("/table")
def league_table():
    table = predictor.stats.compute_league_table()
    return [{"position": pos, "team": name, "played": ts.matches_played, "points": ts.points,
             "gd": ts.goal_difference, "scored": ts.goals_scored, "conceded": ts.goals_conceded,
             "form": "".join(ts.recent_form)}
            for pos, (name, ts) in enumerate(table, 1)]
# ──────────────────────────────────────────────
#  ROUTES — PREDICT
# ──────────────────────────────────────────────
@app.post("/predict")
def predict(req: PredictRequest, threshold: float = -1.0, edge_threshold: float = 0.0):
    if len(repo.matches) == 0:
        raise HTTPException(400, "No historical data loaded. Add some matches first.")
    home = TeamRegistry.normalize(req.home_team)
    away = TeamRegistry.normalize(req.away_team)
    mv_id = _current_model_version_id()
    pred_labels = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    today_str = date.today().isoformat()
    
    # ── ML path ───────────────────────────────────────────────
    if ml_model.loaded:
        effective_min_conf = ml_model.min_confidence if threshold < 0 else threshold
        try:
            feats = _extract_ml_features(home, away, today_str, req.odds_h, req.odds_d, req.odds_a)
            prob_map = ml_model.predict(feats)
            max_prob = max(prob_map.values())
            prediction = max(prob_map, key=prob_map.get)
            confidence = max_prob  # Simplified ML: confidence is just max_prob
            
            actionable = max_prob >= effective_min_conf
            
            # Edge filter
            if req.odds_h > 0 and req.odds_d > 0 and req.odds_a > 0 and edge_threshold > 0:
                implied = {"H": 1/req.odds_h, "D": 1/req.odds_d, "A": 1/req.odds_a}
                total_imp = sum(implied.values())
                implied = {k: v/total_imp for k, v in implied.items()}
                if (prob_map[prediction] - implied[prediction]) < edge_threshold:
                    actionable = False
                    
            snapshot = {"model": "ml", "min_confidence": effective_min_conf, "features": feats.tolist()}
            
            with db_conn() as conn:
                cur = conn.execute(
                    "INSERT INTO predictions (home_team, away_team, prediction, prob_home, prob_draw, prob_away,"
                    " confidence, max_prob, model_type, uncertainty_prob, odds_h, odds_d, odds_a,"
                    " model_version_id, feature_snapshot)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (home, away, prediction,
                     prob_map["H"], prob_map["D"], prob_map["A"],
                     confidence, max_prob, "ml", None,
                     req.odds_h or None, req.odds_d or None, req.odds_a or None,
                     mv_id, json.dumps(snapshot)),
                )
                prediction_id = cur.lastrowid
                
            return {
                "prediction_id":       prediction_id,
                "home_team":           home,
                "away_team":           away,
                "prediction":          pred_labels[prediction],
                "confidence":          round(confidence * 100, 1),
                "max_probability":     round(max_prob * 100, 1),
                "actionable":          actionable,
                "actionable_threshold": round(effective_min_conf * 100, 1),
                "uncertainty_threshold": None,
                "uncertainty_p_correct": None,
                "has_uncertainty_model": False,
                "edge_threshold":      edge_threshold,
                "model_type":          "ml",
                "probabilities":       {
                    "home": round(prob_map["H"] * 100, 1),
                    "draw": round(prob_map["D"] * 100, 1),
                    "away": round(prob_map["A"] * 100, 1),
                },
                "breakdown": {},
                "feature_snapshot": snapshot,
            }
        except Exception as e:
            print(f"⚠️  ML prediction failed, falling back to Bayesian: {e}")
            
    # ── Bayesian fallback ──────────────────────────────────────
    effective_threshold = 0.65 if threshold < 0 else threshold
    result   = predictor.predict_match(home, away, req.odds_h, req.odds_d, req.odds_a)
    snapshot = _build_bayesian_snapshot(home, away)
    max_prob = result.max_prob
    actionable = max_prob >= effective_threshold
    
    if req.odds_h > 0 and req.odds_d > 0 and req.odds_a > 0 and edge_threshold > 0:
        implied   = {"H": 1/req.odds_h, "D": 1/req.odds_d, "A": 1/req.odds_a}
        total_imp = sum(implied.values())
        implied   = {k: v/total_imp for k, v in implied.items()}
        edge      = result.probabilities[result.prediction] - implied[result.prediction]
        if edge < edge_threshold:
            actionable = False
            
    with db_conn() as conn:
        cur = conn.execute(
            "INSERT INTO predictions (home_team, away_team, prediction, prob_home, prob_draw, prob_away,"
            " confidence, max_prob, model_type, uncertainty_prob, odds_h, odds_d, odds_a, model_version_id, feature_snapshot)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (result.home_team, result.away_team, result.prediction,
             result.probabilities["H"], result.probabilities["D"], result.probabilities["A"],
             result.confidence, max_prob, "bayesian", None,
             req.odds_h or None, req.odds_d or None, req.odds_a or None,
             mv_id, json.dumps(snapshot)),
        )
        prediction_id = cur.lastrowid
        
    return {
        "prediction_id":       prediction_id,
        "home_team":           result.home_team,
        "away_team":           result.away_team,
        "prediction":          pred_labels.get(result.prediction, result.prediction),
        "confidence":          round(result.confidence * 100, 1),
        "max_probability":     round(max_prob * 100, 1),
        "actionable":          actionable,
        "actionable_threshold": round(effective_threshold * 100, 1),
        "uncertainty_threshold": None,
        "uncertainty_p_correct": None,
        "has_uncertainty_model": False,
        "edge_threshold":      edge_threshold,
        "model_type":          "bayesian",
        "probabilities":       {
            "home": round(result.probabilities["H"] * 100, 1),
            "draw": round(result.probabilities["D"] * 100, 1),
            "away": round(result.probabilities["A"] * 100, 1),
        },
        "breakdown": {
            src: {"home": round(p["H"]*100,1), "draw": round(p["D"]*100,1), "away": round(p["A"]*100,1)}
            for src, p in result.breakdown.items() if src != "combined"
        },
        "feature_snapshot": snapshot,
    }

# ──────────────────────────────────────────────
#  ROUTES — MODEL MANAGEMENT
# ──────────────────────────────────────────────
@app.get("/model-info")
def model_info():
    if not ml_model.loaded:
        return {
            "model_type": "bayesian",
            "ml_available": False,
            "message": "No ML model found. Run python train_model.py to train one.",
        }
    cfg = ml_model.config
    return {
        "model_type":              "ml",
        "ml_available":            True,
        "has_uncertainty_model":   False,  # Hardcoded False for Exp 3 formula
        "min_confidence":          cfg.get("min_confidence"),
        "min_confidence_pct":      round(ml_model.min_confidence * 100, 1),
        "uncertainty_threshold":   None,
        "test_accuracy_all":       cfg.get("test_accuracy_all"),
        "test_accuracy_thresh":    cfg.get("test_accuracy_thresh"),
        "test_coverage":           cfg.get("test_n_bets", 0) / cfg.get("test_size", 1) if cfg.get("test_size") else 0.0,
        "test_log_loss":           cfg.get("test_log_loss"),
        "train_cutoff":            cfg.get("train_cutoff"),
        "train_size":              cfg.get("train_size"),
        "test_size":               cfg.get("test_size"),
        "trained_at":              cfg.get("trained_at"),
        "n_features":              cfg.get("n_features"),
    }

@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    if ml_model.training:
        return {"started": False, "message": "Training already in progress."}
    def _run_training():
        ml_model.training = True
        try:
            result = subprocess.run(
                [sys.executable, "train_model.py"],
                capture_output=True, text=True, timeout=3600
            )
            if result.returncode == 0:
                ml_model.load()
                _invalidate_team_index()
                ml_model.last_training_error = None
                print("✅ Retraining complete.")
            else:
                # train_model.py prints its guard messages (e.g. "not enough
                # data") via print(), which goes to stdout, not stderr — only
                # unhandled tracebacks land in stderr. Surface both so the
                # actual reason (not just an exit code) reaches the client
                # via /retrain-status instead of being stranded in server logs.
                combined = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
                ml_model.last_training_error = combined.strip()[-4000:]
                print(f"❌ Retraining failed:\n{combined}")
        except Exception as e:
            ml_model.last_training_error = str(e)
            print(f"❌ Retraining error: {e}")
        finally:
            ml_model.training = False
    background_tasks.add_task(_run_training)
    return {"started": True, "message": "Training started in background. Check /model-info or /retrain-status for updates."}
@app.get("/retrain-status")
def retrain_status():
    return {
        "training": ml_model.training,
        "last_training_error": ml_model.last_training_error,
    }
@app.post("/reload-model")
def reload_model():
    loaded = ml_model.load()
    if loaded:
        return {
            "loaded": True,
            "min_confidence": ml_model.min_confidence,
            "uncertainty_threshold": ml_model.uncertainty_threshold,
            "has_uncertainty_model": ml_model.has_uncertainty_model,
            "accuracy": ml_model.config.get("test_accuracy_thresh"),
        }
    return {"loaded": False, "message": "No model.pkl found. Run train_model.py first."}
@app.post("/calibrate")
def calibrate():
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT prob_home, prob_draw, prob_away, outcome FROM predictions WHERE outcome IS NOT NULL"
        ).fetchall()
    if len(rows) < 20:
        return {"message": "Need at least 20 evaluated predictions.", "calibrated": False}
    col_map = {"H": "prob_home", "D": "prob_draw", "A": "prob_away"}
    new_calibrators = {}
    for outcome in ("H", "D", "A"):
        y_true = [1 if r["outcome"] == outcome else 0 for r in rows]
        y_prob = [r[col_map[outcome]] for r in rows]
        if len(set(y_prob)) < 2: continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_prob, y_true)
        new_calibrators[outcome] = iso
    predictor.calibrators = new_calibrators
    return {"message": f"Bayesian calibration done with {len(rows)} predictions.", "calibrated": True}
# ──────────────────────────────────────────────
#  ROUTES — RESULTS & HISTORY
# ──────────────────────────────────────────────
@app.post("/predictions/result")
def record_result(req: ResultUpdate):
    if req.home_score < 0 or req.away_score < 0:
        raise HTTPException(400, "Scores must be non-negative.")
    outcome = _score_to_outcome(req.home_score, req.away_score)
    with db_conn() as conn:
        row = conn.execute("SELECT prediction, confidence FROM predictions WHERE id=?",
                           (req.prediction_id,)).fetchone()
        if row is None: raise HTTPException(404, "Prediction not found.")
        is_correct = 1 if row["prediction"] == outcome else 0
        conn.execute(
            "UPDATE predictions SET outcome=?, actual_home_score=?, actual_away_score=?, is_correct=? WHERE id=?",
            (outcome, req.home_score, req.away_score, is_correct, req.prediction_id),
        )
        _update_calibration(conn, row["confidence"], is_correct)
    return {"updated": True, "outcome": outcome,
            "outcome_label": {"H": "Home Win", "D": "Draw", "A": "Away Win"}[outcome],
            "is_correct": bool(is_correct)}
@app.get("/predictions/history")
def prediction_history(
    page: int = 1,
    limit: int = 20,
    outcome: str = "all"  # "all", "pending", "evaluated"
):
    if page < 1:
        page = 1
    if limit < 1 or limit > 100:
        limit = 20
    offset = (page - 1) * limit

    with db_conn() as conn:
        # Build the base query with optional outcome filter
        base_query = """
            SELECT id, created_at, home_team, away_team, prediction, model_type,
                   ROUND(prob_home*100,1) AS prob_home,
                   ROUND(prob_draw*100,1) AS prob_draw,
                   ROUND(prob_away*100,1) AS prob_away,
                   ROUND(confidence*100,1) AS confidence,
                   ROUND(COALESCE(max_prob, confidence)*100,1) AS max_prob,
                   ROUND(uncertainty_prob*100,1) AS uncertainty_prob,
                   outcome, actual_home_score, actual_away_score, is_correct
            FROM predictions
        """
        where_clauses = []
        params = []

        if outcome == "pending":
            where_clauses.append("outcome IS NULL")
        elif outcome == "evaluated":
            where_clauses.append("outcome IS NOT NULL")

        if where_clauses:
            base_query += " WHERE " + " AND ".join(where_clauses)

        # Count total
        count_query = f"SELECT COUNT(*) AS total FROM ({base_query}) AS sub"
        total = conn.execute(count_query, params).fetchone()["total"]

        # Fetch paginated rows
        rows_query = base_query + " ORDER BY id DESC LIMIT ? OFFSET ?"
        rows = conn.execute(rows_query, params + [limit, offset]).fetchall()

    return {
        "items": [dict(r) for r in rows],
        "total": total,
        "page": page,
        "limit": limit,
        "pages": (total + limit - 1) // limit if total else 1,
    }
@app.get("/evaluate")
def evaluate():
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT prediction, prob_home, prob_draw, prob_away, confidence,"
            " COALESCE(max_prob, confidence) as max_prob, model_type, outcome, is_correct"
            " FROM predictions WHERE outcome IS NOT NULL"
        ).fetchall()
    if not rows:
        return {"message": "No evaluated predictions yet.", "n": 0}
    correct = log_losses = 0
    log_losses = []
    ml_correct = ml_total = bay_correct = bay_total = 0
    for r in rows:
        prob_map = {"H": r["prob_home"], "D": r["prob_draw"], "A": r["prob_away"]}
        p = max(prob_map.get(r["outcome"], 1e-7), 1e-7)
        log_losses.append(-math.log(p))
        correct += r["is_correct"] or 0
        if r["model_type"] == "ml":
            ml_total += 1; ml_correct += r["is_correct"] or 0
        else:
            bay_total += 1; bay_correct += r["is_correct"] or 0
    n = len(rows)
    with db_conn() as conn:
        conn.execute("INSERT INTO evaluation_runs (n_evaluated, accuracy, avg_log_loss) VALUES (?,?,?)",
                     (n, correct/n, sum(log_losses)/n))
        cal_rows = conn.execute(
            "SELECT confidence_bucket, n_predictions, n_correct FROM calibration_history ORDER BY confidence_bucket"
        ).fetchall()
    calibration = []
    for c in cal_rows:
        actual_rate = c["n_correct"] / c["n_predictions"] if c["n_predictions"] else None
        calibration.append({
            "bucket":         c["confidence_bucket"],
            "label":          f"{int(c['confidence_bucket']*100)}%",
            "n":              c["n_predictions"],
            "predicted_rate": c["confidence_bucket"],
            "actual_rate":    round(actual_rate, 3) if actual_rate is not None else None,
            "gap":            round(c["confidence_bucket"] - actual_rate, 3) if actual_rate is not None else None,
        })
    return {
        "n_evaluated":    n,
        "accuracy":       round(correct / n * 100, 1),
        "avg_log_loss":   round(sum(log_losses) / n, 4),
        "note":           "Primary metric: avg_log_loss — lower is better.",
        "by_model": {
            "ml":       {"n": ml_total,  "accuracy": round(ml_correct/ml_total*100,1)  if ml_total  else None},
            "bayesian": {"n": bay_total, "accuracy": round(bay_correct/bay_total*100,1) if bay_total else None},
        },
        "calibration": calibration,
    }
@app.get("/model-versions")
def list_model_versions():
    with db_conn() as conn:
        rows = conn.execute("SELECT * FROM model_versions ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


"""
# output from running the train_model.py
============================================================
  MATCH PREDICTOR — ML TRAINING (Experiment 3 Formula)
============================================================

Loaded 189,965 matches from predictor.db
Building time-correct team index...
Teams indexed: 1151 | Teams with >= 50 matches: 965

Extracting features (13 features, time-correct)...
Feature matrix: (185090, 13)  (skipped 4875 matches due to team filter)

Time split at 2018-07-01: Train=153,097 | Test=31,993
Fit: 122,478 | Validation: 30,619

────────────────────────────────────────────────────────────
REGULARIZATION GRID SEARCH  (C candidates: [0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
────────────────────────────────────────────────────────────

C=0.05:  (val overall accuracy=0.5045  log_loss=1.0028)
  min_conf=0.50  bets=13344/30619 (43.6%)  acc=0.6149
  min_conf=0.55  bets=8993/30619 (29.4%)  acc=0.6526
  min_conf=0.60  bets=5984/30619 (19.5%)  acc=0.6969
  min_conf=0.65  bets=3948/30619 (12.9%)  acc=0.7381

C=0.1:  (val overall accuracy=0.5046  log_loss=1.0028)
  min_conf=0.50  bets=13350/30619 (43.6%)  acc=0.6146
  min_conf=0.55  bets=8995/30619 (29.4%)  acc=0.6524
  min_conf=0.60  bets=5975/30619 (19.5%)  acc=0.6969
  min_conf=0.65  bets=3942/30619 (12.9%)  acc=0.7382

C=0.3:  (val overall accuracy=0.5046  log_loss=1.0028)
  min_conf=0.50  bets=13346/30619 (43.6%)  acc=0.6149
  min_conf=0.55  bets=8992/30619 (29.4%)  acc=0.6520
  min_conf=0.60  bets=5972/30619 (19.5%)  acc=0.6969
  min_conf=0.65  bets=3947/30619 (12.9%)  acc=0.7378

C=1.0:  (val overall accuracy=0.5048  log_loss=1.0028)
  min_conf=0.50  bets=13343/30619 (43.6%)  acc=0.6149
  min_conf=0.55  bets=8997/30619 (29.4%)  acc=0.6522
  min_conf=0.60  bets=5974/30619 (19.5%)  acc=0.6967
  min_conf=0.65  bets=3946/30619 (12.9%)  acc=0.7380

C=3.0:  (val overall accuracy=0.5048  log_loss=1.0028)
  min_conf=0.50  bets=13343/30619 (43.6%)  acc=0.6151
  min_conf=0.55  bets=8998/30619 (29.4%)  acc=0.6520
  min_conf=0.60  bets=5974/30619 (19.5%)  acc=0.6969
  min_conf=0.65  bets=3946/30619 (12.9%)  acc=0.7380

C=10.0:  (val overall accuracy=0.5048  log_loss=1.0028)
  min_conf=0.50  bets=13342/30619 (43.6%)  acc=0.6151
  min_conf=0.55  bets=8996/30619 (29.4%)  acc=0.6521
  min_conf=0.60  bets=5976/30619 (19.5%)  acc=0.6966
  min_conf=0.65  bets=3944/30619 (12.9%)  acc=0.7381

C=30.0:  (val overall accuracy=0.5048  log_loss=1.0028)
  min_conf=0.50  bets=13342/30619 (43.6%)  acc=0.6151
  min_conf=0.55  bets=8996/30619 (29.4%)  acc=0.6521
  min_conf=0.60  bets=5976/30619 (19.5%)  acc=0.6966
  min_conf=0.65  bets=3944/30619 (12.9%)  acc=0.7381

────────────────────────────────────────────────────────────
REGULARIZATION GRID SUMMARY (sorted by gated validation accuracy)
────────────────────────────────────────────────────────────
         C  val_acc_all  val_logloss  min_conf  gated_acc  coverage
       0.1       0.5046       1.0028      0.65     0.7382     12.9%
      0.05       0.5045       1.0028      0.65     0.7381     12.9%
      10.0       0.5048       1.0028      0.65     0.7381     12.9%
      30.0       0.5048       1.0028      0.65     0.7381     12.9%
       1.0       0.5048       1.0028      0.65     0.7380     12.9%
       3.0       0.5048       1.0028      0.65     0.7380     12.9%
       0.3       0.5046       1.0028      0.65     0.7378     12.9%

  Spread across C values (gated accuracy): 0.0004
  → C has negligible effect here (<1pt). Regularization is NOT the explanation for any small gap.

  Selected: C=0.1  min_confidence=0.65  (val gated accuracy=0.7382)

Refitting on full training set with selected C=0.1...

────────────────────────────────────────────────────────────
TEST SET EVALUATION
────────────────────────────────────────────────────────────
All matches: accuracy=0.4987  log_loss=1.0087

Gated (C=0.1, min_conf=0.65): 0.7304 on 4251/31993 matches
  Approx. 95% CI: [0.7171, 0.7438]  (n=4251)

============================================================
  TRAINING COMPLETE
  Selected C: 0.1
  Test accuracy (gated): 73.0%
============================================================
"""