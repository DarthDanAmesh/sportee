"""
FastAPI backend for Bayesian Match Predictor
SQLite persistence — v3
Changes from v2:
  - predictions table gains: actual_home_score, actual_away_score, is_correct
  - OLD /predictions/outcome (manual H/D/A) replaced by /predictions/result (scores → auto-derive)
  - calibration_history is now written automatically on every result entry
  - /evaluate returns per-bucket calibration summary in addition to overall metrics
  - migration guard: ALTER TABLE IF NOT EXISTS equivalent via column introspection
"""
from __future__ import annotations

import csv
import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import date
from io import StringIO
from typing import List, Optional

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from bayesian_match_predictor import (
    MatchRepository,
    BayesianMatchPredictor,
    StatsComputer,
    TeamRegistry,
    _parse_date,
    Match,
)

app = FastAPI(title="Bayesian Match Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
#  DATABASE
# ──────────────────────────────────────────────

DB_FILE = "predictor.db"


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
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r["name"] for r in rows}


def init_db() -> None:
    with db_conn() as conn:
        # ── Core tables ──────────────────────────────────────────────────────
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS matches (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                match_date  TEXT    NOT NULL,
                home_team   TEXT    NOT NULL,
                away_team   TEXT    NOT NULL,
                home_score  INTEGER NOT NULL,
                away_score  INTEGER NOT NULL,
                home_xg     REAL,
                away_xg     REAL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                UNIQUE (match_date, home_team, away_team)
            );

            CREATE TABLE IF NOT EXISTS model_versions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                version             TEXT    NOT NULL,
                created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
                weight_odds         REAL    NOT NULL,
                weight_performance  REAL    NOT NULL,
                weight_historical   REAL    NOT NULL,
                decay_form          REAL    NOT NULL,
                decay_h2h           REAL    NOT NULL,
                home_advantage      REAL    NOT NULL,
                notes               TEXT
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
                home_team           TEXT    NOT NULL,
                away_team           TEXT    NOT NULL,
                prediction          TEXT    NOT NULL,
                prob_home           REAL    NOT NULL,
                prob_draw           REAL    NOT NULL,
                prob_away           REAL    NOT NULL,
                confidence          REAL    NOT NULL,
                odds_h              REAL,
                odds_d              REAL,
                odds_a              REAL,
                model_version_id    INTEGER REFERENCES model_versions(id),
                feature_snapshot    TEXT,
                outcome             TEXT,
                actual_home_score   INTEGER,
                actual_away_score   INTEGER,
                is_correct          INTEGER
            );

            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at       TEXT    NOT NULL DEFAULT (datetime('now')),
                n_evaluated  INTEGER NOT NULL,
                accuracy     REAL,
                avg_log_loss REAL,
                notes        TEXT
            );

            CREATE TABLE IF NOT EXISTS calibration_history (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at          TEXT    NOT NULL DEFAULT (datetime('now')),
                confidence_bucket    REAL    NOT NULL,
                n_predictions        INTEGER NOT NULL DEFAULT 0,
                n_correct            INTEGER NOT NULL DEFAULT 0
            );
        """)

        # ── Migration guard: add new columns to existing predictions tables ──
        pred_cols = _column_names(conn, "predictions")
        migrations = [
            ("actual_home_score", "ALTER TABLE predictions ADD COLUMN actual_home_score INTEGER"),
            ("actual_away_score",  "ALTER TABLE predictions ADD COLUMN actual_away_score INTEGER"),
            ("is_correct",         "ALTER TABLE predictions ADD COLUMN is_correct INTEGER"),
        ]
        for col, stmt in migrations:
            if col not in pred_cols:
                conn.execute(stmt)

        # ── Migration guard for calibration_history ──
        cal_cols = _column_names(conn, "calibration_history")
        if "n_predictions" not in cal_cols:
            conn.execute("ALTER TABLE calibration_history ADD COLUMN n_predictions INTEGER NOT NULL DEFAULT 0")
        if "n_correct" not in cal_cols:
            conn.execute("ALTER TABLE calibration_history ADD COLUMN n_correct INTEGER NOT NULL DEFAULT 0")

        # ── Seed initial model version ────────────────────────────────────────
        if conn.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0] == 0:
            conn.execute("""
                INSERT INTO model_versions
                    (version, weight_odds, weight_performance, weight_historical,
                     decay_form, decay_h2h, home_advantage, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, ("v1.0", 0.40, 0.30, 0.30, 730.0, 365.0, 1.15,
                  "Initial production weights"))


# ──────────────────────────────────────────────
#  CALIBRATION HELPER
# ──────────────────────────────────────────────

def _confidence_bucket(confidence: float) -> float:
    """Round confidence to nearest 0.1 bucket (0.1 … 1.0)."""
    return round(round(confidence * 10) / 10, 1)


def _update_calibration(conn: sqlite3.Connection, confidence: float, is_correct: int) -> None:
    """
    Upsert into calibration_history.
    One row per bucket — n_predictions and n_correct are running totals.
    """
    bucket = _confidence_bucket(confidence)
    existing = conn.execute(
        "SELECT id, n_predictions, n_correct FROM calibration_history WHERE confidence_bucket = ?",
        (bucket,)
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE calibration_history SET n_predictions=?, n_correct=?, recorded_at=datetime('now') WHERE id=?",
            (existing["n_predictions"] + 1, existing["n_correct"] + is_correct, existing["id"])
        )
    else:
        conn.execute(
            "INSERT INTO calibration_history (confidence_bucket, n_predictions, n_correct) VALUES (?, ?, ?)",
            (bucket, 1, is_correct)
        )


# ──────────────────────────────────────────────
#  IN-MEMORY PREDICTOR
# ──────────────────────────────────────────────

repo           = MatchRepository()
predictor      = BayesianMatchPredictor(repo)
stats_computer = StatsComputer(repo)


def _load_all_matches_from_db() -> None:
    repo.reset()
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT match_date, home_team, away_team, home_score, away_score, "
            "home_xg, away_xg FROM matches ORDER BY match_date"
        ).fetchall()
    for r in rows:
        repo.add_match(Match(
            date=_parse_date(r["match_date"]),
            home_team=r["home_team"],
            away_team=r["away_team"],
            home_score=r["home_score"],
            away_score=r["away_score"],
            home_xg=r["home_xg"],
            away_xg=r["away_xg"],
        ), skip_duplicates=True)
    predictor.stats.invalidate_cache()


def _current_model_version_id() -> Optional[int]:
    with db_conn() as conn:
        row = conn.execute(
            "SELECT id FROM model_versions ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return row["id"] if row else None


def _build_feature_snapshot(home: str, away: str) -> dict:
    hs  = stats_computer.compute_team_stats(home)
    as_ = stats_computer.compute_team_stats(away)
    n_h2h = len(repo.get_h2h_matches(home, away))

    def ppg(ts):
        return ts.points / ts.matches_played if ts.matches_played else 0.0

    def goal_ratio(ts):
        d = ts.goals_scored + ts.goals_conceded
        return ts.goals_scored / d if d else 0.5

    return {
        "home_matches":    hs.matches_played,
        "away_matches":    as_.matches_played,
        "home_ppg":        round(ppg(hs), 3),
        "away_ppg":        round(ppg(as_), 3),
        "home_goal_ratio": round(goal_ratio(hs), 3),
        "away_goal_ratio": round(goal_ratio(as_), 3),
        "home_form":       "".join(hs.recent_form),
        "away_form":       "".join(as_.recent_form),
        "h2h_count":       n_h2h,
        "home_league_pos": stats_computer.get_league_position(home),
        "away_league_pos": stats_computer.get_league_position(away),
    }


def _score_to_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    elif away_score > home_score:
        return "A"
    return "D"


# ──────────────────────────────────────────────
#  STARTUP
# ──────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    init_db()
    _load_all_matches_from_db()


# ──────────────────────────────────────────────
#  PYDANTIC MODELS
# ──────────────────────────────────────────────

class MatchEntry(BaseModel):
    date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None


class MatchUpdate(BaseModel):
    date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    odds_h: float = 0
    odds_d: float = 0
    odds_a: float = 0


class DeleteMatchRequest(BaseModel):
    date: str
    home_team: str
    away_team: str


class ResultUpdate(BaseModel):
    """
    Record the actual match result by score.
    The backend derives outcome (H/D/A) automatically.
    Replaces the old OutcomeUpdate model.
    """
    prediction_id: int
    home_score: int
    away_score: int


# ──────────────────────────────────────────────
#  ROUTES — STATIC
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


# ──────────────────────────────────────────────
#  ROUTES — MATCHES (CRUD)
# ──────────────────────────────────────────────

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
    try:
        parsed = _parse_date(entry.date)
    except ValueError as e:
        raise HTTPException(400, str(e))

    home = TeamRegistry.normalize(entry.home_team)
    away = TeamRegistry.normalize(entry.away_team)
    if home == away:
        raise HTTPException(400, "Home and away teams must be different.")

    with db_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM matches WHERE match_date=? AND home_team=? AND away_team=?",
            (parsed.isoformat(), home, away)
        ).fetchone()
        if existing:
            total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            return {"added": False, "total": total}

        conn.execute(
            "INSERT INTO matches (match_date, home_team, away_team, home_score, away_score, home_xg, away_xg) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (parsed.isoformat(), home, away, entry.home_score, entry.away_score,
             entry.home_xg, entry.away_xg)
        )
        total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]

    repo.add_match(Match(
        date=parsed, home_team=home, away_team=away,
        home_score=entry.home_score, away_score=entry.away_score,
        home_xg=entry.home_xg, away_xg=entry.away_xg,
    ), skip_duplicates=True)
    predictor.stats.invalidate_cache()
    return {"added": True, "total": total}


@app.put("/matches")
def update_match(entry: MatchUpdate):
    try:
        parsed = _parse_date(entry.date)
    except ValueError as e:
        raise HTTPException(400, str(e))

    home = TeamRegistry.normalize(entry.home_team)
    away = TeamRegistry.normalize(entry.away_team)

    with db_conn() as conn:
        result = conn.execute(
            "UPDATE matches SET home_score=?, away_score=?, home_xg=?, away_xg=? "
            "WHERE match_date=? AND home_team=? AND away_team=?",
            (entry.home_score, entry.away_score, entry.home_xg, entry.away_xg,
             parsed.isoformat(), home, away)
        )
        if result.rowcount == 0:
            raise HTTPException(404, "Match not found")

    _load_all_matches_from_db()
    return {"updated": True}


@app.delete("/matches")
def delete_match(req: DeleteMatchRequest):
    try:
        d = _parse_date(req.date)
    except ValueError as e:
        raise HTTPException(400, str(e))

    home = TeamRegistry.normalize(req.home_team)
    away = TeamRegistry.normalize(req.away_team)

    with db_conn() as conn:
        result = conn.execute(
            "DELETE FROM matches WHERE match_date=? AND home_team=? AND away_team=?",
            (d.isoformat(), home, away)
        )
        if result.rowcount == 0:
            raise HTTPException(404, "Match not found")
        total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]

    _load_all_matches_from_db()
    return {"deleted": True, "total": total}


@app.delete("/reset")
def reset():
    with db_conn() as conn:
        conn.execute("DELETE FROM matches")
    repo.reset()
    predictor.stats.invalidate_cache()
    return {"reset": True}


# ──────────────────────────────────────────────
#  ROUTES — CSV UPLOAD
# ──────────────────────────────────────────────

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        reader  = csv.DictReader(StringIO(content.decode("utf-8-sig")))
        added, errors = 0, []

        with db_conn() as conn:
            for i, row in enumerate(reader, start=2):
                date_str = row.get("date", "").strip()
                if not date_str:
                    continue
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
                        errors.append(f"Row {i}: invalid team names")
                        continue

                    if conn.execute(
                        "SELECT id FROM matches WHERE match_date=? AND home_team=? AND away_team=?",
                        (parsed.isoformat(), home, away)
                    ).fetchone():
                        continue

                    conn.execute(
                        "INSERT INTO matches (match_date, home_team, away_team, "
                        "home_score, away_score, home_xg, away_xg) VALUES (?,?,?,?,?,?,?)",
                        (parsed.isoformat(), home, away, hs, as_, hxg, axg)
                    )
                    added += 1
                except Exception as row_err:
                    errors.append(f"Row {i}: {row_err}")

        _load_all_matches_from_db()

        with db_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]

        return {"added": added, "total": total, "errors": errors[:10]}
    except Exception as e:
        raise HTTPException(400, f"Error parsing CSV: {str(e)}")


# ──────────────────────────────────────────────
#  ROUTES — TEAMS & TABLE
# ──────────────────────────────────────────────

@app.get("/teams")
def list_teams():
    return sorted(repo.get_all_teams())


@app.get("/table")
def league_table():
    table = predictor.stats.compute_league_table()
    return [
        {
            "position": pos,
            "team":     name,
            "played":   ts.matches_played,
            "points":   ts.points,
            "gd":       ts.goal_difference,
            "scored":   ts.goals_scored,
            "conceded": ts.goals_conceded,
            "form":     "".join(ts.recent_form),
        }
        for pos, (name, ts) in enumerate(table, 1)
    ]


# ──────────────────────────────────────────────
#  ROUTES — PREDICT
# ──────────────────────────────────────────────

@app.post("/predict")
def predict(req: PredictRequest):
    if len(repo.matches) == 0:
        raise HTTPException(400, "No historical data loaded. Add some matches first.")

    home = TeamRegistry.normalize(req.home_team)
    away = TeamRegistry.normalize(req.away_team)

    result   = predictor.predict_match(home, away, req.odds_h, req.odds_d, req.odds_a)
    snapshot = _build_feature_snapshot(home, away)
    mv_id    = _current_model_version_id()

    with db_conn() as conn:
        cur = conn.execute("""
            INSERT INTO predictions
                (home_team, away_team, prediction, prob_home, prob_draw, prob_away,
                 confidence, odds_h, odds_d, odds_a, model_version_id, feature_snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.home_team, result.away_team, result.prediction,
            result.probabilities["H"], result.probabilities["D"], result.probabilities["A"],
            result.confidence,
            req.odds_h if req.odds_h > 0 else None,
            req.odds_d if req.odds_d > 0 else None,
            req.odds_a if req.odds_a > 0 else None,
            mv_id, json.dumps(snapshot),
        ))
        prediction_id = cur.lastrowid

    pred_labels = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    return {
        "prediction_id": prediction_id,
        "home_team":     result.home_team,
        "away_team":     result.away_team,
        "prediction":    pred_labels.get(result.prediction, result.prediction),
        "confidence":    round(result.confidence * 100, 1),
        "probabilities": {
            "home": round(result.probabilities["H"] * 100, 1),
            "draw": round(result.probabilities["D"] * 100, 1),
            "away": round(result.probabilities["A"] * 100, 1),
        },
        "breakdown": {
            src: {
                "home": round(p["H"] * 100, 1),
                "draw": round(p["D"] * 100, 1),
                "away": round(p["A"] * 100, 1),
            }
            for src, p in result.breakdown.items()
            if src != "combined"
        },
        "feature_snapshot": snapshot,
    }


# ──────────────────────────────────────────────
#  ROUTES — RECORD RESULT
# ──────────────────────────────────────────────

@app.post("/predictions/result")
def record_result(req: ResultUpdate):
    """
    Record the actual score for a stored prediction.
    Outcome (H/D/A), is_correct, and calibration are all derived automatically.
    """
    if req.home_score < 0 or req.away_score < 0:
        raise HTTPException(400, "Scores must be non-negative.")

    outcome = _score_to_outcome(req.home_score, req.away_score)

    with db_conn() as conn:
        row = conn.execute(
            "SELECT prediction, confidence FROM predictions WHERE id=?",
            (req.prediction_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(404, "Prediction not found.")

        is_correct = 1 if row["prediction"] == outcome else 0

        conn.execute("""
            UPDATE predictions
            SET outcome           = ?,
                actual_home_score = ?,
                actual_away_score = ?,
                is_correct        = ?
            WHERE id = ?
        """, (outcome, req.home_score, req.away_score, is_correct, req.prediction_id))

        # Update running calibration totals
        _update_calibration(conn, row["confidence"], is_correct)

    outcome_labels = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    return {
        "updated":    True,
        "outcome":    outcome,
        "outcome_label": outcome_labels[outcome],
        "is_correct": bool(is_correct),
    }


# ──────────────────────────────────────────────
#  ROUTES — PREDICTIONS HISTORY
# ──────────────────────────────────────────────

@app.get("/predictions/history")
def prediction_history(limit: int = 50):
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT id, created_at, home_team, away_team, prediction,
                   ROUND(prob_home*100,1)  AS prob_home,
                   ROUND(prob_draw*100,1)  AS prob_draw,
                   ROUND(prob_away*100,1)  AS prob_away,
                   ROUND(confidence*100,1) AS confidence,
                   outcome, actual_home_score, actual_away_score, is_correct
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


# ──────────────────────────────────────────────
#  ROUTES — EVALUATE
# ──────────────────────────────────────────────

@app.get("/evaluate")
def evaluate():
    """
    Overall accuracy + log-loss, plus per-bucket calibration data.
    Primary metric: avg_log_loss (lower = better).
    """
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT prediction, prob_home, prob_draw, prob_away,
                   confidence, outcome, is_correct
            FROM predictions
            WHERE outcome IS NOT NULL
        """).fetchall()

    if not rows:
        return {"message": "No evaluated predictions yet.", "n": 0}

    correct, log_losses = 0, []
    for r in rows:
        prob_map = {"H": r["prob_home"], "D": r["prob_draw"], "A": r["prob_away"]}
        p = max(prob_map.get(r["outcome"], 1e-7), 1e-7)
        log_losses.append(-math.log(p))
        correct += r["is_correct"] or 0

    n            = len(rows)
    avg_log_loss = sum(log_losses) / n
    accuracy     = correct / n

    # Store snapshot of this evaluation run
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO evaluation_runs (n_evaluated, accuracy, avg_log_loss) VALUES (?,?,?)",
            (n, accuracy, avg_log_loss)
        )

        # Calibration summary from running totals
        cal_rows = conn.execute(
            "SELECT confidence_bucket, n_predictions, n_correct "
            "FROM calibration_history ORDER BY confidence_bucket"
        ).fetchall()

    calibration = []
    for c in cal_rows:
        actual_rate = c["n_correct"] / c["n_predictions"] if c["n_predictions"] else None
        calibration.append({
            "bucket":          c["confidence_bucket"],
            "label":           f"{int(c['confidence_bucket']*100)}%",
            "n":               c["n_predictions"],
            "predicted_rate":  c["confidence_bucket"],
            "actual_rate":     round(actual_rate, 3) if actual_rate is not None else None,
            "gap":             round(c["confidence_bucket"] - actual_rate, 3) if actual_rate is not None else None,
        })

    return {
        "n_evaluated":  n,
        "accuracy":     round(accuracy * 100, 1),
        "avg_log_loss": round(avg_log_loss, 4),
        "note":         "Primary metric: avg_log_loss — lower is better. Accuracy is secondary.",
        "calibration":  calibration,
    }


# ──────────────────────────────────────────────
#  ROUTES — MODEL VERSIONS
# ──────────────────────────────────────────────

@app.get("/model-versions")
def list_model_versions():
    with db_conn() as conn:
        rows = conn.execute("SELECT * FROM model_versions ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]
