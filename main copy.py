"""
FastAPI backend for Bayesian Match Predictor
"""
from __future__ import annotations

import csv
import os
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
from datetime import date

app = FastAPI(title="Bayesian Match Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── in-memory store (lives for the process lifetime) ──
repo = MatchRepository()
predictor = BayesianMatchPredictor(repo)
stats_computer = StatsComputer(repo)

# ── persistence ──
CSV_FILE = "historical_matches.csv"

def save_matches():
    repo.save_csv(CSV_FILE)

# Load existing data if any
if os.path.exists(CSV_FILE):
    repo.load_csv(CSV_FILE)

# ── Pydantic models ──

class MatchEntry(BaseModel):
    date: str          # "YYYY-MM-DD"
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


# ── routes ──

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/matches")
def list_matches():
    """Return all stored historical matches, newest first."""
    out = []
    for m in sorted(repo.matches, key=lambda x: x.date, reverse=True):
        out.append({
            "date":       m.date.isoformat(),
            "home_team":  m.home_team,
            "away_team":  m.away_team,
            "home_score": m.home_score,
            "away_score": m.away_score,
            "home_xg":    m.home_xg,
            "away_xg":    m.away_xg,
        })
    return out


@app.post("/matches")
def add_match(entry: MatchEntry):
    """Add a single historical match."""
    try:
        parsed = _parse_date(entry.date)
    except ValueError as e:
        raise HTTPException(400, str(e))

    m = Match(
        date=parsed,
        home_team=TeamRegistry.normalize(entry.home_team),
        away_team=TeamRegistry.normalize(entry.away_team),
        home_score=entry.home_score,
        away_score=entry.away_score,
        home_xg=entry.home_xg,
        away_xg=entry.away_xg,
    )
    added = repo.add_match(m, skip_duplicates=True)
    predictor.stats.invalidate_cache()
    save_matches()
    return {"added": added, "total": len(repo.matches)}


@app.delete("/matches")
def delete_match(req: DeleteMatchRequest):
    """Remove a match by (date, home, away)."""
    try:
        d = _parse_date(req.date)
    except ValueError as e:
        raise HTTPException(400, str(e))
    home = TeamRegistry.normalize(req.home_team)
    away = TeamRegistry.normalize(req.away_team)
    key = (d, home, away)
    if key not in repo._index:
        raise HTTPException(404, "Match not found")
    idx = repo._index.pop(key)
    repo.matches.pop(idx)
    repo._index = {
        (m.date, m.home_team, m.away_team): i
        for i, m in enumerate(repo.matches)
    }
    predictor.stats.invalidate_cache()
    save_matches()
    return {"deleted": True, "total": len(repo.matches)}


@app.post("/predict")
def predict(req: PredictRequest):
    """Run a single prediction."""
    if len(repo.matches) == 0:
        raise HTTPException(400, "No historical data loaded. Add some matches first.")
    result = predictor.predict_match(
        req.home_team, req.away_team,
        req.odds_h, req.odds_d, req.odds_a,
    )
    pred_labels = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    return {
        "home_team":  result.home_team,
        "away_team":  result.away_team,
        "prediction": pred_labels.get(result.prediction, result.prediction),
        "confidence": round(result.confidence * 100, 1),
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
    }


@app.get("/teams")
def list_teams():
    """Return all teams currently in the repository."""
    return sorted(repo.get_all_teams())


@app.get("/table")
def league_table():
    """Return computed league table."""
    table = predictor.stats.compute_league_table()
    return [
        {
            "position": pos,
            "team": name,
            "played": ts.matches_played,
            "points": ts.points,
            "gd": ts.goal_difference,
            "scored": ts.goals_scored,
            "conceded": ts.goals_conceded,
            "form": "".join(ts.recent_form),
        }
        for pos, (name, ts) in enumerate(table, 1)
    ]


@app.delete("/reset")
def reset():
    """Clear all match data."""
    repo.reset()
    predictor.stats.invalidate_cache()
    save_matches()
    return {"reset": True}


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file to add multiple matches at once."""
    try:
        content = await file.read()
        reader = csv.DictReader(StringIO(content.decode("utf-8-sig")))
        matches = []
        for row in reader:
            date_str = row.get("date", "").strip()
            if not date_str:
                continue
            parsed = _parse_date(date_str)
            home = TeamRegistry.normalize(row.get("home_team", "").strip())
            away = TeamRegistry.normalize(row.get("away_team", "").strip())
            hs = int(row.get("home_score", 0))
            as_ = int(row.get("away_score", 0))
            home_xg = row.get("home_xg", "").strip()
            away_xg = row.get("away_xg", "").strip()
            matches.append(Match(
                date=parsed,
                home_team=home,
                away_team=away,
                home_score=hs,
                away_score=as_,
                home_xg=float(home_xg) if home_xg else None,
                away_xg=float(away_xg) if away_xg else None,
            ))
        added = repo.add_matches(matches)
        predictor.stats.invalidate_cache()
        save_matches()
        return {"added": added, "total": len(repo.matches)}
    except Exception as e:
        raise HTTPException(400, f"Error parsing CSV: {str(e)}")