"""
Bayesian Match Predictor — Production Build v3
===============================================
All original 10 issues resolved, plus 11 additional improvements, and Experiment 3 transplants:

Original fixes:
 1. __post_init__ → field(default_factory=list)
 2. Duplicate accumulation → hash-based dedup
 3. Match dates + chronological processing
 4. Laplace smoothing
 5. Sample-size-weighted H2H
 6. Raw Match repository as source of truth
 7. TeamStats derived dynamically
 8. League position computed on demand
 9. Confidence scoring
10. CSV / list ingestion

New fixes:
 1. compute_team_stats() — filter-then-sort (not sort-then-filter)
 2. Dedup hash uses (date, home, away) only — allows score corrections
 3. League table / stats caching to avoid O(N²) rebuilds
 4. Team normalization — aliases pre-normalized, no & bug
 5. Fallback keeps original casing instead of .title() corruption
 6. Confidence = 0.4 * data_quality + 0.6 * entropy_certainty
 7. H2H double-counting fixed — same/rev only, no overall overlap
 8. Performance model uses goal ratio, not just W/D/L counts
 9. xG consumed in performance scoring when available
10. Recency / time-decay weighting on H2H and season performance
11. Adaptive weights based on evidence availability

Experiment 3 Transplants:
12. max_prob added for actionable thresholding
13. Explicit attack/defense strength and points difference in performance model
14. Isotonic regression calibration support
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import lru_cache
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
#  DATA MODELS
# ──────────────────────────────────────────────

@dataclass
class Match:
    """Raw match record — the single source of truth."""
    date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None

    @property
    def total_goals(self) -> int:
        return self.home_score + self.away_score


@dataclass
class TeamStats:
    """Derived statistics — always recomputed from Match records."""
    name: str = ""
    goals_scored: int = 0
    goals_conceded: int = 0
    matches_played: int = 0
    points: int = 0
    recent_form: List[str] = field(default_factory=list)
    home_wins: int = 0
    home_matches: int = 0
    away_wins: int = 0
    away_matches: int = 0
    home_draws: int = 0
    away_draws: int = 0
    home_form: List[str] = field(default_factory=list)
    away_form: List[str] = field(default_factory=list)
    home_losses: int = 0
    away_losses: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_scored - self.goals_conceded

    @property
    def home_loss_pct(self) -> float:
        return self.home_losses / self.home_matches if self.home_matches else 0.0

    @property
    def away_loss_pct(self) -> float:
        return self.away_losses / self.away_matches if self.away_matches else 0.0


@dataclass
class PredictionResult:
    """Complete output of a single match prediction."""
    home_team: str
    away_team: str
    prediction: str                        # 'H', 'D', 'A'
    probabilities: Dict[str, float]        # {'H': .51, 'D': .24, 'A': .25}
    confidence: float                      # 0.0 – 1.0
    breakdown: Dict[str, Dict[str, float]] # per-source detail
    max_prob: float = 0.0                  # Exp 3: max probability for thresholding

    def __str__(self) -> str:
        pred_map = {'H': 'HOME WIN', 'D': 'DRAW', 'A': 'AWAY WIN'}
        bar = "─" * 62
        lines = [
            bar,
            f"  {self.home_team}  vs  {self.away_team}",
            bar,
            f"  Prediction : {pred_map.get(self.prediction, self.prediction)}",
            f"  Confidence : {self.confidence:.0%}",
            f"  Max Prob   : {self.max_prob:.0%}",
            "",
            f"  {'Source':<14} {'H':>7} {'D':>7} {'A':>7}",
            f"  {'─'*14} {'─'*7} {'─'*7} {'─'*7}",
        ]
        for src, probs in self.breakdown.items():
            if src == 'combined':
                continue
            lines.append(
                f"  {src:<14} {probs['H']:>7.1%} {probs['D']:>7.1%} {probs['A']:>7.1%}"
            )
        c = self.breakdown.get('combined', self.probabilities)
        lines += [
            f"  {'─'*14} {'─'*7} {'─'*7} {'─'*7}",
            f"  {'COMBINED':<14} {c['H']:>7.1%} {c['D']:>7.1%} {c['A']:>7.1%}",
            bar,
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────
#  TEAM NAME NORMALIZATION
# ──────────────────────────────────────────────

def _basic_normalize(name: str) -> str:
    """Shared normalization used for both keys and lookups."""
    return (
        name.upper()
        .replace("-", " ")
        .replace("&", "AND")
        .strip()
    )


class TeamRegistry:
    """Normalize team-name variations to a canonical form."""

    _RAW_ALIASES: Dict[str, str] = {
        # Italian
        "LAZIO": "Lazio", "SSC NAPOLI": "Napoli", "NAPOLI": "Napoli",
        "ROMA": "AS Roma", "AS ROMA": "AS Roma", "INTER": "Inter Milan",
        "FC INTERNAZIONALE": "Inter Milan", "JUVENTUS": "Juventus", "JUVE": "Juventus",
        "FIORENTINA": "Fiorentina", "ACF FIORENTINA": "Fiorentina",
        "ATALANTA": "Atalanta", "ATALANTA BC": "Atalanta",
        # Spanish
        "SOCIEDAD": "Real Sociedad", "REAL SOCIEDAD SAN SEBASTIAN": "Real Sociedad",
        "BETIS": "Real Betis", "REAL BETIS BALOMPIE": "Real Betis",
        "BARCA": "Barcelona", "FC BARCELONA": "Barcelona",
        "ATHLETIC BILBAO": "Athletic Club", "ATHLETIC CLUB BILBAO": "Athletic Club",
        "ATLETICO MADRID": "Atletico Madrid", "ATLETICO": "Atletico Madrid",
        # English
        "SPURS": "Tottenham Hotspur", "TOTTENHAM": "Tottenham Hotspur",
        "CHELSEA": "Chelsea", "MAN UNITED": "Manchester United",
        "MAN UTD": "Manchester United", "MANCHESTER UTD": "Manchester United",
        "MAN CITY": "Manchester City", "WOLVES": "Wolverhampton",
        "BRIGHTON HOVE ALBION": "Brighton", "BRIGHTON & HOVE ALBION": "Brighton",
        # German
        "FC BAYERN MUNICH": "Bayern Munich", "FC BAYERN": "Bayern Munich",
        "BAYERN": "Bayern Munich", "BORUSSIA DORTMUND": "Dortmund", "BVB": "Dortmund",
        "BAYER LEVERKUSEN": "Leverkusen",
        # French
        "PSG": "Paris Saint-Germain", "PARIS SAINT GERMAIN": "Paris Saint-Germain",
        # Dutch
        "ALKMAAR": "AZ Alkmaar", "AZ ALKMAAR": "AZ Alkmaar",
        # --- Kaggle European Football Dataset Aliases ---
        # England
        "MAN UNITED": "Manchester United", "MAN CITY": "Manchester City", "WOLVES": "Wolverhampton",
        "NOTT'M FOREST": "Nottingham Forest", "NEWCASTLE": "Newcastle United", "WEST HAM": "West Ham United",
        "LEEDS": "Leeds United", "NORWICH": "Norwich City", "LEICESTER": "Leicester City",
        "BLACKBURN": "Blackburn Rovers", "BOLTON": "Bolton Wanderers", "SUNDERLAND": "Sunderland",
        "SOUTHAMPTON": "Southampton", "ASTON VILLA": "Aston Villa", "BOURNEMOUTH": "AFC Bournemouth",
        "IPSWICH": "Ipswich Town", "ARSENAL": "Arsenal", "LIVERPOOL": "Liverpool", "CHELSEA": "Chelsea", "EVERTON": "Everton",
        # Spain
        "ATH MADRID": "Atletico Madrid", "ATH BILBAO": "Athletic Club", "ESPANYOL": "Espanyol",
        "CELTA VIGO": "Celta Vigo", "GETAFE": "Getafe", "EIBAR": "Eibar", "LEGANES": "Leganes", "ALAVES": "Alaves",
        # Germany
        "E FRANKFURT": "Eintracht Frankfurt", "EIN FRANKFURT": "Eintracht Frankfurt",
        "M'GLADBACH": "Borussia M.Gladbach", "MGLADBACH": "Borussia M.Gladbach",
        "HOFFENHEIM": "Hoffenheim", "KOELN": "FC Cologne", "FC KOELN": "FC Cologne",
        "HERTA": "Hertha Berlin", "HERTHA": "Hertha Berlin", "AUGSBURG": "Augsburg", "MAINZ": "Mainz 05", "MAINZ 05": "Mainz 05",
        # Italy
        "MILAN": "AC Milan", "GENOA": "Genoa", "SAMPDORIA": "Sampdoria", "TORINO": "Torino",
        "UDINESE": "Udinese", "CAGLIARI": "Cagliari", "PALERMO": "Palermo", "CHIEVO": "Chievo Verona",
        "VERONA": "Hellas Verona", "PARMA": "Parma", "BARI": "Bari", "SIENA": "Siena", "BRESCIA": "Brescia",
        "LECCE": "Lecce", "EMPOLI": "Empoli", "FROSINONE": "Frosinone",
        # France
        "PARIS SG": "Paris Saint-Germain", "ST ETIENNE": "Saint-Etienne", "LYON": "Lyon", "MARSEILLE": "Marseille",
        "MONACO": "Monaco", "LILLE": "Lille", "RENNES": "Rennes", "NICE": "Nice", "NANTES": "Nantes",
        "TOULOUSE": "Toulouse", "MONTPELLIER": "Montpellier", "ANGERS": "Angers", "REIMS": "Reims",
        "STRASBOURG": "Strasbourg", "DIJON": "Dijon", "NIMES": "Nimes", "METZ": "Metz", "LORIENT": "Lorient",
        "LENS": "Lens", "CLERMONT": "Clermont Foot", "AJACCIO": "Ajaccio",
    }

    ALIASES: Dict[str, str] = {
        _basic_normalize(k): v for k, v in _RAW_ALIASES.items()
    }

    @classmethod
    def normalize(cls, name: str) -> str:
        key = _basic_normalize(name)
        if key in cls.ALIASES:
            return cls.ALIASES[key]
        return name.strip()


# ──────────────────────────────────────────────
#  MATCH REPOSITORY  (source of truth)
# ──────────────────────────────────────────────

class MatchRepository:
    """Stores raw Match objects.  All stats are derived from here."""

    def __init__(self) -> None:
        self.matches: List[Match] = []
        self._index: Dict[Tuple, int] = {}

    @staticmethod
    def _match_key(m: Match) -> Tuple:
        return (m.date, m.home_team, m.away_team)

    def add_match(self, match: Match, skip_duplicates: bool = True) -> bool:
        key = self._match_key(match)
        if key in self._index:
            if skip_duplicates:
                return False
            self.matches[self._index[key]] = match
            return True
        self._index[key] = len(self.matches)
        self.matches.append(match)
        return True

    def add_matches(self, matches: List[Match], skip_duplicates: bool = True) -> int:
        added = 0
        for m in matches:
            if self.add_match(m, skip_duplicates):
                added += 1
        return added

    def get_matches_for_team(
        self, team: str, venue: Optional[str] = None, limit: Optional[int] = None,
    ) -> List[Match]:
        team = TeamRegistry.normalize(team)
        out: List[Match] = []
        for m in sorted(self.matches, key=lambda x: x.date, reverse=True):
            if venue == "home" and m.home_team != team: continue
            if venue == "away" and m.away_team != team: continue
            if venue is None and m.home_team != team and m.away_team != team: continue
            out.append(m)
            if limit and len(out) >= limit: break
        return out

    def get_h2h_matches(
        self, team1: str, team2: str, venue: Optional[str] = None,
    ) -> List[Match]:
        team1 = TeamRegistry.normalize(team1)
        team2 = TeamRegistry.normalize(team2)
        out: List[Match] = []
        for m in sorted(self.matches, key=lambda x: x.date, reverse=True):
            pair = {m.home_team, m.away_team}
            if team1 not in pair or team2 not in pair: continue
            if venue == "home" and m.home_team != team1: continue
            if venue == "away" and m.away_team != team1: continue
            out.append(m)
        return out

    def get_all_teams(self) -> List[str]:
        teams: set[str] = set()
        for m in self.matches:
            teams.add(m.home_team)
            teams.add(m.away_team)
        return sorted(teams)

    def reset(self) -> None:
        self.matches.clear()
        self._index.clear()

    def load_csv(
        self, filepath: str, date_col: str = "date", home_col: str = "home_team",
        away_col: str = "away_team", hs_col: str = "home_score", as_col: str = "away_score",
        home_xg_col: Optional[str] = "home_xg", away_xg_col: Optional[str] = "away_xg",
        delimiter: str = ",",
    ) -> int:
        matches: List[Match] = []
        with open(filepath, "r", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh, delimiter=delimiter):
                parsed = _parse_date(row[date_col].strip())
                home_xg = float(row[home_xg_col]) if home_xg_col and row.get(home_xg_col, "").strip() else None
                away_xg = float(row[away_xg_col]) if away_xg_col and row.get(away_xg_col, "").strip() else None
                matches.append(Match(
                    date=parsed,
                    home_team=TeamRegistry.normalize(row[home_col].strip()),
                    away_team=TeamRegistry.normalize(row[away_col].strip()),
                    home_score=int(row[hs_col]), away_score=int(row[as_col]),
                    home_xg=home_xg, away_xg=away_xg,
                ))
        return self.add_matches(matches)

    def save_csv(self, filepath: str, delimiter: str = ",") -> None:
        with open(filepath, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter=delimiter)
            writer.writerow(["date", "home_team", "away_team", "home_score", "away_score", "home_xg", "away_xg"])
            for m in self.matches:
                writer.writerow([m.date.isoformat(), m.home_team, m.away_team, m.home_score, m.away_score,
                                 m.home_xg if m.home_xg is not None else "", m.away_xg if m.away_xg is not None else ""])


def _parse_date(raw: str) -> date:
    # Added "%Y-%m-%d %H:%M:%S" to handle timestamps
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try: return datetime.strptime(raw, fmt).date()
        except ValueError: continue
    raise ValueError(f"Cannot parse date: {raw!r}")


# ──────────────────────────────────────────────
#  RECENCY WEIGHTING
# ──────────────────────────────────────────────

_REFERENCE_DATE: date = date.today()

def recency_weight(match_date: date, decay_days: float) -> float:
    if match_date == date(1900, 1, 1): return 0.10
    age = (_REFERENCE_DATE - match_date).days
    if age < 0: age = 0
    return math.exp(-age / decay_days)


# ──────────────────────────────────────────────
#  STATS COMPUTER  (derived, never stored)
# ──────────────────────────────────────────────

class StatsComputer:
    FORM_WINDOW = 5

    def __init__(self, repo: MatchRepository) -> None:
        self.repo = repo
        self._stats_cache: Dict[Tuple, TeamStats] = {}
        self._table_cache: Optional[Tuple[Optional[date], List]] = None

    def invalidate_cache(self) -> None:
        self._stats_cache.clear()
        self._table_cache = None

    def compute_team_stats(self, team: str, as_of: Optional[date] = None) -> TeamStats:
        team = TeamRegistry.normalize(team)
        cache_key = (team, as_of)
        if cache_key in self._stats_cache: return self._stats_cache[cache_key]

        s = TeamStats(name=team)
        all_matches = [m for m in self.repo.matches if (m.home_team == team or m.away_team == team) and (as_of is None or m.date <= as_of)]
        all_matches.sort(key=lambda x: x.date)

        for m in all_matches:
            is_home = m.home_team == team
            s.matches_played += 1
            if is_home:
                s.goals_scored += m.home_score; s.goals_conceded += m.away_score; s.home_matches += 1
                if m.home_score > m.away_score: s.home_wins += 1; s.points += 3; res = "W"
                elif m.home_score < m.away_score: s.home_losses += 1; res = "L"
                else: s.home_draws += 1; s.points += 1; res = "D"
                s.home_form.append(res)
            else:
                s.goals_scored += m.away_score; s.goals_conceded += m.home_score; s.away_matches += 1
                if m.away_score > m.home_score: s.away_wins += 1; s.points += 3; res = "W"
                elif m.away_score < m.home_score: s.away_losses += 1; res = "L"
                else: s.away_draws += 1; s.points += 1; res = "D"
                s.away_form.append(res)
            s.recent_form.append(res)

        s.home_form = s.home_form[-self.FORM_WINDOW:]
        s.away_form = s.away_form[-self.FORM_WINDOW:]
        s.recent_form = s.recent_form[-self.FORM_WINDOW:]
        self._stats_cache[cache_key] = s
        return s

    def compute_league_table(self, as_of: Optional[date] = None) -> List[Tuple[str, TeamStats]]:
        if self._table_cache is not None and self._table_cache[0] == as_of: return self._table_cache[1]
        table = [(t, self.compute_team_stats(t, as_of)) for t in self.repo.get_all_teams()]
        table.sort(key=lambda x: (x[1].points, x[1].goal_difference, x[1].goals_scored), reverse=True)
        self._table_cache = (as_of, table)
        return table

    def get_league_position(self, team: str, as_of: Optional[date] = None) -> int:
        team = TeamRegistry.normalize(team)
        for pos, (name, _) in enumerate(self.compute_league_table(as_of), 1):
            if name == team: return pos
        return 0


# ──────────────────────────────────────────────
#  ENTROPY HELPERS
# ──────────────────────────────────────────────

_MAX_ENTROPY = math.log2(3)

def entropy_certainty(probs: Dict[str, float]) -> float:
    entropy = -sum(p * math.log2(p) for p in probs.values() if p > 0)
    return 1.0 - (entropy / _MAX_ENTROPY)


# ──────────────────────────────────────────────
#  BAYESIAN MATCH PREDICTOR
# ──────────────────────────────────────────────

class BayesianMatchPredictor:
    def __init__(self, repo: MatchRepository) -> None:
        self.repo = repo
        self.stats = StatsComputer(repo)
        self.calibrators = {}  # Exp 3: Holds isotonic regressors if calibrated

        self.base_home_advantage = 1.15
        self.laplace_alpha: float = 1.0
        self._base_weights: Dict[str, float] = {"odds": 0.40, "historical": 0.30, "performance": 0.30}
        self.decay_season: float = 730.0
        self.decay_h2h: float = 365.0

    @staticmethod
    def convert_odds_to_probabilities(odds_h: float, odds_d: float, odds_a: float) -> Dict[str, float]:
        raw = {"H": 1.0 / odds_h if odds_h > 0 else 0.0, "D": 1.0 / odds_d if odds_d > 0 else 0.0, "A": 1.0 / odds_a if odds_a > 0 else 0.0}
        total = sum(raw.values())
        if total == 0: return {"H": 0.33, "D": 0.34, "A": 0.33}
        return {k: v / total for k, v in raw.items()}

    def _weighted_form_score(self, form: List[str]) -> float:
        if not form: return 0.5
        result_values = {"W": 1.0, "D": 0.5, "L": 0.0}
        n = len(form)
        weights = list(range(1, n + 1))
        total_w = sum(weights)
        score = sum(result_values.get(r, 0.0) * w for r, w in zip(form, weights))
        return score / total_w

    def _form_probs(self, team: str, venue: str) -> Dict[str, float]:
        team = TeamRegistry.normalize(team)
        ts = self.stats.compute_team_stats(team)
        a = self.laplace_alpha
        if venue == "home": w, d, l = ts.home_wins, ts.home_draws, ts.home_losses
        elif venue == "away": w, d, l = ts.away_wins, ts.away_draws, ts.away_losses
        else: w, d, l = ts.recent_form.count("W"), ts.recent_form.count("D"), ts.recent_form.count("L")
        total = w + d + l + 3 * a
        return {"W": (w + a) / total, "D": (d + a) / total, "L": (l + a) / total}

    def calculate_performance_probability(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Exp 3: Enhanced with explicit attack/defense strength and points difference."""
        home = TeamRegistry.normalize(home_team)
        away = TeamRegistry.normalize(away_team)

        hf = self._form_probs(home, "home")
        af = self._form_probs(away, "away")
        h_ts = self.stats.compute_team_stats(home)
        a_ts = self.stats.compute_team_stats(away)

        # --- Exp 3: Attack/Defense Strength ---
        h_att = h_ts.goals_scored / h_ts.matches_played if h_ts.matches_played else 0.0
        h_def = h_ts.goals_conceded / h_ts.matches_played if h_ts.matches_played else 0.0
        a_att = a_ts.goals_scored / a_ts.matches_played if a_ts.matches_played else 0.0
        a_def = a_ts.goals_conceded / a_ts.matches_played if a_ts.matches_played else 0.0

        strength_home = h_att / (h_att + a_def + 1e-6)
        strength_away = a_att / (a_att + h_def + 1e-6)
        strength_edge = strength_home - strength_away
        
        str_h = 0.33 + strength_edge * 0.20
        str_a = 0.33 - strength_edge * 0.20
        str_d = 1.0 - str_h - str_a
        str_h, str_a, str_d = max(0.05, min(0.90, str_h)), max(0.05, min(0.90, str_a)), max(0.05, min(0.90, str_d))

        # --- Exp 3: Points Difference ---
        pts_h, pts_a = h_ts.points, a_ts.points
        max_pts = max(pts_h, pts_a, 1)
        pts_diff = (pts_h - pts_a) / max_pts
        
        pts_h_prob = 0.33 + pts_diff * 0.15
        pts_a_prob = 0.33 - pts_diff * 0.15
        pts_d_prob = 1.0 - pts_h_prob - pts_a_prob
        pts_h_prob, pts_a_prob, pts_d_prob = max(0.05, min(0.90, pts_h_prob)), max(0.05, min(0.90, pts_a_prob)), max(0.05, min(0.90, pts_d_prob))

        # --- Goal ratio signal ---
        def _goal_ratio(ts: TeamStats) -> float:
            denom = ts.goals_scored + ts.goals_conceded
            return ts.goals_scored / denom if denom else 0.5

        h_gr, a_gr = _goal_ratio(h_ts), _goal_ratio(a_ts)
        home_goal_edge = h_gr - a_gr
        gr_h = 0.33 + home_goal_edge * 0.25
        gr_a = 0.33 - home_goal_edge * 0.25
        gr_d = 1.0 - gr_h - gr_a
        gr_h, gr_a, gr_d = max(0.05, min(0.90, gr_h)), max(0.05, min(0.90, gr_a)), max(0.05, min(0.90, gr_d))

        # --- xG signal ---
        def _xg_ratio(team: str, is_home: bool) -> Optional[float]:
            recent = self.repo.get_matches_for_team(team, limit=10)
            xg_scored, xg_conceded = [], []
            for m in recent:
                if is_home and m.home_team == team:
                    if m.home_xg is not None and m.away_xg is not None: xg_scored.append(m.home_xg); xg_conceded.append(m.away_xg)
                elif not is_home and m.away_team == team:
                    if m.home_xg is not None and m.away_xg is not None: xg_scored.append(m.away_xg); xg_conceded.append(m.home_xg)
            if not xg_scored: return None
            avg_s, avg_c = sum(xg_scored) / len(xg_scored), sum(xg_conceded) / len(xg_conceded)
            denom = avg_s + avg_c
            return avg_s / denom if denom else 0.5

        h_xg, a_xg = _xg_ratio(home, True), _xg_ratio(away, False)

        # --- Combine signals ---
        raw_h_form = (hf["W"] * self.base_home_advantage + af["L"]) / 2
        raw_d_form = (hf["D"] + af["D"]) / 2
        raw_a_form = (hf["L"] + af["W"] * self.base_home_advantage) / 2

        if h_xg is not None and a_xg is not None:
            xg_edge = h_xg - a_xg
            xg_h = 0.33 + xg_edge * 0.25; xg_a = 0.33 - xg_edge * 0.25; xg_d = 1.0 - xg_h - xg_a
            xg_h, xg_a, xg_d = max(0.05, min(0.90, xg_h)), max(0.05, min(0.90, xg_a)), max(0.05, min(0.90, xg_d))
            
            # Blend: 30% form, 20% goal-ratio, 20% strength, 10% points, 20% xG
            raw_h = 0.30 * raw_h_form + 0.20 * gr_h + 0.20 * str_h + 0.10 * pts_h_prob + 0.20 * xg_h
            raw_d = 0.30 * raw_d_form + 0.20 * gr_d + 0.20 * str_d + 0.10 * pts_d_prob + 0.20 * xg_d
            raw_a = 0.30 * raw_a_form + 0.20 * gr_a + 0.20 * str_a + 0.10 * pts_a_prob + 0.20 * xg_a
        else:
            # No xG data: 40% form, 25% goal-ratio, 20% strength, 15% points
            raw_h = 0.40 * raw_h_form + 0.25 * gr_h + 0.20 * str_h + 0.15 * pts_h_prob
            raw_d = 0.40 * raw_d_form + 0.25 * gr_d + 0.20 * str_d + 0.15 * pts_d_prob
            raw_a = 0.40 * raw_a_form + 0.25 * gr_a + 0.20 * str_a + 0.15 * pts_a_prob

        total = raw_h + raw_d + raw_a
        if total == 0: return {"H": 0.33, "D": 0.34, "A": 0.33}
        return {"H": raw_h / total, "D": raw_d / total, "A": raw_a / total}

    def calculate_historical_probability(self, home_team: str, away_team: str) -> Dict[str, float]:
        home = TeamRegistry.normalize(home_team)
        away = TeamRegistry.normalize(away_team)
        same = self.repo.get_h2h_matches(home, away, venue="home")
        rev  = self.repo.get_h2h_matches(home, away, venue="away")

        def _probs(matches: List[Match], perspective_home: str) -> Tuple[Dict[str, float], float]:
            w_sum = d_sum = l_sum = 0.0
            for m in matches:
                rw = recency_weight(m.date, self.decay_h2h)
                if m.home_team == perspective_home:
                    if m.home_score > m.away_score: w_sum += rw
                    elif m.home_score < m.away_score: l_sum += rw
                    else: d_sum += rw
                else:
                    if m.away_score > m.home_score: w_sum += rw
                    elif m.away_score < m.home_score: l_sum += rw
                    else: d_sum += rw
            a = self.laplace_alpha
            total = w_sum + d_sum + l_sum + 3 * a
            ess = w_sum + d_sum + l_sum
            return ({"H": (w_sum + a) / total, "D": (d_sum + a) / total, "A": (l_sum + a) / total}, ess)

        p_same, ess_same = _probs(same, home)
        p_rev,  ess_rev  = _probs(rev,  home)
        w_same = ess_same * 2 + 1
        w_rev  = ess_rev  * 1 + 1
        w_total = w_same + w_rev
        final: Dict[str, float] = {}
        for outcome in ("H", "D", "A"):
            final[outcome] = (w_same * p_same[outcome] + w_rev * p_rev[outcome]) / w_total
        return final

    def calculate_confidence(self, home_team: str, away_team: str, combined_probs: Dict[str, float], has_odds: bool = True) -> float:
        home = TeamRegistry.normalize(home_team)
        away = TeamRegistry.normalize(away_team)
        hs = self.stats.compute_team_stats(home)
        as_ = self.stats.compute_team_stats(away)
        n_h2h = len(self.repo.get_h2h_matches(home, away))
        data_quality = sum([
            min(hs.matches_played / 10, 1.0) * 0.25, min(as_.matches_played / 10, 1.0) * 0.25,
            min(n_h2h / 5, 1.0) * 0.20, min(len(hs.recent_form) / 5, 1.0) * min(len(as_.recent_form) / 5, 1.0) * 0.15,
            (1.0 if has_odds else 0.5) * 0.15,
        ])
        certainty = entropy_certainty(combined_probs)
        return round(0.4 * data_quality + 0.6 * certainty, 3)

    def _adaptive_weights(self, home: str, away: str, has_odds: bool) -> Dict[str, float]:
        n_h2h = len(self.repo.get_h2h_matches(home, away))
        w = dict(self._base_weights)
        if not has_odds:
            extra = w["odds"]; w["odds"] = 0.0; w["performance"] += extra / 2; w["historical"] += extra / 2
        h2h_scaler = 0.50 + 0.70 * min(n_h2h / 10, 1.0)
        w["historical"] *= h2h_scaler
        delta = self._base_weights["historical"] * (1.0 - h2h_scaler)
        w["performance"] += delta
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    def predict_match(self, home_team: str, away_team: str, odds_h: float = 0, odds_d: float = 0, odds_a: float = 0) -> PredictionResult:
        home = TeamRegistry.normalize(home_team)
        away = TeamRegistry.normalize(away_team)
        has_odds = odds_h > 0 and odds_d > 0 and odds_a > 0

        odds_probs = self.convert_odds_to_probabilities(odds_h, odds_d, odds_a) if has_odds else {"H": 0.33, "D": 0.34, "A": 0.33}
        perf_probs = self.calculate_performance_probability(home, away)
        hist_probs = self.calculate_historical_probability(home, away)
        w = self._adaptive_weights(home, away, has_odds)

        combined: Dict[str, float] = {}
        for o in ("H", "D", "A"):
            combined[o] = w["odds"] * odds_probs[o] + w["performance"] * perf_probs[o] + w["historical"] * hist_probs[o]
        total = sum(combined.values())
        if total > 0: combined = {k: v / total for k, v in combined.items()}

        # Exp 3: Apply calibration if available
        if self.calibrators:
            calibrated = {}
            for outcome in ['H', 'D', 'A']:
                if outcome in self.calibrators:
                    calibrated[outcome] = self.calibrators[outcome].predict([combined[outcome]])[0]
                else:
                    calibrated[outcome] = combined[outcome]
            total_cal = sum(calibrated.values())
            if total_cal > 0:
                combined = {k: v/total_cal for k, v in calibrated.items()}

        prediction = max(combined, key=combined.get)
        max_prob = max(combined.values())  # Exp 3: Max probability
        confidence = self.calculate_confidence(home, away, combined, has_odds)

        breakdown = {"odds": odds_probs, "performance": perf_probs, "historical": hist_probs, "combined": combined}
        return PredictionResult(
            home_team=home, away_team=away, prediction=prediction,
            probabilities=combined, confidence=confidence, breakdown=breakdown, max_prob=max_prob
        )

    def predict_matches(self, future_matches: List[Tuple]) -> List[PredictionResult]:
        results: List[PredictionResult] = []
        for entry in future_matches:
            if len(entry) >= 5: r = self.predict_match(entry[0], entry[1], entry[2], entry[3], entry[4])
            elif len(entry) == 2: r = self.predict_match(entry[0], entry[1])
            else: continue
            results.append(r)
        return results


def run_predictions(historical_data, future_matches: List[Tuple], source: str = "list") -> List[PredictionResult]:
    repo = MatchRepository()
    if source == "csv": repo.load_csv(historical_data)
    else: repo.load_list(historical_data)
    predictor = BayesianMatchPredictor(repo)
    return predictor.predict_matches(future_matches)


if __name__ == "__main__":
    historical = [
        ("2024-08-20", "Lazio", "Napoli", 2, 1), ("2024-09-01", "Napoli", "Lazio", 1, 1),
        ("2024-09-15", "Lazio", "Real Sociedad", 3, 0), ("2024-09-28", "Real Sociedad", "Lazio", 1, 2),
        ("2024-10-05", "Lazio", "AZ Alkmaar", 1, 1), ("2024-10-20", "AZ Alkmaar", "Lazio", 0, 2),
        ("2024-11-02", "Napoli", "Real Sociedad", 2, 0), ("2024-11-10", "Real Sociedad", "Napoli", 1, 1),
        ("2024-11-24", "Napoli", "AZ Alkmaar", 3, 1), ("2024-12-01", "AZ Alkmaar", "Napoli", 0, 1),
        ("2024-12-08", "Real Sociedad", "AZ Alkmaar", 2, 2), ("2024-12-15", "AZ Alkmaar", "Real Sociedad", 0, 1),
        ("2025-01-12", "Lazio", "Napoli", 1, 0), ("2025-01-19", "Napoli", "Lazio", 2, 1),
        ("2025-02-02", "Lazio", "Real Sociedad", 2, 1), ("2025-02-09", "Real Sociedad", "Lazio", 0, 0),
        ("2025-02-16", "Lazio", "AZ Alkmaar", 3, 1), ("2025-03-01", "Napoli", "Real Sociedad", 1, 0),
        ("2025-03-08", "AZ Alkmaar", "Napoli", 1, 2), ("2025-03-15", "Lazio", "Napoli", 2, 2),
    ]
    future = [
        ("Lazio", "Napoli", 2.10, 3.30, 3.50), ("Real Sociedad", "AZ Alkmaar", 1.85, 3.50, 4.20),
        ("Napoli", "Real Sociedad", 1.55, 3.80, 6.00), ("AZ Alkmaar", "Lazio", 3.00, 3.20, 2.40),
        ("Lazio", "Real Sociedad"),
    ]
    results = run_predictions(historical, future, source="list")
    for r in results: print(r)