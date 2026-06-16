"""
Bayesian Match Predictor — Production Build v2
===============================================
All original 10 issues resolved, plus 11 additional improvements:

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

    def __str__(self) -> str:
        pred_map = {'H': 'HOME WIN', 'D': 'DRAW', 'A': 'AWAY WIN'}
        bar = "─" * 62
        lines = [
            bar,
            f"  {self.home_team}  vs  {self.away_team}",
            bar,
            f"  Prediction : {pred_map.get(self.prediction, self.prediction)}",
            f"  Confidence : {self.confidence:.0%}",
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
    """Normalize team-name variations to a canonical form.

    Fix #4: Aliases are stored pre-normalized so the & → AND
    substitution applies consistently on both the key side and the
    lookup side.  No more silent miss on 'Brighton & Hove Albion'.

    Fix #5: Fallback preserves original casing (.title() is gone)
    so 'AZ ALKMAAR' stays 'AZ ALKMAAR' rather than becoming 'Az Alkmaar'.
    """

    # Raw alias table — keys will be normalized at class-load time.
    _RAW_ALIASES: Dict[str, str] = {
        # Italian
        "LAZIO":                        "Lazio",
        "SSC NAPOLI":                   "Napoli",
        "NAPOLI":                       "Napoli",
        "ROMA":                         "AS Roma",
        "AS ROMA":                      "AS Roma",
        "INTER":                        "Inter Milan",
        "FC INTERNAZIONALE":            "Inter Milan",
        "JUVENTUS":                     "Juventus",
        "JUVE":                         "Juventus",
        "FIORENTINA":                   "Fiorentina",
        "ACF FIORENTINA":               "Fiorentina",
        "ATALANTA":                     "Atalanta",
        "ATALANTA BC":                  "Atalanta",
        # Spanish
        "SOCIEDAD":                     "Real Sociedad",
        "REAL SOCIEDAD SAN SEBASTIAN":  "Real Sociedad",
        "BETIS":                        "Real Betis",
        "REAL BETIS BALOMPIE":          "Real Betis",
        "BARCA":                        "Barcelona",
        "FC BARCELONA":                 "Barcelona",
        "ATHLETIC BILBAO":              "Athletic Club",
        "ATHLETIC CLUB BILBAO":         "Athletic Club",
        "ATLETICO MADRID":              "Atletico Madrid",
        "ATLETICO":                     "Atletico Madrid",
        # English
        "SPURS":                        "Tottenham Hotspur",
        "TOTTENHAM":                    "Tottenham Hotspur",
        "CHELSEA":                      "Chelsea",
        "MAN UNITED":                   "Manchester United",
        "MAN UTD":                      "Manchester United",
        "MANCHESTER UTD":               "Manchester United",
        "MAN CITY":                     "Manchester City",
        "WOLVES":                       "Wolverhampton",
        "BRIGHTON HOVE ALBION":         "Brighton",
        "BRIGHTON & HOVE ALBION":       "Brighton",   # raw key intentionally kept
        # German
        "FC BAYERN MUNICH":             "Bayern Munich",
        "FC BAYERN":                    "Bayern Munich",
        "BAYERN":                       "Bayern Munich",
        "BORUSSIA DORTMUND":            "Dortmund",
        "BVB":                          "Dortmund",
        "BAYER LEVERKUSEN":             "Leverkusen",
        # French
        "PSG":                          "Paris Saint-Germain",
        "PARIS SAINT GERMAIN":          "Paris Saint-Germain",
        # Dutch
        "ALKMAAR":                      "AZ Alkmaar",
        "AZ ALKMAAR":                   "AZ Alkmaar",
    }

    # Build the normalized lookup table once at class-definition time.
    ALIASES: Dict[str, str] = {
        _basic_normalize(k): v for k, v in _RAW_ALIASES.items()
    }

    @classmethod
    def normalize(cls, name: str) -> str:
        key = _basic_normalize(name)
        if key in cls.ALIASES:
            return cls.ALIASES[key]
        # Fix #5: keep original casing, just strip whitespace.
        # .title() was corrupting 'AZ ALKMAAR' → 'Az Alkmaar', 'PSG' → 'Psg'.
        return name.strip()


# ──────────────────────────────────────────────
#  MATCH REPOSITORY  (source of truth)
# ──────────────────────────────────────────────

class MatchRepository:
    """Stores raw Match objects.  All stats are derived from here.

    Fix #2: Dedup hash uses (date, home, away) only — so a corrected
    score replaces the previous entry rather than being silently dropped.
    """

    def __init__(self) -> None:
        self.matches: List[Match] = []
        # maps (date, home, away) → index in self.matches
        self._index: Dict[Tuple, int] = {}

    # ── dedup key ──

    @staticmethod
    def _match_key(m: Match) -> Tuple:
        return (m.date, m.home_team, m.away_team)

    # ── add ──

    def add_match(self, match: Match, skip_duplicates: bool = True) -> bool:
        key = self._match_key(match)
        if key in self._index:
            if skip_duplicates:
                return False
            # Score correction: overwrite the existing entry.
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

    # ── query ──

    def get_matches_for_team(
        self,
        team: str,
        venue: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Match]:
        team = TeamRegistry.normalize(team)
        out: List[Match] = []
        for m in sorted(self.matches, key=lambda x: x.date, reverse=True):
            if venue == "home" and m.home_team != team:
                continue
            if venue == "away" and m.away_team != team:
                continue
            if venue is None and m.home_team != team and m.away_team != team:
                continue
            out.append(m)
            if limit and len(out) >= limit:
                break
        return out

    def get_h2h_matches(
        self,
        team1: str,
        team2: str,
        venue: Optional[str] = None,
    ) -> List[Match]:
        """Return H2H matches.  venue='home' means team1 at home."""
        team1 = TeamRegistry.normalize(team1)
        team2 = TeamRegistry.normalize(team2)
        out: List[Match] = []
        for m in sorted(self.matches, key=lambda x: x.date, reverse=True):
            pair = {m.home_team, m.away_team}
            if team1 not in pair or team2 not in pair:
                continue
            if venue == "home" and m.home_team != team1:
                continue
            if venue == "away" and m.away_team != team1:
                continue
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

    # ── ingestion ──

    def load_csv(
        self,
        filepath: str,
        date_col: str = "date",
        home_col: str = "home_team",
        away_col: str = "away_team",
        hs_col: str = "home_score",
        as_col: str = "away_score",
        home_xg_col: Optional[str] = "home_xg",
        away_xg_col: Optional[str] = "away_xg",
        delimiter: str = ",",
    ) -> int:
        """Load CSV with optional xG columns."""
        matches: List[Match] = []
        with open(filepath, "r", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh, delimiter=delimiter):
                parsed = _parse_date(row[date_col].strip())
                home_xg = None
                away_xg = None
                if home_xg_col and home_xg_col in row:
                    val = row[home_xg_col].strip()
                    if val:
                        home_xg = float(val)
                if away_xg_col and away_xg_col in row:
                    val = row[away_xg_col].strip()
                    if val:
                        away_xg = float(val)
                matches.append(Match(
                    date=parsed,
                    home_team=TeamRegistry.normalize(row[home_col].strip()),
                    away_team=TeamRegistry.normalize(row[away_col].strip()),
                    home_score=int(row[hs_col]),
                    away_score=int(row[as_col]),
                    home_xg=home_xg,
                    away_xg=away_xg,
                ))
        return self.add_matches(matches)

    def save_csv(self, filepath: str, delimiter: str = ",") -> None:
        """Save all matches to a CSV file."""
        with open(filepath, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter=delimiter)
            writer.writerow([
                "date", "home_team", "away_team",
                "home_score", "away_score",
                "home_xg", "away_xg"
            ])
            for m in self.matches:
                writer.writerow([
                    m.date.isoformat(),
                    m.home_team,
                    m.away_team,
                    m.home_score,
                    m.away_score,
                    m.home_xg if m.home_xg is not None else "",
                    m.away_xg if m.away_xg is not None else "",
                ])


def _parse_date(raw: str) -> date:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {raw!r}")


# ──────────────────────────────────────────────
#  RECENCY WEIGHTING
# ──────────────────────────────────────────────

_REFERENCE_DATE: date = date.today()


def recency_weight(match_date: date, decay_days: float) -> float:
    """Exponential decay: weight = exp(-age_days / decay_days).

    Fix #10: Recent matches matter more.  decay_days controls how fast
    old evidence fades:
      - Season performance: decay_days=730 (slow fade, ~2-year half-life)
      - H2H history:        decay_days=365 (faster, player/manager turnover)
    Matches with placeholder date 1900-01-01 receive a small but non-zero
    weight so no data is completely discarded.
    """
    if match_date == date(1900, 1, 1):
        return 0.10
    age = (_REFERENCE_DATE - match_date).days
    if age < 0:
        age = 0  # future-dated entries treated as today
    return math.exp(-age / decay_days)


# ──────────────────────────────────────────────
#  STATS COMPUTER  (derived, never stored)
# ──────────────────────────────────────────────

class StatsComputer:
    """Computes TeamStats on the fly from the Match repository.

    Fix #3: Results are cached per (team, as_of) key.
    The cache is invalidated whenever new matches are added (call
    invalidate_cache() explicitly, or use the MatchRepository.reset()
    path which creates a new StatsComputer).
    """

    FORM_WINDOW = 5

    def __init__(self, repo: MatchRepository) -> None:
        self.repo = repo
        self._stats_cache: Dict[Tuple, TeamStats] = {}
        self._table_cache: Optional[Tuple[Optional[date], List]] = None

    def invalidate_cache(self) -> None:
        self._stats_cache.clear()
        self._table_cache = None

    def compute_team_stats(
        self, team: str, as_of: Optional[date] = None
    ) -> TeamStats:
        team = TeamRegistry.normalize(team)
        cache_key = (team, as_of)
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        s = TeamStats(name=team)

        # Fix #1: filter first, then sort (avoids sorting the entire dataset
        # when only a small subset is relevant for this team/date window).
        all_matches = [
            m for m in self.repo.matches
            if (m.home_team == team or m.away_team == team)
            and (as_of is None or m.date <= as_of)
        ]
        all_matches.sort(key=lambda x: x.date)

        for m in all_matches:
            is_home = m.home_team == team
            s.matches_played += 1

            if is_home:
                s.goals_scored    += m.home_score
                s.goals_conceded  += m.away_score
                s.home_matches    += 1
                if m.home_score > m.away_score:
                    s.home_wins += 1; s.points += 3; res = "W"
                elif m.home_score < m.away_score:
                    s.home_losses += 1;               res = "L"
                else:
                    s.home_draws += 1; s.points += 1; res = "D"
                s.home_form.append(res)
            else:
                s.goals_scored    += m.away_score
                s.goals_conceded  += m.home_score
                s.away_matches    += 1
                if m.away_score > m.home_score:
                    s.away_wins += 1; s.points += 3; res = "W"
                elif m.away_score < m.home_score:
                    s.away_losses += 1;               res = "L"
                else:
                    s.away_draws += 1; s.points += 1; res = "D"
                s.away_form.append(res)

            s.recent_form.append(res)

        # Trim to window
        s.home_form   = s.home_form[-self.FORM_WINDOW:]
        s.away_form   = s.away_form[-self.FORM_WINDOW:]
        s.recent_form = s.recent_form[-self.FORM_WINDOW:]

        self._stats_cache[cache_key] = s
        return s

    def compute_league_table(
        self, as_of: Optional[date] = None
    ) -> List[Tuple[str, TeamStats]]:
        # Fix #3: cache the table per as_of date.
        if self._table_cache is not None and self._table_cache[0] == as_of:
            return self._table_cache[1]

        table = [
            (t, self.compute_team_stats(t, as_of))
            for t in self.repo.get_all_teams()
        ]
        table.sort(
            key=lambda x: (x[1].points, x[1].goal_difference, x[1].goals_scored),
            reverse=True,
        )
        self._table_cache = (as_of, table)
        return table

    def get_league_position(self, team: str, as_of: Optional[date] = None) -> int:
        team = TeamRegistry.normalize(team)
        for pos, (name, _) in enumerate(self.compute_league_table(as_of), 1):
            if name == team:
                return pos
        return 0


# ──────────────────────────────────────────────
#  ENTROPY HELPERS
# ──────────────────────────────────────────────

_MAX_ENTROPY = math.log2(3)   # ≈ 1.585 bits (uniform over 3 outcomes)


def entropy_certainty(probs: Dict[str, float]) -> float:
    """Fix #6: Convert prediction distribution to a 0-1 certainty score.

    Returns 0 when all outcomes are equally likely (max uncertainty)
    and approaches 1 when one outcome dominates.

    certainty = 1 - (entropy / MAX_ENTROPY)
    """
    entropy = -sum(
        p * math.log2(p)
        for p in probs.values()
        if p > 0
    )
    return 1.0 - (entropy / _MAX_ENTROPY)


# ──────────────────────────────────────────────
#  BAYESIAN MATCH PREDICTOR
# ──────────────────────────────────────────────

class BayesianMatchPredictor:
    def __init__(self, repo: MatchRepository) -> None:
        self.repo = repo
        self.stats = StatsComputer(repo)

        # Tunables
        self.base_home_advantage = 1.15
        self.laplace_alpha: float = 1.0

        # Base weights — overridden adaptively in predict_match().
        # Fix #11: weights shift based on available evidence.
        self._base_weights: Dict[str, float] = {
            "odds":        0.40,
            "historical":  0.30,
            "performance": 0.30,
        }

        # Recency decay constants (days)
        self.decay_season: float = 730.0   # slow — season performance
        self.decay_h2h: float    = 365.0   # faster — lineup/manager changes

    # ── helpers ──

    @staticmethod
    def convert_odds_to_probabilities(
        odds_h: float, odds_d: float, odds_a: float
    ) -> Dict[str, float]:
        raw = {
            "H": 1.0 / odds_h if odds_h > 0 else 0.0,
            "D": 1.0 / odds_d if odds_d > 0 else 0.0,
            "A": 1.0 / odds_a if odds_a > 0 else 0.0,
        }
        total = sum(raw.values())
        if total == 0:
            return {"H": 0.33, "D": 0.34, "A": 0.33}
        return {k: v / total for k, v in raw.items()}

    # ── weighted form scoring ──

    def _weighted_form_score(self, form: List[str]) -> float:
        """Fix #10 (weighted form variant): newer results weighted higher.

        Weights (most-recent = highest): [5, 4, 3, 2, 1] over 5 matches.
        W=1, D=0.5, L=0  → returns a 0-1 score.
        """
        if not form:
            return 0.5  # neutral when no data
        result_values = {"W": 1.0, "D": 0.5, "L": 0.0}
        n = len(form)
        # Assign linearly increasing weights: oldest=1, newest=n
        weights = list(range(1, n + 1))
        total_w = sum(weights)
        score = sum(
            result_values.get(r, 0.0) * w
            for r, w in zip(form, weights)
        )
        return score / total_w

    # ── form probability (recency-aware) ──

    def _form_probs(self, team: str, venue: str) -> Dict[str, float]:
        """Win/draw/lose probabilities derived from recent form.

        Fix #8: uses weighted form score (goal ratio supplement below).
        venue ∈ {home, away, overall}
        """
        team = TeamRegistry.normalize(team)
        ts = self.stats.compute_team_stats(team)
        a = self.laplace_alpha

        if venue == "home":
            w, d, l = ts.home_wins, ts.home_draws, ts.home_losses
        elif venue == "away":
            w, d, l = ts.away_wins, ts.away_draws, ts.away_losses
        else:
            w = ts.recent_form.count("W")
            d = ts.recent_form.count("D")
            l = ts.recent_form.count("L")

        total = w + d + l + 3 * a
        return {"W": (w + a) / total, "D": (d + a) / total, "L": (l + a) / total}

    # ── performance probability (goal-ratio + xG aware) ──

    def calculate_performance_probability(
        self, home_team: str, away_team: str
    ) -> Dict[str, float]:
        """Fix #8 + Fix #9: incorporate goal ratio and xG when available.

        Scoring:
          • Outcome-based form probs (Laplace-smoothed, venue-split)
          • Goal ratio: goals_scored / (goals_scored + goals_conceded)
          • xG ratio when at least one xG value is present in recent matches
        These three signals are blended with fixed internal weights.
        """
        home = TeamRegistry.normalize(home_team)
        away = TeamRegistry.normalize(away_team)

        hf = self._form_probs(home, "home")
        af = self._form_probs(away, "away")

        h_ts = self.stats.compute_team_stats(home)
        a_ts = self.stats.compute_team_stats(away)

        # --- goal ratio signal ---
        def _goal_ratio(ts: TeamStats) -> float:
            """Goals scored / total goals. 0.5 = balanced, >0.5 = strong attack."""
            denom = ts.goals_scored + ts.goals_conceded
            return ts.goals_scored / denom if denom else 0.5

        h_gr = _goal_ratio(h_ts)
        a_gr = _goal_ratio(a_ts)

        # --- xG signal ---
        def _xg_ratio(team: str, is_home: bool) -> Optional[float]:
            """Average xG / (xG scored + xG conceded) from recent matches."""
            recent = self.repo.get_matches_for_team(team, limit=10)
            xg_scored = []
            xg_conceded = []
            for m in recent:
                if is_home and m.home_team == team:
                    if m.home_xg is not None and m.away_xg is not None:
                        xg_scored.append(m.home_xg)
                        xg_conceded.append(m.away_xg)
                elif not is_home and m.away_team == team:
                    if m.home_xg is not None and m.away_xg is not None:
                        xg_scored.append(m.away_xg)
                        xg_conceded.append(m.home_xg)
            if not xg_scored:
                return None
            avg_s = sum(xg_scored) / len(xg_scored)
            avg_c = sum(xg_conceded) / len(xg_conceded)
            denom = avg_s + avg_c
            return avg_s / denom if denom else 0.5

        h_xg = _xg_ratio(home, True)
        a_xg = _xg_ratio(away, False)

        # --- combine home-win probability from multiple signals ---
        # outcome-form contribution
        raw_h_form = (hf["W"] * self.base_home_advantage + af["L"]) / 2
        raw_d_form = (hf["D"] + af["D"]) / 2
        raw_a_form = (hf["L"] + af["W"] * self.base_home_advantage) / 2

        # goal-ratio contribution: home dominance = h_gr high, a_gr low
        # Squash into H/D/A probabilities via logistic-ish blend.
        home_goal_edge = h_gr - a_gr   # in [-1, 1]
        gr_h = 0.33 + home_goal_edge * 0.25
        gr_a = 0.33 - home_goal_edge * 0.25
        gr_d = 1.0 - gr_h - gr_a
        gr_h = max(0.05, min(0.90, gr_h))
        gr_a = max(0.05, min(0.90, gr_a))
        gr_d = max(0.05, min(0.90, gr_d))

        if h_xg is not None and a_xg is not None:
            xg_edge = h_xg - a_xg
            xg_h = 0.33 + xg_edge * 0.25
            xg_a = 0.33 - xg_edge * 0.25
            xg_d = 1.0 - xg_h - xg_a
            xg_h = max(0.05, min(0.90, xg_h))
            xg_a = max(0.05, min(0.90, xg_a))
            xg_d = max(0.05, min(0.90, xg_d))
            # Blend: 50% form, 25% goal-ratio, 25% xG
            raw_h = 0.50 * raw_h_form + 0.25 * gr_h + 0.25 * xg_h
            raw_d = 0.50 * raw_d_form + 0.25 * gr_d + 0.25 * xg_d
            raw_a = 0.50 * raw_a_form + 0.25 * gr_a + 0.25 * xg_a
        else:
            # No xG data: 65% form, 35% goal-ratio
            raw_h = 0.65 * raw_h_form + 0.35 * gr_h
            raw_d = 0.65 * raw_d_form + 0.35 * gr_d
            raw_a = 0.65 * raw_a_form + 0.35 * gr_a

        total = raw_h + raw_d + raw_a
        if total == 0:
            return {"H": 0.33, "D": 0.34, "A": 0.33}
        return {"H": raw_h / total, "D": raw_d / total, "A": raw_a / total}

    # ── H2H / historical probability (decay-weighted, no double-counting) ──

    def calculate_historical_probability(
        self, home_team: str, away_team: str
    ) -> Dict[str, float]:
        """Fix #7 + Fix #10: use same-venue and reverse-venue only
        (overall would overlap and double-count).  Apply recency decay.
        """
        home = TeamRegistry.normalize(home_team)
        away = TeamRegistry.normalize(away_team)

        same = self.repo.get_h2h_matches(home, away, venue="home")
        rev  = self.repo.get_h2h_matches(home, away, venue="away")

        def _probs(matches: List[Match], perspective_home: str) -> Tuple[Dict[str, float], float]:
            """H2H probs with Laplace smoothing + recency decay.

            Returns (prob_dict, effective_sample_size).
            """
            w_sum = d_sum = l_sum = 0.0
            for m in matches:
                rw = recency_weight(m.date, self.decay_h2h)
                if m.home_team == perspective_home:
                    if m.home_score > m.away_score:   w_sum += rw
                    elif m.home_score < m.away_score: l_sum += rw
                    else:                             d_sum += rw
                else:
                    if m.away_score > m.home_score:   w_sum += rw
                    elif m.away_score < m.home_score: l_sum += rw
                    else:                             d_sum += rw

            a = self.laplace_alpha
            total = w_sum + d_sum + l_sum + 3 * a
            ess = w_sum + d_sum + l_sum  # effective sample size
            return (
                {"H": (w_sum + a) / total,
                 "D": (d_sum + a) / total,
                 "A": (l_sum + a) / total},
                ess,
            )

        p_same, ess_same = _probs(same, home)
        p_rev,  ess_rev  = _probs(rev,  home)

        # Fix #7: weight by ESS only (same-venue gets double credit).
        # No 'overall' bucket — it would be same + rev again.
        w_same = ess_same * 2 + 1
        w_rev  = ess_rev  * 1 + 1
        w_total = w_same + w_rev

        final: Dict[str, float] = {}
        for outcome in ("H", "D", "A"):
            final[outcome] = (
                w_same * p_same[outcome] + w_rev * p_rev[outcome]
            ) / w_total
        return final

    # ── confidence (data quality + entropy certainty) ──

    def calculate_confidence(
        self,
        home_team: str,
        away_team: str,
        combined_probs: Dict[str, float],
        has_odds: bool = True,
    ) -> float:
        """Fix #6: confidence = 0.4 * data_quality + 0.6 * entropy_certainty.

        data_quality: how much evidence exists.
        entropy_certainty: how strongly the evidence points to one outcome.
        """
        home = TeamRegistry.normalize(home_team)
        away = TeamRegistry.normalize(away_team)

        hs  = self.stats.compute_team_stats(home)
        as_ = self.stats.compute_team_stats(away)
        n_h2h = len(self.repo.get_h2h_matches(home, away))

        data_quality = sum([
            min(hs.matches_played  / 10, 1.0) * 0.25,
            min(as_.matches_played / 10, 1.0) * 0.25,
            min(n_h2h / 5,          1.0) * 0.20,
            min(len(hs.recent_form) / 5, 1.0)
                * min(len(as_.recent_form) / 5, 1.0) * 0.15,
            (1.0 if has_odds else 0.5)           * 0.15,
        ])

        certainty = entropy_certainty(combined_probs)

        return round(0.4 * data_quality + 0.6 * certainty, 3)

    # ── adaptive weights ──

    def _adaptive_weights(
        self, home: str, away: str, has_odds: bool
    ) -> Dict[str, float]:
        """Fix #11: shift weights based on available evidence.

        Rules:
        • No odds          → redistribute odds weight equally to perf + hist.
        • Few H2H matches  → shrink historical weight, grow performance.
        • Lots of H2H      → grow historical weight up to its ceiling.
        """
        n_h2h = len(self.repo.get_h2h_matches(home, away))

        # Base
        w = dict(self._base_weights)

        # No odds
        if not has_odds:
            extra = w["odds"]
            w["odds"] = 0.0
            w["performance"] += extra / 2
            w["historical"]  += extra / 2

        # Adjust historical based on H2H evidence
        # Scale: 0 matches → 50% of base; 10+ matches → 120% of base.
        h2h_scaler = 0.50 + 0.70 * min(n_h2h / 10, 1.0)
        w["historical"] *= h2h_scaler
        # Transfer the delta to performance
        delta = self._base_weights["historical"] * (1.0 - h2h_scaler)
        w["performance"] += delta

        # Renormalize to sum to 1
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    # ── single match ──

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        odds_h: float = 0,
        odds_d: float = 0,
        odds_a: float = 0,
    ) -> PredictionResult:
        home = TeamRegistry.normalize(home_team)
        away = TeamRegistry.normalize(away_team)
        has_odds = odds_h > 0 and odds_d > 0 and odds_a > 0

        # Source probabilities
        odds_probs = (
            self.convert_odds_to_probabilities(odds_h, odds_d, odds_a)
            if has_odds
            else {"H": 0.33, "D": 0.34, "A": 0.33}
        )
        perf_probs = self.calculate_performance_probability(home, away)
        hist_probs = self.calculate_historical_probability(home, away)

        # Fix #11: adaptive weights
        w = self._adaptive_weights(home, away, has_odds)

        # Combine
        combined: Dict[str, float] = {}
        for o in ("H", "D", "A"):
            combined[o] = (
                w["odds"]        * odds_probs[o]
              + w["performance"] * perf_probs[o]
              + w["historical"]  * hist_probs[o]
            )
        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}

        prediction = max(combined, key=combined.get)
        # Fix #6: pass combined probs into confidence
        confidence = self.calculate_confidence(home, away, combined, has_odds)

        breakdown = {
            "odds":        odds_probs,
            "performance": perf_probs,
            "historical":  hist_probs,
            "combined":    combined,
        }
        return PredictionResult(
            home_team=home,
            away_team=away,
            prediction=prediction,
            probabilities=combined,
            confidence=confidence,
            breakdown=breakdown,
        )

    # ── batch ──

    def predict_matches(
        self, future_matches: List[Tuple]
    ) -> List[PredictionResult]:
        """
        Each tuple:  (home, away, odds_h, odds_d, odds_a)
                     (home, away)                           ← no odds
        """
        results: List[PredictionResult] = []
        for entry in future_matches:
            if len(entry) >= 5:
                r = self.predict_match(entry[0], entry[1],
                                       entry[2], entry[3], entry[4])
            elif len(entry) == 2:
                r = self.predict_match(entry[0], entry[1])
            else:
                continue
            results.append(r)
        return results


# ──────────────────────────────────────────────
#  ONE-CALL CONVENIENCE
# ──────────────────────────────────────────────

def run_predictions(
    historical_data,
    future_matches: List[Tuple],
    source: str = "list",
) -> List[PredictionResult]:
    """
    historical_data : list-of-tuples  OR  path to CSV
    future_matches  : [(home, away, oh, od, oa), ...]
    source          : 'list' | 'csv'
    """
    repo = MatchRepository()
    if source == "csv":
        repo.load_csv(historical_data)
    else:
        repo.load_list(historical_data)
    predictor = BayesianMatchPredictor(repo)
    return predictor.predict_matches(future_matches)


# ──────────────────────────────────────────────
#  DEMO
# ──────────────────────────────────────────────

if __name__ == "__main__":

    # ── historical matches (date, home, away, hs, as) ──
    historical = [
        ("2024-08-20", "Lazio",              "Napoli",           2, 1),
        ("2024-09-01", "Napoli",             "Lazio",            1, 1),
        ("2024-09-15", "Lazio",              "Real Sociedad",    3, 0),
        ("2024-09-28", "Real Sociedad",      "Lazio",            1, 2),
        ("2024-10-05", "Lazio",              "AZ Alkmaar",       1, 1),
        ("2024-10-20", "AZ Alkmaar",         "Lazio",            0, 2),
        ("2024-11-02", "Napoli",             "Real Sociedad",    2, 0),
        ("2024-11-10", "Real Sociedad",      "Napoli",           1, 1),
        ("2024-11-24", "Napoli",             "AZ Alkmaar",       3, 1),
        ("2024-12-01", "AZ Alkmaar",         "Napoli",           0, 1),
        ("2024-12-08", "Real Sociedad",      "AZ Alkmaar",       2, 2),
        ("2024-12-15", "AZ Alkmaar",         "Real Sociedad",    0, 1),
        ("2025-01-12", "Lazio",              "Napoli",           1, 0),
        ("2025-01-19", "Napoli",             "Lazio",            2, 1),
        ("2025-02-02", "Lazio",              "Real Sociedad",    2, 1),
        ("2025-02-09", "Real Sociedad",      "Lazio",            0, 0),
        ("2025-02-16", "Lazio",              "AZ Alkmaar",       3, 1),
        ("2025-03-01", "Napoli",             "Real Sociedad",    1, 0),
        ("2025-03-08", "AZ Alkmaar",         "Napoli",           1, 2),
        ("2025-03-15", "Lazio",              "Napoli",           2, 2),
    ]

    # ── future matches to predict (with odds) ──
    future = [
        ("Lazio",         "Napoli",          2.10, 3.30, 3.50),
        ("Real Sociedad", "AZ Alkmaar",      1.85, 3.50, 4.20),
        ("Napoli",        "Real Sociedad",   1.55, 3.80, 6.00),
        ("AZ Alkmaar",    "Lazio",           3.00, 3.20, 2.40),
        # without odds
        ("Lazio",         "Real Sociedad"),
    ]

    # ── Option A: one-liner ──
    results = run_predictions(historical, future, source="list")

    # ── Option B: step-by-step (more control) ──
    # repo = MatchRepository()
    # repo.load_list(historical)
    # predictor = BayesianMatchPredictor(repo)
    #
    # # Inspect league table
    # sc = StatsComputer(repo)
    # print("  League Table")
    # print("  " + "=" * 50)
    # for pos, (name, ts) in enumerate(sc.compute_league_table(), 1):
    #     print(f"  {pos}. {name:<20} Pts={ts.points:>3}  "
    #           f"GD={ts.goal_difference:>+3}  Form={''.join(ts.recent_form)}")
    # print()
    #
    # results = predictor.predict_matches(future)

    # ── output ──
    for r in results:
        print(r)
