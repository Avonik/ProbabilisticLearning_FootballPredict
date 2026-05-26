"""
model.py
========
Kern des Rue-Salvesen-Modells:
  - Tormodell mit Trunkierung, Dixon-Coles-Korrektur und Mischung
  - Berechnung der Lambdas aus Teamstärken
  - Bookkeeping über Teams und Spielzeitpunkte
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from math import lgamma


# === Globale Parameter (Paper-Werte) ===
TAU = 100.0          # Loss-of-memory Zeitkonstante in Tagen
GAMMA = 0.1          # Psychologischer Effekt
EPSILON = 0.2        # Mischungs-Wahrscheinlichkeit
MAX_GOALS = 5        # Trunkierungsgrenze für Tore
PRIOR_VAR = 1 / 37   # Prior-Varianz für Stärken (aus Paper)


# Vorberechnete log(k!) für schnelle Poisson-PMF
_LOG_FACT = np.array([lgamma(k + 1) for k in range(MAX_GOALS + 1)])


def _poisson_pmf_fast(k: int, lam: float) -> float:
    """Schnelle Poisson-PMF ohne scipy."""
    if lam <= 0:
        return 0.0
    return np.exp(k * np.log(lam) - lam - _LOG_FACT[k])


def _poisson_tail(max_k: int, lam: float) -> float:
    """P(X >= max_k) für Poisson — direkt aus PMFs aufsummiert."""
    if lam <= 0:
        return 1.0 if max_k == 0 else 0.0
    # 1 - sum_{k=0}^{max_k-1} pmf(k)
    s = 0.0
    log_lam = np.log(lam)
    for k in range(max_k):
        s += np.exp(k * log_lam - lam - _LOG_FACT[k])
    return max(1e-12, 1.0 - s)


@dataclass
class League:
    """Repräsentiert eine Liga / Saison."""
    teams: list[str]                # Teamnamen
    home_idx: np.ndarray            # Spielindex -> Team-ID Heim
    away_idx: np.ndarray            # Spielindex -> Team-ID Auswärts
    home_goals: np.ndarray          # Tore Heim pro Spiel
    away_goals: np.ndarray          # Tore Auswärts pro Spiel
    match_days: np.ndarray          # Tag-seit-Saisonstart pro Spiel
    c_x: float                      # log mittl. Heimtore
    c_y: float                      # log mittl. Auswärtstore
    # Für jedes Team: chronologisch sortierte Liste von Spielindizes
    team_matches: dict[int, list[int]]
    # Reverse: (team, global_match_idx) -> Position in team_matches[team]
    local_idx: dict[tuple[int, int], int]
    dates: pd.DatetimeIndex


def build_league(df: pd.DataFrame) -> League:
    """Baut die League-Struktur aus dem Spiele-DataFrame."""
    df = df.copy().sort_values("Date").reset_index(drop=True)

    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    t2i = {t: i for i, t in enumerate(teams)}

    home_idx = np.array([t2i[t] for t in df["HomeTeam"]])
    away_idx = np.array([t2i[t] for t in df["AwayTeam"]])
    home_goals = df["FTHG"].values.astype(int)
    away_goals = df["FTAG"].values.astype(int)

    date_min = df["Date"].min()
    match_days = np.array([(d - date_min).days for d in df["Date"]])

    c_x = float(np.log(home_goals.mean()))
    c_y = float(np.log(away_goals.mean()))

    team_matches: dict[int, list[int]] = {i: [] for i in range(len(teams))}
    for m in range(len(df)):
        team_matches[home_idx[m]].append(m)
        team_matches[away_idx[m]].append(m)

    local_idx: dict[tuple[int, int], int] = {}
    for team, mlist in team_matches.items():
        for k, gm in enumerate(mlist):
            local_idx[(team, gm)] = k

    return League(
        teams=teams,
        home_idx=home_idx, away_idx=away_idx,
        home_goals=home_goals, away_goals=away_goals,
        match_days=match_days,
        c_x=c_x, c_y=c_y,
        team_matches=team_matches,
        local_idx=local_idx,
        dates=pd.DatetimeIndex(df["Date"]),
    )


def compute_lambdas(a_h, d_h, a_a, d_a, c_x, c_y, gamma=GAMMA):
    """
    Berechnet (λ_x, λ_y) für ein Spiel: Heim-Tore und Auswärtstore.

        log λ_x = c_x + a_h - d_a - γ Δ
        log λ_y = c_y + a_a - d_h + γ Δ
        Δ       = (a_h + d_h - a_a - d_a) / 2
    """
    delta = (a_h + d_h - a_a - d_a) / 2
    log_lam_x = c_x + a_h - d_a - gamma * delta
    log_lam_y = c_y + a_a - d_h + gamma * delta
    return np.exp(log_lam_x), np.exp(log_lam_y)


def trunc_poisson_pmf(k: int, lam: float, max_k: int = MAX_GOALS) -> float:
    """PMF der trunkierten Poisson-Verteilung (k=max_k absorbiert Tail)."""
    if k < 0 or k > max_k:
        return 0.0
    if k < max_k:
        return _poisson_pmf_fast(k, lam)
    return _poisson_tail(max_k, lam)


def dc_correction(x: int, y: int, lam_x: float, lam_y: float) -> float:
    """Dixon-Coles-Korrekturfaktor κ."""
    if x == 0 and y == 0:
        return 1.0 + 0.1 * lam_x * lam_y
    if x == 0 and y == 1:
        return max(1e-6, 1.0 - 0.1 * lam_x)
    if x == 1 and y == 0:
        return max(1e-6, 1.0 - 0.1 * lam_y)
    if x == 1 and y == 1:
        return 1.1
    return 1.0


def match_likelihood(x: int, y: int, lam_x: float, lam_y: float,
                     lam_x_avg: float, lam_y_avg: float,
                     eps: float = EPSILON) -> float:
    """
    Likelihood eines Ergebnisses (x, y) unter dem Mischungsmodell:
        π_g = (1-ε)·π*_g1(x,y | λ_x, λ_y) + ε·π*_g1(x,y | λ_x_avg, λ_y_avg)
    """
    p1 = (trunc_poisson_pmf(x, lam_x) * trunc_poisson_pmf(y, lam_y) *
          dc_correction(x, y, lam_x, lam_y))
    p2 = (trunc_poisson_pmf(x, lam_x_avg) * trunc_poisson_pmf(y, lam_y_avg) *
          dc_correction(x, y, lam_x_avg, lam_y_avg))
    return (1.0 - eps) * p1 + eps * p2


def predict_match(samples, L, team1: str, team2: str,
                  neutral: bool = True) -> dict:
    """
    Bayesianische Vorhersage für ein Spiel aus MCMC-Samples.

    Mittelt predict_outcome_probs über alle Posterior-Samples.
    Nutzt die Stärken vom letzten bekannten Spieltag jedes Teams.

    neutral=True entfernt den Heimvorteil (z.B. Pokalfinale in Berlin).
    """
    if team1 not in L.teams:
        raise ValueError(f"'{team1}' nicht gefunden. Teams: {L.teams}")
    if team2 not in L.teams:
        raise ValueError(f"'{team2}' nicht gefunden. Teams: {L.teams}")

    t1 = L.teams.index(team1)
    t2 = L.teams.index(team2)

    c_x, c_y = L.c_x, L.c_y
    if neutral:
        c_x = c_y = (c_x + c_y) / 2

    p1_list, pd_list, p2_list = [], [], []
    for sa, sd in zip(samples["attack"], samples["defense"]):
        a1, d1 = float(sa[t1][-1]), float(sd[t1][-1])
        a2, d2 = float(sa[t2][-1]), float(sd[t2][-1])
        p1, pd, p2 = predict_outcome_probs(a1, d1, a2, d2, c_x, c_y)
        p1_list.append(p1)
        pd_list.append(pd)
        p2_list.append(p2)

    result = {
        "team1": team1, "team2": team2,
        "p1":    float(np.mean(p1_list)),
        "pd":    float(np.mean(pd_list)),
        "p2":    float(np.mean(p2_list)),
    }

    venue = "neutralem Platz" if neutral else f"{team1} Heim"
    print(f"\nVorhersage: {team1} vs {team2} ({venue})")
    print(f"  P({team1} gewinnt): {result['p1']:.1%}")
    print(f"  P(Unentschieden):   {result['pd']:.1%}")
    print(f"  P({team2} gewinnt): {result['p2']:.1%}")
    return result


def predict_scoreline(samples, L, team1: str, team2: str,
                      neutral: bool = True) -> np.ndarray:
    """
    Posteriori-Wahrscheinlichkeitsmatrix aller Ergebnisse (team1-Tore x team2-Tore).
    Zeilen = team1-Tore (0..MAX_GOALS), Spalten = team2-Tore.
    """
    t1 = L.teams.index(team1)
    t2 = L.teams.index(team2)

    c_x, c_y = L.c_x, L.c_y
    if neutral:
        c_x = c_y = (c_x + c_y) / 2

    mat = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for sa, sd in zip(samples["attack"], samples["defense"]):
        a1, d1 = float(sa[t1][-1]), float(sd[t1][-1])
        a2, d2 = float(sa[t2][-1]), float(sd[t2][-1])
        lam_x, lam_y = compute_lambdas(a1, d1, a2, d2, c_x, c_y)
        lam_x_avg, lam_y_avg = np.exp(c_x), np.exp(c_y)
        for x in range(MAX_GOALS + 1):
            for y in range(MAX_GOALS + 1):
                mat[x, y] += match_likelihood(x, y, lam_x, lam_y,
                                              lam_x_avg, lam_y_avg)
    mat /= mat.sum()
    return mat


def predict_outcome_probs(a_h, d_h, a_a, d_a, c_x, c_y,
                          gamma=GAMMA, eps=EPSILON,
                          max_k: int = MAX_GOALS) -> tuple[float, float, float]:
    """
    P(Heimsieg), P(Unentschieden), P(Auswärtssieg) — durch Aufsummieren
    über alle Score-Kombinationen 0..max_k.
    """
    lam_x, lam_y = compute_lambdas(a_h, d_h, a_a, d_a, c_x, c_y, gamma)
    lam_x_avg, lam_y_avg = np.exp(c_x), np.exp(c_y)

    p_home = p_draw = p_away = 0.0
    norm = 0.0
    for x in range(max_k + 1):
        for y in range(max_k + 1):
            p = match_likelihood(x, y, lam_x, lam_y, lam_x_avg, lam_y_avg, eps)
            norm += p
            if x > y:
                p_home += p
            elif x == y:
                p_draw += p
            else:
                p_away += p
    return p_home / norm, p_draw / norm, p_away / norm
