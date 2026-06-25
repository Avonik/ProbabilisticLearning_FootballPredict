"""
model.py
========
Kern des Rue-Salvesen-Modells (v2):

  - Tormodell mit Trunkierung, Dixon-Coles-Korrektur und Mischung
  - Berechnung der Lambdas aus Teamstärken
  - Bookkeeping über Teams und Spielzeitpunkte

Änderungen gegenüber v1:
  * TAU / GAMMA / EPSILON sind Felder von ``League`` statt globaler Konstanten
    → sie lassen sich pro Fit tunen (siehe ``tune.py``).
  * Zusätzlich zu der dict-basierten Schnittstelle (``team_matches``,
    ``local_idx``) speichert ``League`` flache numpy-Arrays
    (``team_start``, ``team_matches_flat``, ``match_home_local`` …).
    Diese flachen Arrays sind die Eingaben für den Numba-MCMC-Kern.
  * ``use_xg=True`` ersetzt die Beobachtungen (FTHG/FTAG) durch
    gerundete xG-Werte, sofern Spalten ``xG_home`` / ``xG_away`` vorhanden
    sind. Das Likelihoodmodell bleibt identisch, nur das Signal ist sauberer.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from math import lgamma

import numpy as np
import pandas as pd


# === Paper-Defaults (Rue & Salvesen 2000) ===
DEFAULT_TAU = 100.0          # Loss-of-memory Zeitkonstante (Tage)
DEFAULT_GAMMA = 0.1          # Psychologischer Effekt
DEFAULT_EPSILON = 0.2        # Mischungs-Wahrscheinlichkeit
MAX_GOALS = 5                # Trunkierungsgrenze
PRIOR_VAR = 1.0 / 37.0       # Prior-Varianz (Paper)
DEFAULT_PHI = 5.0            # xG-Präzision: Var[xG|λ] = λ²/φ (nur continuous_obs)

# Rückwärtskompatible Aliase (falls externer Code sie noch nutzt)
TAU = DEFAULT_TAU
GAMMA = DEFAULT_GAMMA
EPSILON = DEFAULT_EPSILON


# Log(k!) für schnelle Poisson-PMF
_LOG_FACT = np.array([lgamma(k + 1) for k in range(MAX_GOALS + 1)])


# ─────────────────────────────────────────────────────────────────────
# League-Datenstruktur
# ─────────────────────────────────────────────────────────────────────

@dataclass
class League:
    """Repräsentiert eine Liga / Saison.

    Doppelte Repräsentation:
      * dict-Felder (``team_matches``, ``local_idx``) — kompatibel zu v1,
        wird von Visualisierungen und Animationen verwendet.
      * flache numpy-Arrays (``team_start``, ``team_matches_flat``,
        ``strength_days``, ``match_home_local``, ``match_away_local``) —
        Eingaben für den Numba-MCMC-Kern.
    """
    # --- Stammdaten ---------------------------------------------------
    teams: list[str]
    home_idx: np.ndarray
    away_idx: np.ndarray
    home_goals: np.ndarray            # int — tatsächliche Tore (Heim)
    away_goals: np.ndarray            # int — tatsächliche Tore (Auswärts)
    match_days: np.ndarray            # int — Tage seit Saisonbeginn
    c_x: float
    c_y: float
    team_matches: dict                # team_id -> list[match_idx]  (v1-API)
    local_idx: dict                   # (team_id, match_idx) -> local position
    dates: pd.DatetimeIndex

    # --- Was das Likelihood-Modell tatsächlich sieht ------------------
    obs_x: np.ndarray                 # Heim-Beobachtung: int (Goals/round(xG))
                                      # ODER float (kontinuierliches xG, s. continuous_obs)
    obs_y: np.ndarray                 # Auswärts-Beobachtung (gleicher dtype wie obs_x)

    # --- Numba-freundliche flache Arrays ------------------------------
    team_start: np.ndarray            # int, shape (n_teams+1,)
    team_matches_flat: np.ndarray     # int, shape (sum n_local,)
    strength_days: np.ndarray         # int, shape (sum n_local,)
    match_home_local: np.ndarray      # int, shape (n_matches,)
    match_away_local: np.ndarray      # int, shape (n_matches,)

    # --- Hyperparameter (pro League tunbar) ---------------------------
    tau: float = DEFAULT_TAU
    gamma: float = DEFAULT_GAMMA
    epsilon: float = DEFAULT_EPSILON
    phi: float = DEFAULT_PHI          # xG-Präzision (nur bei continuous_obs)

    # --- Informativer Prior (optional) --------------------------------
    init_prior_mean: np.ndarray | None = None   # shape (n_teams,): Prior-Mittel
                                                # der Initial-Stärke je Team
                                                # (κ·z(log Marktwert)). None/zeros
                                                # ⇒ klassischer N(0,σ²)-Prior.

    # --- xG-Felder (optional) -----------------------------------------
    xg_home: np.ndarray | None = None
    xg_away: np.ndarray | None = None
    use_xg: bool = False
    continuous_obs: bool = False      # True ⇒ Gamma-Likelihood auf rohem xG
                                      # statt Trunc-Poisson auf round(xG)

    @property
    def n_teams(self) -> int:
        return len(self.teams)

    @property
    def n_matches(self) -> int:
        return int(self.home_idx.shape[0])


def build_league(df: pd.DataFrame,
                 use_xg: bool = False,
                 tau: float = DEFAULT_TAU,
                 gamma: float = DEFAULT_GAMMA,
                 epsilon: float = DEFAULT_EPSILON,
                 continuous_xg: bool = False,
                 phi: float = DEFAULT_PHI,
                 team_values: dict | None = None,
                 market_kappa: float = 0.0) -> League:
    """Baut die League-Struktur aus dem Spiele-DataFrame.

    Wenn ``use_xg=True`` und die Spalten ``xG_home`` / ``xG_away`` vorhanden
    sind, werden die zur Likelihood gefütterten Beobachtungen durch
    ``round(xG)`` ersetzt — sonst werden die tatsächlichen Tore verwendet.

    Wenn zusätzlich ``continuous_xg=True``, wird *nicht* gerundet: das rohe
    (kontinuierliche) xG geht als float in die Likelihood, und der MCMC-Kern
    nutzt ein Gamma-Beobachtungsmodell (Präzision ``phi``) statt der
    Trunc-Poisson·Dixon-Coles-PMF. Die Vorhersage bleibt davon unberührt
    (weiterhin diskretes Poisson-Gitter).
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    t2i = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    home_idx = np.array([t2i[t] for t in df["HomeTeam"]], dtype=np.int64)
    away_idx = np.array([t2i[t] for t in df["AwayTeam"]], dtype=np.int64)
    home_goals = df["FTHG"].values.astype(np.int64)
    away_goals = df["FTAG"].values.astype(np.int64)

    date_min = df["Date"].min()
    match_days = np.array([(d - date_min).days for d in df["Date"]],
                          dtype=np.int64)

    # Per-Team chronologische Spiele-Listen
    team_matches: dict[int, list[int]] = {i: [] for i in range(n_teams)}
    for m in range(len(df)):
        team_matches[home_idx[m]].append(m)
        team_matches[away_idx[m]].append(m)

    local_idx: dict[tuple[int, int], int] = {}
    for t, mlist in team_matches.items():
        for k, gm in enumerate(mlist):
            local_idx[(t, gm)] = k

    # Flache Arrays
    n_local = np.array([len(team_matches[t]) for t in range(n_teams)],
                       dtype=np.int64)
    team_start = np.zeros(n_teams + 1, dtype=np.int64)
    team_start[1:] = np.cumsum(n_local)
    total_strengths = int(team_start[-1])

    team_matches_flat = np.zeros(total_strengths, dtype=np.int64)
    strength_days = np.zeros(total_strengths, dtype=np.int64)
    for t in range(n_teams):
        s = team_start[t]
        for k, gm in enumerate(team_matches[t]):
            team_matches_flat[s + k] = gm
            strength_days[s + k] = match_days[gm]

    match_home_local = np.zeros(len(df), dtype=np.int64)
    match_away_local = np.zeros(len(df), dtype=np.int64)
    for m in range(len(df)):
        match_home_local[m] = local_idx[(int(home_idx[m]), m)]
        match_away_local[m] = local_idx[(int(away_idx[m]), m)]

    # xG-Felder, falls vorhanden
    xg_home = xg_away = None
    if "xG_home" in df.columns and "xG_away" in df.columns:
        xg_home = df["xG_home"].values.astype(np.float64)
        xg_away = df["xG_away"].values.astype(np.float64)

    # Welche Beobachtung füttern wir der Likelihood?
    continuous_obs = False
    if use_xg and xg_home is not None:
        if continuous_xg:
            # Rohes, kontinuierliches xG (kein Runden, kein oberes Clip):
            # die Gamma-Likelihood im Kern handhabt reelle, beliebig große Werte.
            obs_x = np.clip(xg_home, 0.0, None).astype(np.float64)
            obs_y = np.clip(xg_away, 0.0, None).astype(np.float64)
            continuous_obs = True
        else:
            obs_x = np.clip(np.round(xg_home), 0, MAX_GOALS).astype(np.int64)
            obs_y = np.clip(np.round(xg_away), 0, MAX_GOALS).astype(np.int64)
        c_x = float(np.log(max(0.05, np.nanmean(xg_home))))
        c_y = float(np.log(max(0.05, np.nanmean(xg_away))))
    else:
        obs_x = home_goals.copy()
        obs_y = away_goals.copy()
        c_x = float(np.log(home_goals.mean()))
        c_y = float(np.log(away_goals.mean()))

    # Informativer Prior aus Marktwerten (Option A):
    # Prior-Mittel der Initial-Stärke je Team = market_kappa · z(log Marktwert),
    # z-standardisiert über die Teams DIESER Liga (Σ≈0 → kompatibel mit der
    # Sum-to-Zero-Projektion im MCMC-Kern). Fehlende/≤0-Werte ⇒ 0 (0-Prior).
    # market_kappa=0 oder team_values=None ⇒ Nullvektor ⇒ bisheriges Verhalten.
    init_prior_mean = np.zeros(n_teams, dtype=np.float64)
    if team_values is not None and market_kappa != 0.0:
        raw = np.array([team_values.get(t, np.nan) for t in teams],
                       dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            logv = np.log(raw)
        mask = np.isfinite(logv)
        if int(mask.sum()) >= 2:
            mu = float(logv[mask].mean())
            sd = float(logv[mask].std())
            if sd > 0.0:
                init_prior_mean[mask] = market_kappa * (logv[mask] - mu) / sd

    return League(
        teams=teams,
        home_idx=home_idx, away_idx=away_idx,
        home_goals=home_goals, away_goals=away_goals,
        match_days=match_days,
        c_x=c_x, c_y=c_y,
        team_matches=team_matches, local_idx=local_idx,
        dates=pd.DatetimeIndex(df["Date"]),
        obs_x=obs_x, obs_y=obs_y,
        team_start=team_start,
        team_matches_flat=team_matches_flat,
        strength_days=strength_days,
        match_home_local=match_home_local,
        match_away_local=match_away_local,
        tau=tau, gamma=gamma, epsilon=epsilon, phi=phi,
        init_prior_mean=init_prior_mean,
        xg_home=xg_home, xg_away=xg_away, use_xg=use_xg,
        continuous_obs=continuous_obs,
    )


# ─────────────────────────────────────────────────────────────────────
# Vorhersagen (Pure-Python, von Visualisierungen genutzt)
# ─────────────────────────────────────────────────────────────────────

def _poisson_pmf_fast(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return float(np.exp(k * np.log(lam) - lam - _LOG_FACT[k]))


def _poisson_tail(max_k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0
    s = 0.0
    log_lam = np.log(lam)
    for k in range(max_k):
        s += np.exp(k * log_lam - lam - _LOG_FACT[k])
    return max(1e-12, 1.0 - s)


def trunc_poisson_pmf(k: int, lam: float, max_k: int = MAX_GOALS) -> float:
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


def compute_lambdas(a_h, d_h, a_a, d_a, c_x, c_y, gamma=DEFAULT_GAMMA):
    """λ_x (Heim) und λ_y (Auswärts) aus Stärken berechnen."""
    delta = (a_h + d_h - a_a - d_a) / 2.0
    log_lam_x = c_x + a_h - d_a - gamma * delta
    log_lam_y = c_y + a_a - d_h + gamma * delta
    return np.exp(log_lam_x), np.exp(log_lam_y)


def match_likelihood(x: int, y: int, lam_x: float, lam_y: float,
                     lam_x_avg: float, lam_y_avg: float,
                     eps: float = DEFAULT_EPSILON) -> float:
    """Mischmodell-Likelihood eines Ergebnisses (x, y)."""
    p1 = (trunc_poisson_pmf(x, lam_x) * trunc_poisson_pmf(y, lam_y) *
          dc_correction(x, y, lam_x, lam_y))
    p2 = (trunc_poisson_pmf(x, lam_x_avg) * trunc_poisson_pmf(y, lam_y_avg) *
          dc_correction(x, y, lam_x_avg, lam_y_avg))
    return (1.0 - eps) * p1 + eps * p2


def predict_outcome_probs(a_h, d_h, a_a, d_a, c_x, c_y,
                          gamma=DEFAULT_GAMMA, eps=DEFAULT_EPSILON,
                          max_k: int = MAX_GOALS):
    """P(Heimsieg), P(Unentschieden), P(Auswärtssieg)."""
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


def predict_match(samples, L, team1: str, team2: str,
                  neutral: bool = True) -> dict:
    """Bayesianische Vorhersage für ein einzelnes Spiel."""
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
        p1, pd_, p2 = predict_outcome_probs(
            a1, d1, a2, d2, c_x, c_y,
            gamma=L.gamma, eps=L.epsilon,
        )
        p1_list.append(p1)
        pd_list.append(pd_)
        p2_list.append(p2)

    result = {
        "team1": team1, "team2": team2,
        "p1": float(np.mean(p1_list)),
        "pd": float(np.mean(pd_list)),
        "p2": float(np.mean(p2_list)),
    }
    venue = "neutralem Platz" if neutral else f"{team1} Heim"
    print(f"\nVorhersage: {team1} vs {team2} ({venue})")
    print(f"  P({team1} gewinnt): {result['p1']:.1%}")
    print(f"  P(Unentschieden):   {result['pd']:.1%}")
    print(f"  P({team2} gewinnt): {result['p2']:.1%}")
    return result


def predict_scoreline(samples, L, team1: str, team2: str,
                      neutral: bool = True) -> np.ndarray:
    """Posteriori-Wahrscheinlichkeitsmatrix aller Ergebnisse."""
    t1 = L.teams.index(team1)
    t2 = L.teams.index(team2)

    c_x, c_y = L.c_x, L.c_y
    if neutral:
        c_x = c_y = (c_x + c_y) / 2

    mat = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for sa, sd in zip(samples["attack"], samples["defense"]):
        a1, d1 = float(sa[t1][-1]), float(sd[t1][-1])
        a2, d2 = float(sa[t2][-1]), float(sd[t2][-1])
        lam_x, lam_y = compute_lambdas(a1, d1, a2, d2, c_x, c_y, gamma=L.gamma)
        lam_x_avg, lam_y_avg = np.exp(c_x), np.exp(c_y)
        for x in range(MAX_GOALS + 1):
            for y in range(MAX_GOALS + 1):
                mat[x, y] += match_likelihood(
                    x, y, lam_x, lam_y, lam_x_avg, lam_y_avg, eps=L.epsilon,
                )
    mat /= mat.sum()
    return mat
