"""
evaluation.py
=============
Bewertung der Vorhersagequalität.

Implementiert:
  - **Ranked Probability Score (RPS)** — der Goldstandard für 3-Klassen-
    Fußballvorhersagen (Constantinou & Fenton 2012). Niedriger = besser.
  - **Log-Loss** — Standard für probabilistische Klassifikation.
  - **Brier-Score** — quadratischer Mittelwert der Klassifikationsfehler.
  - **Kalibrierungsanalyse** — vorhergesagte vs. empirische Wahrscheinlichkeiten.
  - **Buchmacher-Vergleich** — wie schlägt sich das Modell gegen Pinnacle / Bet365?

Baselines:
  - **Uniform (1/3, 1/3, 1/3)** — kein Wissen.
  - **Empirische Basisrate** — historische Heimsieg-/Remis-/Auswärtssieg-Verteilung.
  - **Buchmacher** — implizite Wahrscheinlichkeiten aus Schlussquoten.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model import predict_outcome_probs, MAX_GOALS
from data import sort_matches


# ─────────────────────────────────────────────────────────────────────
# Skalare Metriken pro Spiel
# ─────────────────────────────────────────────────────────────────────

def rps_one(probs: np.ndarray, outcome_idx: int) -> float:
    """
    Ranked Probability Score für ein 3-Klassen-Outcome (Heim/Remis/Auswärts).

    Formel (Epstein 1969 / Constantinou & Fenton 2012):

        RPS  =  (1/(r-1)) · Σ_{k=1..r-1} ( Σ_{j=1..k} (p_j - o_j) )²

    Für r=3 also:
        RPS = ½ · [ (p_H - o_H)² + (p_H + p_D - o_H - o_D)² ]

    Bereich: [0, 1]. Niedriger = bessere kalibrierte Vorhersage.
    """
    p = np.asarray(probs, dtype=float)
    o = np.zeros(3)
    o[outcome_idx] = 1.0
    cum_p = np.cumsum(p)
    cum_o = np.cumsum(o)
    return float(0.5 * ((cum_p[0] - cum_o[0]) ** 2 + (cum_p[1] - cum_o[1]) ** 2))


def log_loss_one(probs: np.ndarray, outcome_idx: int, eps: float = 1e-12) -> float:
    """−log P(beobachtetes Ergebnis). Niedriger = besser."""
    return float(-np.log(max(probs[outcome_idx], eps)))


def brier_one(probs: np.ndarray, outcome_idx: int) -> float:
    """Brier-Score = Σ_k (p_k - o_k)². Niedriger = besser."""
    o = np.zeros(3)
    o[outcome_idx] = 1.0
    return float(np.sum((np.asarray(probs) - o) ** 2))


# ─────────────────────────────────────────────────────────────────────
# Vorhersage aus MCMC-Samples
# ─────────────────────────────────────────────────────────────────────

def predict_match_from_samples(samples, L, home_team: str, away_team: str,
                               last_local: bool = True) -> np.ndarray:
    """Posteriori-Mittel der Outcome-Wahrscheinlichkeiten für ein Spiel.

    Args:
        last_local: Wenn True, nutzt die Stärken am Saisonende
            (letzte verfügbare Stärke pro Team).
    """
    if home_team not in L.teams or away_team not in L.teams:
        return np.array([np.nan, np.nan, np.nan])

    h = L.teams.index(home_team)
    a = L.teams.index(away_team)
    p_h_list, p_d_list, p_a_list = [], [], []
    home_samples = samples.get("home_advantage")
    for sample_i, (sa, sd) in enumerate(zip(samples["attack"], samples["defense"])):
        a_h = float(sa[h][-1] if last_local else sa[h].mean())
        d_h = float(sd[h][-1] if last_local else sd[h].mean())
        a_a = float(sa[a][-1] if last_local else sa[a].mean())
        d_a = float(sd[a][-1] if last_local else sd[a].mean())
        home_adv = (float(home_samples[sample_i][h])
                    if home_samples is not None else 0.0)
        p_h, p_d, p_a = predict_outcome_probs(
            a_h, d_h, a_a, d_a, L.c_x, L.c_y,
            gamma=L.gamma, eps=L.epsilon, max_k=MAX_GOALS,
            home_advantage=home_adv,
        )
        p_h_list.append(p_h)
        p_d_list.append(p_d)
        p_a_list.append(p_a)
    return np.array([np.mean(p_h_list), np.mean(p_d_list), np.mean(p_a_list)])


def predict_match_at_local(samples, L, home_team: str, away_team: str,
                           home_local: int, away_local: int) -> np.ndarray:
    """Vorhersage mit Stärken zu spezifischen Spieltagen (für Replay-Eval)."""
    h = L.teams.index(home_team)
    a = L.teams.index(away_team)
    p_h_list, p_d_list, p_a_list = [], [], []
    home_samples = samples.get("home_advantage")
    for sample_i, (sa, sd) in enumerate(zip(samples["attack"], samples["defense"])):
        a_h = float(sa[h][home_local])
        d_h = float(sd[h][home_local])
        a_a = float(sa[a][away_local])
        d_a = float(sd[a][away_local])
        home_adv = (float(home_samples[sample_i][h])
                    if home_samples is not None else 0.0)
        p_h, p_d, p_a = predict_outcome_probs(
            a_h, d_h, a_a, d_a, L.c_x, L.c_y,
            gamma=L.gamma, eps=L.epsilon, max_k=MAX_GOALS,
            home_advantage=home_adv,
        )
        p_h_list.append(p_h)
        p_d_list.append(p_d)
        p_a_list.append(p_a)
    return np.array([np.mean(p_h_list), np.mean(p_d_list), np.mean(p_a_list)])


# ─────────────────────────────────────────────────────────────────────
# Aggregierte Eval über mehrere Spiele
# ─────────────────────────────────────────────────────────────────────

def outcome_index(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def evaluate_predictions(probs_pred: np.ndarray, outcomes: np.ndarray) -> dict:
    """
    Berechnet RPS / Log-Loss / Brier für eine Menge von Vorhersagen.

    probs_pred: shape (n, 3)
    outcomes:   shape (n,)  mit Werten in {0, 1, 2}
    """
    rps_vals, ll_vals, brier_vals = [], [], []
    valid_mask = np.all(np.isfinite(probs_pred), axis=1)
    valid_probs = probs_pred[valid_mask]
    valid_out = outcomes[valid_mask]
    for p, o in zip(valid_probs, valid_out):
        rps_vals.append(rps_one(p, int(o)))
        ll_vals.append(log_loss_one(p, int(o)))
        brier_vals.append(brier_one(p, int(o)))
    return {
        "rps": float(np.mean(rps_vals)) if rps_vals else np.nan,
        "log_loss": float(np.mean(ll_vals)) if ll_vals else np.nan,
        "brier": float(np.mean(brier_vals)) if brier_vals else np.nan,
        "n": int(valid_mask.sum()),
    }


def evaluate_season(samples, L, df_season: pd.DataFrame,
                    holdout_frac: float = 0.3) -> dict:
    """
    In-Sample-Eval: Vorhersagen für jeden Spieltag mit den Stärken *vor*
    dem Spiel (also dem letzten verfügbaren Sample für jedes Team aus den
    vorherigen Spielen).

    Args:
        holdout_frac: Anteil der späten Saison, der als Holdout bewertet wird.
            (Nur diese späten Spiele fließen in die Metriken, weil frühe
            Spiele zu nah am Prior liegen.)

    Returns: dict mit Metriken sowie ``probs_pred`` und ``outcomes``.
    """
    df = sort_matches(df_season)
    n = len(df)
    cutoff = int(n * (1.0 - holdout_frac))

    probs = np.full((n, 3), np.nan)
    outcomes = np.zeros(n, dtype=int)

    for m in range(n):
        row = df.iloc[m]
        h_name, a_name = row["HomeTeam"], row["AwayTeam"]
        if h_name not in L.teams or a_name not in L.teams:
            outcomes[m] = outcome_index(int(row["FTHG"]), int(row["FTAG"]))
            continue

        h = L.teams.index(h_name)
        a = L.teams.index(a_name)
        # Stärken VOR Spiel m: letzter local-Idx < m
        h_locals = [k for k, gm in enumerate(L.team_matches[h]) if gm < m]
        a_locals = [k for k, gm in enumerate(L.team_matches[a]) if gm < m]
        if not h_locals or not a_locals:
            outcomes[m] = outcome_index(int(row["FTHG"]), int(row["FTAG"]))
            continue
        hl = h_locals[-1]
        al = a_locals[-1]
        p = predict_match_at_local(samples, L, h_name, a_name, hl, al)
        probs[m] = p
        outcomes[m] = outcome_index(int(row["FTHG"]), int(row["FTAG"]))

    metrics = evaluate_predictions(probs[cutoff:], outcomes[cutoff:])
    return {
        "metrics": metrics,
        "probs": probs,
        "outcomes": outcomes,
        "cutoff": cutoff,
        "n": n,
    }


# ─────────────────────────────────────────────────────────────────────
# Buchmacher- und Baseline-Vergleich
# ─────────────────────────────────────────────────────────────────────

def bookmaker_probs_from_df(df_season: pd.DataFrame) -> np.ndarray:
    """Extrahiert die normalisierten Buchmacherwahrscheinlichkeiten."""
    n = len(df_season)
    probs = np.full((n, 3), np.nan)
    for i, (_, row) in enumerate(df_season.iterrows()):
        if pd.notna(row.get("bm_pH")) and pd.notna(row.get("bm_pD")) and pd.notna(row.get("bm_pA")):
            probs[i] = [row["bm_pH"], row["bm_pD"], row["bm_pA"]]
    return probs


def uniform_baseline(n: int) -> np.ndarray:
    """Konstante (1/3, 1/3, 1/3)-Vorhersage."""
    return np.tile([1.0 / 3, 1.0 / 3, 1.0 / 3], (n, 1))


def empirical_baseline(df_history: pd.DataFrame, n_pred: int) -> np.ndarray:
    """Konstante Vorhersage aus historischen Basisraten."""
    h = (df_history["FTHG"] > df_history["FTAG"]).mean()
    d = (df_history["FTHG"] == df_history["FTAG"]).mean()
    a = 1.0 - h - d
    return np.tile([h, d, a], (n_pred, 1))


def compare_to_bookmaker(samples, L, df_season: pd.DataFrame,
                         df_history: pd.DataFrame | None = None,
                         holdout_frac: float = 0.3) -> dict:
    """
    Vergleicht Modell vs Buchmacher vs Baselines auf der Holdout-Hälfte.

    Returns ein dict mit Schlüsseln 'model', 'bookmaker', 'uniform', 'empirical',
    jeweils mit Metrics-dict.
    """
    df = sort_matches(df_season)
    n = len(df)
    cutoff = int(n * (1.0 - holdout_frac))

    # Modell-Vorhersagen + tatsächliche Ergebnisse

    # Only slice if the value has a length (is an array/list)
    samples_sub = {
        k: (v[::100] if hasattr(v, "__len__") else v)
        for k, v in samples.items()
    }

    eval_out = evaluate_season(samples_sub, L, df, holdout_frac=holdout_frac)
    probs_model = eval_out["probs"]
    outcomes = eval_out["outcomes"]

    # Buchmacher
    probs_bm = bookmaker_probs_from_df(df)

    # Baselines
    probs_uni = uniform_baseline(n)
    if df_history is None or len(df_history) == 0:
        probs_emp = uniform_baseline(n)
    else:
        probs_emp = empirical_baseline(df_history, n)

    # Holdout-Slice
    slc = slice(cutoff, n)

    return {
        "model":     evaluate_predictions(probs_model[slc], outcomes[slc]),
        "bookmaker": evaluate_predictions(probs_bm[slc],    outcomes[slc]),
        "uniform":   evaluate_predictions(probs_uni[slc],   outcomes[slc]),
        "empirical": evaluate_predictions(probs_emp[slc],   outcomes[slc]),
        "probs_model":     probs_model,
        "probs_bookmaker": probs_bm,
        "outcomes":        outcomes,
        "cutoff":          cutoff,
        "n":               n,
    }


# ─────────────────────────────────────────────────────────────────────
# Kalibrierung
# ─────────────────────────────────────────────────────────────────────

def calibration_bins(probs: np.ndarray, outcomes: np.ndarray,
                     n_bins: int = 10) -> dict:
    """
    Kalibrierungsanalyse: für jedes 3-Klassen-Ergebnis getrennt.

    Für jede Klasse k ∈ {0,1,2}:
      - Binne nach vorhergesagter Wahrscheinlichkeit p_k in [0,1]
      - Vergleiche mittlere p_k im Bin mit empirischer Rate von outcome=k

    Returns: dict pro Klasse mit 'bin_centers', 'mean_pred', 'empirical', 'n'.
    """
    valid = np.all(np.isfinite(probs), axis=1)
    p = probs[valid]
    y = outcomes[valid]
    edges = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    result = {}
    for k in range(3):
        pk = p[:, k]
        yk = (y == k).astype(float)
        bins = np.digitize(pk, edges) - 1
        bins = np.clip(bins, 0, n_bins - 1)
        mean_pred = np.full(n_bins, np.nan)
        emp = np.full(n_bins, np.nan)
        ns = np.zeros(n_bins, dtype=int)
        for b in range(n_bins):
            mask = bins == b
            if mask.sum() > 0:
                mean_pred[b] = pk[mask].mean()
                emp[b] = yk[mask].mean()
                ns[b] = mask.sum()
        result[k] = {
            "bin_centers": centers,
            "mean_pred": mean_pred,
            "empirical": emp,
            "n": ns,
        }
    return result
