"""
run.py
======
Hauptpipeline v2 (RETROSPEKTIV): lädt Daten, kalibriert xG, tunt
Hyperparameter, fitted das Modell mit Numba-MCMC auf die *ganze* Saison
und erzeugt alle Visualisierungen + Animationen.

Der Voll-Saison-Fit ist eine glättende (retrospektive) Schätzung und damit
NICHT für eine ehrliche Vorhersage-Evaluation geeignet. Der Buchmacher-
Vergleich (Walk-Forward-RPS) lebt deshalb separat in ``backtest.py``.

Phasen
------
  1  Daten laden (inkl. Schussdaten + Buchmacherquoten)
  2  xG kalibrieren (β_off, β_on aus Schüssen → Tore)
  3  Datenexploration (Tore vs. xG, Heimvorteil)
  4  Modell-Intuition (Brownsche Bewegung, Tor-Modell, DAG)
  5  Hyperparameter-Tuning (τ, γ, ε via Holdout-Validierung)
  6  Haupt-MCMC mit getunten Parametern (parallele Ketten)
  7  Vorhersage-Visualisierungen
  8  Animationen

→ Ehrlicher Buchmacher-Vergleich: ``python backtest.py``
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

# UTF-8-Konsole auch unter Windows (sonst UnicodeEncodeError bei τ, →, ⌀, ...)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

import numpy as np
import pandas as pd

from data import load_bundesliga, extract_bookmaker_probs
from xg import fit_xg_weights, add_xg_columns, xg_summary
from model import build_league
from parallel import run_mcmc_parallel
from tune import tune_hyperparameters, tune_paperstyle
from viz import (
    plot_goal_distribution,
    plot_home_advantage_over_time,
    plot_brownian_motion_examples,
    plot_tor_modell_intuition,
    plot_dag_structure,
    plot_mcmc_trace,
    plot_strength_evolution,
    plot_predicted_rankings,
    plot_surprise_matches,
    plot_team_ranking_summary,
    plot_xg_vs_goals,
    plot_hyperparameter_scan,
)
from viz_interactive import (
    plot_strength_evolution_html,
    plot_mcmc_trace_html,
    plot_team_profile_html,
)
from animations import (
    animate_strength_evolution,
    animate_season_table,
    animate_mcmc_walk,
)


# === Konfiguration ===
START_YEAR = 1993          # football-data.co.uk hat ab 1993/94 Daten
END_YEAR   = 2026          # exklusiv  → bis Saison 2025/26
ANALYSIS_SEASON = "2025/26"

# xG verwenden? (True → gerundetes xG füttert die Likelihood)
USE_XG = True

# Hyperparameter-Tuning
#   "paper":  Multi-Season-Tuning analog Rue & Salvesen 2000, Abschnitt 3.2
#             (5+ Saisons, ~750 Holdout-Matches; ~3–5 min, stabil)
#   "single": Schnelles 1-Saison-Tuning (~30 s, aber Ergebnis wackelt)
#   "off":    Paper-Defaults verwenden (τ=100, γ=0.10, ε=0.20)
TUNE_MODE     = "paper"
TUNE_TAUS     = (40.0, 80.0, 120.0, 200.0)
TUNE_GAMMAS   = (0.0, 0.05, 0.10, 0.20)
TUNE_EPSILONS = (0.05, 0.15, 0.25)
TUNE_N_ITER   = 6000     # genug Iterationen, da xG-Signal sauber
TUNE_BURNIN   = 2000
TUNE_THIN     = 10
# Wie viele Saisons als Tuning-Historie? (nur für TUNE_MODE="paper")
# Paper nutzte 4 Saisons × 2 Ligen = 3892 Matches; 5 BL-Saisons ≈ 1500 Matches
TUNE_HISTORY_N_SEASONS = 5

# === Haupt-MCMC ===
# Paper, Abschnitt 4: 100 000 Iterationen für Prediction, 1 000 000 für
# retrospektive Analyse — Single Chain. Mit Numba (~5000 iter/s) ist
# sogar die retrospektive Einstellung in wenigen Minuten möglich.
#
# Empfohlene Werte (Kompromiss aus Vorhersage- und Retro-Use-Case):
#   4 Ketten × 500 000 Iter ≈ 2 min wallclock, R-hat sicher unter 1.01.
# Mehr Ketten als 4 bringen seit Numba kaum noch Speedup (wallclock-
# gebunden statt Anzahl-gebunden); 4 reichen für R-hat-Diagnostik.
N_CHAINS     = 4
N_ITER       = 500_000
BURNIN       = 50_000     # 10 % — sicher über den User-R-hat-Befund hinaus
THIN         = 25          # 18 000 Samples pro Kette × 4 = 72 000 — reicht
PROPOSAL_SD  = 0.06        # Paper ~55 % Akzeptanz; 0.06 → ~50 % (gut)
SEED         = 42

OUT = Path("output")
OUT.mkdir(exist_ok=True)

CACHE_DIR = Path("cache_v2")
CACHE_DIR.mkdir(exist_ok=True)
MCMC_CACHE = CACHE_DIR / f"mcmc_{ANALYSIS_SEASON.replace('/','_')}_{N_CHAINS}c_{N_ITER}i_{'xg' if USE_XG else 'goals'}.pkl"
TUNE_CACHE = CACHE_DIR / f"tune_{TUNE_MODE}_{ANALYSIS_SEASON.replace('/','_')}_{'xg' if USE_XG else 'goals'}.pkl"


def banner(title: str):
    print("\n" + "=" * 64)
    print(f" {title}")
    print("=" * 64)


def main():
    t_start = time.time()

    # ─── Phase 1: Daten laden ────────────────────────────────────────
    banner("PHASE 1: Datenladung (inkl. Schüsse + Buchmacherquoten)")
    df = load_bundesliga(START_YEAR, END_YEAR, with_extras=True)
    df = extract_bookmaker_probs(df)
    print(f"  Spalten: {list(df.columns)[:12]} ... ({len(df.columns)} total)")

    # ─── Phase 2: xG kalibrieren ─────────────────────────────────────
    banner("PHASE 2: xG-Kalibrierung (Proxy aus Schussdaten)")
    beta_off, beta_on = fit_xg_weights(df)
    df = add_xg_columns(df, beta_off, beta_on)
    summary = xg_summary(df)
    if summary:
        print(f"  ⌀ Tore Heim:    {summary['mean_goals_home']:.3f}  | "
              f"⌀ xG Heim:    {summary['mean_xg_home']:.3f}")
        print(f"  ⌀ Tore Auswärts:{summary['mean_goals_away']:.3f}  | "
              f"⌀ xG Auswärts:{summary['mean_xg_away']:.3f}")
        print(f"  corr(xG, Tore) Heim    = {summary['corr_xg_goals_home']:.3f}")
        print(f"  corr(xG, Tore) Auswärts= {summary['corr_xg_goals_away']:.3f}")
        print(f"  {summary['n_with_xg']}/{summary['n_total']} Spiele mit xG")

    # ─── Phase 3: Datenexploration ───────────────────────────────────
    banner("PHASE 3: Datenexploration über alle Saisons")
    print("  → Tor-Verteilung ...")
    plot_goal_distribution(df, OUT / "01_tor_verteilung.png")
    print("  → Heimvorteil-Entwicklung ...")
    plot_home_advantage_over_time(df, OUT / "02_heimvorteil_2000_2026.png")

    # ─── Phase 4: Modell-Intuition ───────────────────────────────────
    banner("PHASE 4: Modellbausteine (Konzept-Visualisierungen)")
    print("  → Brownsche Bewegungspfade ...")
    plot_brownian_motion_examples(OUT / "03_brownsche_bewegung.png")
    print("  → Tor-Modell-Aufbau ...")
    plot_tor_modell_intuition(OUT / "04_tormodell_aufbau.png")
    print("  → DAG der Modellstruktur ...")
    plot_dag_structure(OUT / "05_dag_struktur.png")

    # ─── Saison-Subset ───────────────────────────────────────────────
    df_season = df[df["Season"] == ANALYSIS_SEASON].copy()
    if len(df_season) == 0:
        fallback = sorted(df["Season"].unique())[-1]
        print(f"  Saison '{ANALYSIS_SEASON}' nicht gefunden, "
              f"verwende {fallback}")
        df_season = df[df["Season"] == fallback].copy()
    print(f"\n  Analyse-Saison: {ANALYSIS_SEASON} — {len(df_season)} Spiele, "
          f"{len(set(df_season['HomeTeam']) | set(df_season['AwayTeam']))} Teams")

    # xG-Plot für die Analyse-Saison
    print("  → xG vs. Tore (Saison-Plot) ...")
    plot_xg_vs_goals(df_season, OUT / "11_xg_vs_tore.png",
                     beta_off=beta_off, beta_on=beta_on)

    # ─── Phase 5: Hyperparameter-Tuning ──────────────────────────────
    banner(f"PHASE 5: Hyperparameter-Tuning (Modus: {TUNE_MODE})")
    if TUNE_MODE == "off":
        best_tau, best_gamma, best_eps = 100.0, 0.10, 0.20
        tune_result = None
        print("  (Tuning deaktiviert — verwende Paper-Defaults)")
    elif TUNE_CACHE.exists():
        print(f"  Lade Tuning-Cache: {TUNE_CACHE.name}")
        with open(TUNE_CACHE, "rb") as f:
            tune_result = pickle.load(f)
        best = tune_result["best"]
        best_tau, best_gamma, best_eps = (float(best["tau"]),
                                          float(best["gamma"]),
                                          float(best["epsilon"]))
    else:
        if TUNE_MODE == "paper":
            # Saisons mit xG-Daten und VOR der Analyse-Saison
            all_seasons = sorted(df["Season"].unique())
            analysis_idx = all_seasons.index(ANALYSIS_SEASON) \
                if ANALYSIS_SEASON in all_seasons else len(all_seasons)
            history = all_seasons[max(0, analysis_idx - TUNE_HISTORY_N_SEASONS):
                                  analysis_idx]
            # Wenn xG genutzt wird: nur Saisons mit Schussdaten
            if USE_XG:
                history = [s for s in history
                           if df[(df["Season"] == s) & df.get("has_xg", False)].shape[0] > 50]
                history = history[-TUNE_HISTORY_N_SEASONS:]
            print(f"  Paper-Style Tuning auf {len(history)} Saisons: {history}")
            tune_result = tune_paperstyle(
                df, history_seasons=history, use_xg=USE_XG,
                taus=TUNE_TAUS, gammas=TUNE_GAMMAS, epsilons=TUNE_EPSILONS,
                n_iter=TUNE_N_ITER, burnin=TUNE_BURNIN, thin=TUNE_THIN,
                proposal_sd=PROPOSAL_SD, seed=SEED,
            )
        elif TUNE_MODE == "single":
            print("  Single-Season Tuning ...")
            tune_result = tune_hyperparameters(
                df_season, use_xg=USE_XG,
                taus=TUNE_TAUS, gammas=TUNE_GAMMAS, epsilons=TUNE_EPSILONS,
                n_iter=TUNE_N_ITER, burnin=TUNE_BURNIN, thin=TUNE_THIN,
                proposal_sd=PROPOSAL_SD, seed=SEED,
            )
        else:
            raise ValueError(f"Unbekannter TUNE_MODE: {TUNE_MODE}")

        best = tune_result["best"]
        best_tau, best_gamma, best_eps = (float(best["tau"]),
                                          float(best["gamma"]),
                                          float(best["epsilon"]))
        with open(TUNE_CACHE, "wb") as f:
            pickle.dump(tune_result, f)

    if tune_result is not None:
        print(f"  → Hyperparameter-Scan ...")
        plot_hyperparameter_scan(tune_result, OUT / "12_hyperparameter_scan.png")
    print(f"  → Modell verwendet τ={best_tau}, γ={best_gamma}, ε={best_eps}")

    # ─── Phase 6: Haupt-MCMC ─────────────────────────────────────────
    banner(f"PHASE 6: MCMC-Inferenz auf Saison {ANALYSIS_SEASON}")
    L = build_league(df_season, use_xg=USE_XG,
                     tau=best_tau, gamma=best_gamma, epsilon=best_eps)
    print(f"  League: {L.n_teams} Teams, {L.n_matches} Spiele, "
          f"{int(L.team_start[-1])} Stärken-Knoten")
    print(f"  Beobachtung: {'gerundetes xG' if USE_XG else 'tatsächliche Tore'}")

    if MCMC_CACHE.exists():
        print(f"  Lade MCMC-Cache: {MCMC_CACHE.name}")
        with open(MCMC_CACHE, "rb") as f:
            samples = pickle.load(f)
    else:
        t0 = time.time()
        samples = run_mcmc_parallel(
            L, n_chains=N_CHAINS, n_iter_per_chain=N_ITER,
            burnin=BURNIN, thin=THIN,
            proposal_sd=PROPOSAL_SD, base_seed=SEED,
        )
        print(f"  MCMC fertig in {time.time()-t0:.1f}s")
        with open(MCMC_CACHE, "wb") as f:
            pickle.dump(samples, f)
        print(f"  Cache gespeichert: {MCMC_CACHE.name}")

    print(f"  {len(samples['attack'])} Samples nach Burn-in/Thinning")

    # ─── Phase 7: Vorhersage-Visualisierungen ────────────────────────
    banner("PHASE 7: Ergebnis-Visualisierungen")
    print("  → MCMC-Trace (PNG + interaktives HTML) ...")
    plot_mcmc_trace(samples, L, OUT / "06_mcmc_trace.png")
    plot_mcmc_trace_html(samples, L, OUT / "06_mcmc_trace.html")
    print("  → Stärken-Evolution (PNG + interaktives HTML) ...")
    plot_strength_evolution(samples, L, OUT / "07_staerken_evolution.png")
    plot_strength_evolution_html(samples, L, OUT / "07_staerken_evolution.html")
    print("  → Team-Profile (PNG + interaktives HTML) ...")
    plot_team_ranking_summary(samples, L, OUT / "08_team_profile.png")
    plot_team_profile_html(samples, L, OUT / "08_team_profile.html")
    print("  → Rangverteilung (Replay-Simulation) ...")
    plot_predicted_rankings(samples, L, OUT / "09_rangverteilung.png",
                            n_replays=500)
    print("  → Überraschungsspiele ...")
    plot_surprise_matches(samples, L, df_season,
                          OUT / "10_ueberraschungs_spiele.png")

    # ─── Buchmacher-Vergleich: NICHT hier ────────────────────────────
    # Die ehrliche Vorhersage-Evaluation (RPS-Vergleich gegen Buchmacher)
    # ist bewusst aus dieser Pipeline herausgelöst und lebt in
    # ``backtest.py``. Grund: Der Voll-Saison-Fit oben ist eine
    # *retrospektive* (glättende) Schätzung — perfekt für die obigen
    # Visualisierungen, aber NICHT als Vorhersage verwendbar, weil die
    # Stärke zu jedem Spieltag durch *zukünftige* Spiele mitbestimmt ist
    # (Look-ahead-Leak). ``backtest.py`` fittet stattdessen pro Spieltag
    # neu auf nur den davor liegenden Spielen (Walk-Forward).
    print("\n  Buchmacher-Vergleich ausgelagert → `python backtest.py` "
          "(ehrlicher Walk-Forward-RPS, ohne Look-ahead).")

    # ─── Phase 8: Animationen ────────────────────────────────────────
    banner("PHASE 8: Animationen (kann etwas dauern)")
    try:
        print("  → Stärken-Animation ...")
        animate_strength_evolution(samples, L, OUT / "anim_01_staerken.gif")
        print("  → Tabellen-Animation ...")
        animate_season_table(L, OUT / "anim_02_tabelle.gif")
        print("  → MCMC-Walk ...")
        animate_mcmc_walk(samples, L, OUT / "anim_03_mcmc.gif")
    except Exception as e:
        print(f"  Animationen übersprungen: {e}")

    total_min = (time.time() - t_start) / 60
    banner(f"FERTIG in {total_min:.1f} min! Alle Ergebnisse in: {OUT.resolve()}")


if __name__ == "__main__":
    main()
