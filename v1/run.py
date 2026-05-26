"""
run.py
======
Hauptpipeline: Lädt Daten, baut Modell, läuft MCMC, erzeugt alle Visualisierungen.

Aufruf:
    python run.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

from data import load_bundesliga
from model import build_league
from mcmc import posterior_means
from parallel import run_mcmc_parallel
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
)
from animations import (
    animate_strength_evolution,
    animate_season_table,
    animate_mcmc_walk,
)


# === Konfiguration ===
START_YEAR = 1990
END_YEAR   = 2026          # exklusiv; 2026 => bis Saison 2025/26
ANALYSIS_SEASON = "2025/26"  # Welche Saison soll mit MCMC analysiert werden

# MCMC-Einstellungen
N_CHAINS     = 1       # parallele Ketten (= Anzahl CPU-Kerne empfohlen)
N_ITER       = 1_000_000   # mehr Iterationen für besseres Mixing
BURNIN       = 300_000   # längerer Burn-in damit Ketten konvergieren
THIN         = 20
PROPOSAL_SD  = 0.08
SEED         = 42

OUT = Path("output")
OUT.mkdir(exist_ok=True)
CACHE = Path(f"mcmc_cache_{N_CHAINS}chains_{N_ITER}iter.pkl")


def main():
    # ─── Phase 1: Daten laden ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" PHASE 1: Datenladung")
    print("=" * 60)
    df = load_bundesliga(START_YEAR, END_YEAR)

    # ─── Phase 2: Datenexploration ───────────────────────────────────────
    print("\n" + "=" * 60)
    print(" PHASE 2: Datenexploration über alle Saisons")
    print("=" * 60)
    print("  → Tor-Verteilung ...")
    plot_goal_distribution(df, OUT / "01_tor_verteilung.png")
    print("  → Heimvorteil-Entwicklung ...")
    plot_home_advantage_over_time(df, OUT / "02_heimvorteil_2000_2026.png")

    # ─── Phase 3: Modell-Intuition ──────────────────────────────────────
    print("\n" + "=" * 60)
    print(" PHASE 3: Modellbausteine (Konzept-Visualisierungen)")
    print("=" * 60)
    print("  → Brownsche Bewegungspfade ...")
    plot_brownian_motion_examples(OUT / "03_brownsche_bewegung.png")
    print("  → Tor-Modell-Aufbau ...")
    plot_tor_modell_intuition(OUT / "04_tormodell_aufbau.png")
    print("  → DAG der Modellstruktur ...")
    plot_dag_structure(OUT / "05_dag_struktur.png")

    # ─── Phase 4: MCMC auf einer Saison ─────────────────────────────────
    print("\n" + "=" * 60)
    print(f" PHASE 4: MCMC-Inferenz auf Saison {ANALYSIS_SEASON}")
    print("=" * 60)

    df_season = df[df["Season"] == ANALYSIS_SEASON].copy()
    if len(df_season) == 0:
        # Fallback: nimm die letzte Saison
        ANALYSIS_SEASON_FB = sorted(df["Season"].unique())[-1]
        print(f"  Saison '{ANALYSIS_SEASON}' nicht gefunden, "
              f"verwende {ANALYSIS_SEASON_FB}")
        df_season = df[df["Season"] == ANALYSIS_SEASON_FB].copy()

    print(f"  {len(df_season)} Spiele, "
          f"{len(set(df_season['HomeTeam']) | set(df_season['AwayTeam']))} Teams")

    L = build_league(df_season)

    # Cache: nicht jedes Mal neu rechnen
    if CACHE.exists():
        print(f"  Lade MCMC-Cache aus {CACHE}")
        with open(CACHE, "rb") as f:
            samples = pickle.load(f)
    else:
        samples = run_mcmc_parallel(
            L, n_chains=N_CHAINS, n_iter_per_chain=N_ITER,
            burnin=BURNIN, thin=THIN,
            proposal_sd=PROPOSAL_SD, base_seed=SEED,
        )
        with open(CACHE, "wb") as f:
            pickle.dump(samples, f)
        print(f"  Cache gespeichert: {CACHE}")

    print(f"  {len(samples['attack'])} Samples nach Burn-in/Thinning")

    # ─── Phase 5: Ergebnisse visualisieren ──────────────────────────────
    print("\n" + "=" * 60)
    print(" PHASE 5: Ergebnis-Visualisierungen")
    print("=" * 60)
    print("  → MCMC-Trace ...")
    plot_mcmc_trace(samples, L, OUT / "06_mcmc_trace.png")
    print("  → Stärken-Evolution ...")
    plot_strength_evolution(samples, L, OUT / "07_staerken_evolution.png")
    print("  → Team-Profile ...")
    plot_team_ranking_summary(samples, L, OUT / "08_team_profile.png")
    print("  → Rangverteilung (Replay-Simulation) ...")
    plot_predicted_rankings(samples, L, OUT / "09_rangverteilung.png",
                            n_replays=500)
    print("  → Überraschungsspiele ...")
    plot_surprise_matches(samples, L, df_season,
                          OUT / "10_ueberraschungs_spiele.png")

    # ─── Phase 6: Animationen ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" PHASE 6: Animationen (kann etwas dauern)")
    print("=" * 60)
    try:
        print("  → Stärken-Animation ...")
        animate_strength_evolution(samples, L, OUT / "anim_01_staerken.gif")
        print("  → Tabellen-Animation ...")
        animate_season_table(L, OUT / "anim_02_tabelle.gif")
        print("  → MCMC-Walk ...")
        animate_mcmc_walk(samples, L, OUT / "anim_03_mcmc.gif")
    except Exception as e:
        print(f"  Animationen übersprungen: {e}")

    print("\n" + "=" * 60)
    print(f" FERTIG! Alle Ergebnisse in: {OUT.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
