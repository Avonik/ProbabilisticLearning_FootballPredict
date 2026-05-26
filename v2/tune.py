"""
tune.py
=======
Tuning der Hyperparameter τ, γ, ε.

Hintergrund — Wie der Paper-Original tunt
-----------------------------------------
Rue & Salvesen (2000), Abschnitt 3.2:
  * Tuning-Daten: **1684 Premier-League- + 2208 Division-1-Spiele aus 4 Saisons**
    (1993–1997) = 3892 Matches.
  * Vorgehen: Pro Saison erste Hälfte zum Trainieren, zweite Hälfte
    rollierend vorhersagen. Metrik: geometrisches Mittel der vorhergesagten
    Wahrscheinlichkeiten = normalisierte Pseudo-Likelihood (PL).
  * Grid-Search über τ × γ × ε.
  * Befund: τ=100, γ=0.1, ε=0.2 — überraschend stabil über alle Saisons
    und beide Ligen. Die Likelihood-Oberfläche ist *flach* (viele
    Kombinationen geben fast denselben PL-Wert).

Implementierung in v2
---------------------
Zwei Varianten:

  ``tune_hyperparameters(df_single_season, ...)``
      Schnelles Single-Season-Tuning auf 70/30-Split. Dauert ~30 s, aber
      die Validierungsmenge (~90 Matches) ist klein — Ergebnisse wackeln
      zwischen Läufen.

  ``tune_paperstyle(df_full, history_seasons, ...)``  ← **bevorzugt**
      Vorgehen wie im Paper. Pro Saison: 1. Hälfte trainieren, 2. Hälfte
      bewerten. Aggregiert über mehrere Saisons hinweg den gewichteten
      Mittelwert der Outcome-Log-Likelihood. Dauert mehrere Minuten,
      gibt aber stabile, reproduzierbare Optimalwerte.
"""

from __future__ import annotations

import itertools
import time
import numpy as np
import pandas as pd

from model import build_league, predict_outcome_probs, MAX_GOALS
from mcmc import run_mcmc, posterior_means
from evaluation import outcome_index, rps_one


# ─────────────────────────────────────────────────────────────────────
# Holdout-Bewertung
# ─────────────────────────────────────────────────────────────────────

def evaluate_holdout(samples, L_train, df_holdout: pd.DataFrame,
                     metric: str = "log_lik") -> float:
    """
    Bewertung der Validierungsspiele mit posteriori Mittelwert der Stärken
    aus der letzten verfügbaren Schätzung pro Team.

    metric ∈ {'log_lik', 'rps'}. log_lik: höher = besser. rps: niedriger = besser
    (wir geben −RPS zurück, sodass "höher = besser" durchgängig gilt).
    """
    mean_attack, mean_defense, _ = posterior_means(samples)

    total = 0.0
    n = 0
    for _, r in df_holdout.iterrows():
        h_name, a_name = r["HomeTeam"], r["AwayTeam"]
        if h_name not in L_train.teams or a_name not in L_train.teams:
            continue
        h = L_train.teams.index(h_name)
        a = L_train.teams.index(a_name)
        a_h, d_h = float(mean_attack[h][-1]), float(mean_defense[h][-1])
        a_a, d_a = float(mean_attack[a][-1]), float(mean_defense[a][-1])
        p_h, p_d, p_a = predict_outcome_probs(
            a_h, d_h, a_a, d_a, L_train.c_x, L_train.c_y,
            gamma=L_train.gamma, eps=L_train.epsilon, max_k=MAX_GOALS,
        )
        outcome = outcome_index(int(r["FTHG"]), int(r["FTAG"]))
        if metric == "log_lik":
            probs = [p_h, p_d, p_a]
            total += float(np.log(max(probs[outcome], 1e-12)))
        elif metric == "rps":
            total += -rps_one(np.array([p_h, p_d, p_a]), outcome)
        else:
            raise ValueError(f"Unbekannte metric: {metric}")
        n += 1
    return total / max(1, n)


# ─────────────────────────────────────────────────────────────────────
# Grid-Search
# ─────────────────────────────────────────────────────────────────────

def split_by_date(df: pd.DataFrame, frac_train: float = 0.7
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zeit-Split: frühe frac_train für Training, Rest für Holdout."""
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    k = int(len(df_sorted) * frac_train)
    return df_sorted.iloc[:k].copy(), df_sorted.iloc[k:].copy()


def tune_hyperparameters(
    df_season: pd.DataFrame,
    use_xg: bool = False,
    taus: tuple = (40.0, 80.0, 120.0, 200.0),
    gammas: tuple = (0.0, 0.05, 0.10, 0.20),
    epsilons: tuple = (0.05, 0.15, 0.25),
    n_iter: int = 4000,
    burnin: int = 1500,
    thin: int = 10,
    proposal_sd: float = 0.08,
    metric: str = "log_lik",
    frac_train: float = 0.7,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Grid-Search über (τ, γ, ε). Gibt das Tripel mit bester Holdout-Metric zurück.

    Returns:
        dict mit Schlüsseln:
            'best': {'tau': ..., 'gamma': ..., 'epsilon': ..., 'score': ...}
            'all_results': list[dict]   — vollständige Ergebnistabelle
            'metric': name
    """
    df_train, df_val = split_by_date(df_season, frac_train=frac_train)
    if verbose:
        print(f"  Train: {len(df_train)} Spiele, Val: {len(df_val)} Spiele")
        n_combos = len(taus) * len(gammas) * len(epsilons)
        print(f"  Grid: {len(taus)} tau x {len(gammas)} gamma x "
              f"{len(epsilons)} eps  =  {n_combos} Kombinationen")

    results = []
    combos = list(itertools.product(taus, gammas, epsilons))
    t_start = time.time()
    for i, (tau, gamma, eps) in enumerate(combos):
        t0 = time.time()
        L_train = build_league(df_train, use_xg=use_xg,
                               tau=tau, gamma=gamma, epsilon=eps)
        samples = run_mcmc(L_train, n_iter=n_iter, burnin=burnin, thin=thin,
                           proposal_sd=proposal_sd, seed=seed, verbose=False)
        score = evaluate_holdout(samples, L_train, df_val, metric=metric)
        elapsed = time.time() - t0
        results.append({
            "tau": tau, "gamma": gamma, "epsilon": eps,
            "score": score,
            "acceptance": samples["acceptance"],
            "time_s": elapsed,
        })
        if verbose:
            print(f"  [{i+1:>2}/{len(combos)}] tau={tau:>5.0f}  "
                  f"gamma={gamma:.2f}  eps={eps:.2f}  ->  "
                  f"{metric}={score:+.4f}  "
                  f"(acc={samples['acceptance']:.0%}, {elapsed:.1f}s)")

    total_time = time.time() - t_start
    best = max(results, key=lambda r: r["score"])
    if verbose:
        print(f"\n  Beste Kombination: tau={best['tau']}, gamma={best['gamma']}, "
              f"eps={best['epsilon']}  ->  {metric}={best['score']:+.4f}")
        print(f"  Gesamt: {total_time:.1f}s")

    return {
        "best": best,
        "all_results": results,
        "metric": metric,
        "total_time_s": total_time,
        "frac_train": frac_train,
        "n_train": len(df_train),
        "n_val": len(df_val),
    }


# ─────────────────────────────────────────────────────────────────────
# Paper-Style: Multi-Season-Tuning (Rue & Salvesen 2000, Abschnitt 3.2)
# ─────────────────────────────────────────────────────────────────────

def _evaluate_holdout_loglik(samples, L_train, df_holdout: pd.DataFrame
                             ) -> tuple[float, int]:
    """Wie evaluate_holdout, aber gibt (Σ log P, n) statt Mittelwert zurück."""
    mean_attack, mean_defense, _ = posterior_means(samples)
    total = 0.0
    n = 0
    for _, r in df_holdout.iterrows():
        h_name, a_name = r["HomeTeam"], r["AwayTeam"]
        if h_name not in L_train.teams or a_name not in L_train.teams:
            continue
        h = L_train.teams.index(h_name)
        a = L_train.teams.index(a_name)
        a_h, d_h = float(mean_attack[h][-1]), float(mean_defense[h][-1])
        a_a, d_a = float(mean_attack[a][-1]), float(mean_defense[a][-1])
        p_h, p_d, p_a = predict_outcome_probs(
            a_h, d_h, a_a, d_a, L_train.c_x, L_train.c_y,
            gamma=L_train.gamma, eps=L_train.epsilon, max_k=MAX_GOALS,
        )
        outcome = outcome_index(int(r["FTHG"]), int(r["FTAG"]))
        probs = (p_h, p_d, p_a)
        total += float(np.log(max(probs[outcome], 1e-12)))
        n += 1
    return total, n


def tune_paperstyle(
    df_full: pd.DataFrame,
    history_seasons: list[str],
    use_xg: bool = False,
    taus: tuple = (40.0, 80.0, 120.0, 200.0),
    gammas: tuple = (0.0, 0.05, 0.10, 0.20),
    epsilons: tuple = (0.05, 0.15, 0.25),
    n_iter: int = 6000,
    burnin: int = 2000,
    thin: int = 10,
    proposal_sd: float = 0.06,
    frac_train: float = 0.5,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Multi-Season-Tuning analog Rue & Salvesen 2000, Abschnitt 3.2.

    Für jede (τ, γ, ε)-Kombination und jede Saison in ``history_seasons``:
      1) Erste Hälfte der Saison → Train (MCMC-Fit)
      2) Zweite Hälfte → Holdout (predict + Σ log P(Ergebnis))
    Aggregiere Σ log P über alle Saisons; teile durch Gesamtzahl Matches
    → gewichteter Mittelwert log-Likelihood (≡ log der geometrischen
    Mittel-Wahrscheinlichkeit = normalisierter Pseudolikelihood-PL).

    Im Paper wurden 3892 Matches verwendet (4 Saisons × 2 Ligen).
    Mit 5 Bundesliga-Saisons bekommen wir ~750 Holdout-Matches → fast
    10× soviel Signal wie das Single-Season-Tuning (~90 Matches).
    """
    history_dfs = []
    for s in history_seasons:
        sub = df_full[df_full["Season"] == s].copy()
        if len(sub) > 0:
            history_dfs.append((s, sub))
    if not history_dfs:
        raise ValueError(f"Keine Saisons aus {history_seasons} im df_full!")

    if verbose:
        total_matches = sum(len(d) for _, d in history_dfs)
        print(f"  Paper-Style Tuning: {len(history_dfs)} Saisons, "
              f"{total_matches} Matches gesamt")
        for s, d in history_dfs:
            print(f"    {s}: {len(d)} Matches "
                  f"(~{int(len(d) * frac_train)} train / "
                  f"~{int(len(d) * (1 - frac_train))} holdout)")
        n_combos = len(taus) * len(gammas) * len(epsilons)
        n_fits = n_combos * len(history_dfs)
        print(f"  Grid: {len(taus)} tau x {len(gammas)} gamma x "
              f"{len(epsilons)} eps  =  {n_combos} Kombinationen")
        print(f"  Insgesamt {n_fits} MCMC-Fits (kann mehrere Minuten dauern)")

    results = []
    combos = list(itertools.product(taus, gammas, epsilons))
    t_start = time.time()

    for i, (tau, gamma, eps) in enumerate(combos):
        t0 = time.time()
        sum_log_lik = 0.0
        n_total = 0
        accs = []
        per_season = []

        for s_name, df_season in history_dfs:
            df_tr, df_val = split_by_date(df_season, frac_train=frac_train)
            if len(df_tr) < 10 or len(df_val) < 10:
                continue
            L_train = build_league(df_tr, use_xg=use_xg,
                                   tau=tau, gamma=gamma, epsilon=eps)
            samples = run_mcmc(L_train, n_iter=n_iter, burnin=burnin,
                               thin=thin, proposal_sd=proposal_sd,
                               seed=seed, verbose=False)
            ll, n = _evaluate_holdout_loglik(samples, L_train, df_val)
            sum_log_lik += ll
            n_total += n
            accs.append(samples["acceptance"])
            per_season.append({"season": s_name, "n": n, "mean_ll": ll / max(1, n)})

        mean_ll = sum_log_lik / max(1, n_total)
        elapsed = time.time() - t0
        results.append({
            "tau": tau, "gamma": gamma, "epsilon": eps,
            "score": mean_ll,
            "n_total": n_total,
            "acceptance": float(np.mean(accs)) if accs else 0.0,
            "time_s": elapsed,
            "per_season": per_season,
        })
        if verbose:
            print(f"  [{i+1:>2}/{len(combos)}] tau={tau:>5.0f}  "
                  f"gamma={gamma:.2f}  eps={eps:.2f}  ->  "
                  f"log_lik={mean_ll:+.4f}  (n={n_total}, "
                  f"acc={np.mean(accs):.0%}, {elapsed:.1f}s)")

    total_time = time.time() - t_start
    best = max(results, key=lambda r: r["score"])

    # Standardfehler des besten Mittelwerts (für "ist der Unterschied real?")
    n_best = best["n_total"]
    se_approx = 1.0 / np.sqrt(max(1, n_best))  # grob: σ_ll ~ 1
    # "praktisch gleichwertig" = innerhalb 1 SE vom Besten
    close_to_best = [r for r in results
                     if best["score"] - r["score"] < se_approx]

    if verbose:
        print(f"\n  Beste Kombination: tau={best['tau']}, "
              f"gamma={best['gamma']}, eps={best['epsilon']}  "
              f"->  log_lik={best['score']:+.4f}  (n={n_best})")
        print(f"  Geschätzter SE des Mittelwerts: ~{se_approx:.4f}")
        print(f"  {len(close_to_best)}/{len(results)} Kombinationen sind "
              f"innerhalb 1 SE vom Besten — wenn das viele sind, ist die "
              f"Likelihood-Oberfläche flach (siehe Paper-Befund).")
        print(f"  Gesamt: {total_time:.1f}s ({total_time/60:.1f} min)")

    return {
        "best": best,
        "all_results": results,
        "metric": "log_lik",
        "total_time_s": total_time,
        "n_seasons": len(history_dfs),
        "season_names": [s for s, _ in history_dfs],
        "n_holdout_total": best["n_total"],
        "se_approx": se_approx,
        "n_close_to_best": len(close_to_best),
        "frac_train": frac_train,
    }
