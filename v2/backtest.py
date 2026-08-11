"""
backtest.py
===========
EHRLICHER Walk-Forward-Buchmacher-Vergleich (out-of-sample).

Warum eine eigene Datei?
------------------------
``run.py`` fittet das Modell *retrospektiv* auf die GANZE Saison. Diese
Glättungs-Schätzung ist ideal für die Visualisierungen, aber als Vorhersage
unbrauchbar: die Stärke eines Teams zu Spieltag d ist im Voll-Saison-Fit
durch *zukünftige* Spiele mitbestimmt (Look-ahead-Leak über den Brownian-
Motion-Zeit-Prior). Wer damit die Holdout-Spiele "vorhersagt" und gegen den
Buchmacher antritt, schlägt ihn künstlich.

Dieses Skript macht es richtig — **Walk-Forward / Expanding Window**:

  Für jeden Holdout-Spieltag d:
    1. Baue die Liga NUR aus Spielen mit Datum < d  (Trainings-Präfix).
    2. Fitte das Modell darauf (MCMC).
    3. Sage die Spiele von Tag d aus der *letzten* Stärke pro Team voraus
       (Filter-Schätzung am Trainings-Ende — kennt nur die Vergangenheit).

  Warm-Start: jeder Refit startet von den zuletzt bekannten Stärken des
  vorherigen Spieltags → kurzer Burn-in, wenige Iterationen genügen.

Geschlossene Lecks gegenüber dem alten ``compare_to_bookmaker``:
  * #1 Look-ahead/Smoothing  → Fit nur auf Präfix, letzter Knoten = Filter.
  * #2 xG-β auf Vollsaison   → β nur aus Trainings-Saisons (vor der Analyse).
  * #3 c_x/c_y auf Vollsaison → automatisch aus dem Präfix berechnet.

Aufruf:  python backtest.py
"""

from __future__ import annotations

import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

import numpy as np
import pandas as pd

from data import load_bundesliga, extract_bookmaker_probs, sort_matches, MATCH_ID_COL
from xg import fit_xg_weights, add_xg_columns
from model import (build_league, predict_outcome_probs, MAX_GOALS,
                   DEFAULT_TAU, DEFAULT_GAMMA, DEFAULT_EPSILON,
                   DEFAULT_HOME_ADV_PRIOR_SD)
from mcmc import (run_mcmc, posterior_means, posterior_home_advantage,
                  warmup_jit)
from parallel import gelman_rubin
from evaluation import (
    bookmaker_probs_from_df, uniform_baseline, empirical_baseline,
    evaluate_predictions, outcome_index,
)
from historical_home import estimate_historical_home_prior
# viz (matplotlib) wird erst in main() importiert — so müssen die parallelen
# Ketten-Worker-Prozesse kein matplotlib laden.


# ─────────────────────────────────────────────────────────────────────
# Konfiguration
# ─────────────────────────────────────────────────────────────────────
START_YEAR      = 1993
END_YEAR        = 2026
ANALYSIS_SEASON = "2021/22"
USE_XG          = True
HOLDOUT_FRAC    = 0.3        # letzte 30 % der Saison (wie zuvor)

# Kontinuierliches xG-Beobachtungsmodell?  → rohes xG (kein Runden) mit
# Gamma-Likelihood (Präzision PHI) statt Trunc-Poisson auf round(xG).
# Var[xG|λ] = λ²/PHI: großes PHI ⇒ xG wird stärker vertraut ⇒ schärfere
# Stärken-Updates ⇒ weniger Hedging. Braucht USE_XG=True.
USE_CONTINUOUS_XG = True
PHI               = 5.0

# Team-specific deviation from the league-wide home advantage.  All team
# effects share a centered N(0, sd²) prior and therefore partially pool toward
# zero.  The sum-to-zero constraint is preserved by pairwise MCMC proposals.
USE_TEAM_HOME_ADVANTAGE = True
TEAM_HOME_ADV_PRIOR_SD = DEFAULT_HOME_ADV_PRIOR_SD

# Carry team-specific home effects across seasons without leaking target-season
# outcomes. Each completed historical season is fitted once and cached. The
# latest seasons receive the largest weight; promoted/unknown teams start at 0.
USE_HISTORICAL_HOME_PRIOR = True
HOME_ADV_HISTORY_SEASONS = 3
HOME_ADV_HISTORY_DECAY = 0.65
HOME_ADV_HISTORY_ITER = 8_000
HOME_ADV_HISTORY_BURNIN = 2_000
HOME_ADV_HISTORY_THIN = 10

# Echtes Understat-xG (ab 2014/15) statt Schuss-Proxy?  → real_xg.py holt es
# via soccerdata (gecached) und überschreibt xG_home/xG_away wo verfügbar;
# Spiele davor behalten den Proxy. Braucht USE_XG=True + Internet beim 1. Lauf.
USE_REAL_XG       = True

# Getunte Hyperparameter (τ/γ/ε aus dem tune_paper-Cache) verwenden — oder die
# FESTEN Paper-Defaults (τ=100, γ=0.10, ε=0.20)?
#   True  : resolve_hyperparams lädt den getunten Cache (sonst Defaults).
#   False : immer feste Paper-Defaults → garantiert leckfrei (kein Tuning-Cache,
#           der evtl. Saisons NACH der Analyse-Saison gesehen hat). Empfohlen
#           für die Multi-Season-Signifikanzstudie.
USE_TUNED_HYPERPARAMS = False

# Walk-Forward-MCMC-Budget (getunt + Warm-Start).
#   Erster Refit (Trainings-Präfix, ~70 % der Saison) wird kalt gestartet
#   → größeres Budget. Folge-Refits warm-gestartet → kleines Budget.
WF_BASE_ITER    = 40_000 #40k
WF_BASE_BURNIN  = 8_000 #8k
WF_WARM_ITER    = 20_000 #20k
WF_WARM_BURNIN  = 3_000 #3k
WF_THIN         = 10 #10
PROPOSAL_SD     = 0.06 #0.06
SEED            = 42

# Mehrere MCMC-Ketten pro Refit?  → liefert eine R-hat-Konvergenzdiagnostik.
#   Die Ketten laufen PARALLEL (max_workers=N_CHAINS); der Worker-Pool bleibt
#   über ALLE Spieltage offen → kaum Mehrkosten, solange N_CHAINS ≤ CPU-Kerne.
#   Die Startpunkte werden überdispergiert (Jitter um den Warm-Start), damit
#   R-hat aussagekräftig ist (gleiche Starts → falsch-optimistisches R-hat).
USE_MULTI_CHAIN = True
N_CHAINS        = 4
CHAIN_JITTER_SD = 0.1

# Informativer Prior aus Team-Marktwerten (Option A). USE_MARKET_PRIOR=False
# ⇒ κ=0 ⇒ exakt das bisherige Modell. Die CSV-Werte MÜSSEN as-of Saisonstart
# sein (leak-frei!), Spalten: Season, Team, MarketValue.
USE_MARKET_PRIOR   = True
MARKET_PRIOR_KAPPA = 0.10
MARKET_VALUE_CSV   = "data_cache/transfermarkt_squad_values.csv"

# Zusätzlich das PAPER-STANDARDMODELL (Rue & Salvesen 2000, unverändert) als
# Vergleichsbalken? Gleiche Walk-Forward-Maschine, aber alle v2-Erweiterungen
# AUS: Tore statt xG, feste Paper-Hyperparameter (τ=100, γ=0.10, ε=0.20),
# 0-Prior. Zeigt, wie viel die v2-Anpassungen gegenüber dem Original bringen.
# Kostet einen ZWEITEN Walk-Forward-Lauf (~doppelte Backtest-Laufzeit).
INCLUDE_PAPER_BASELINE = True

OUTPUT_ROOT = Path("output")
OUT = OUTPUT_ROOT
CACHE_DIR = Path("cache_v2")


def timestamped_output_dir(prefix: str) -> Path:
    OUTPUT_ROOT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUTPUT_ROOT / f"{prefix}_{stamp}"
    out = base
    i = 2
    while out.exists():
        out = OUTPUT_ROOT / f"{base.name}_{i}"
        i += 1
    out.mkdir(parents=True)
    return out


# ─────────────────────────────────────────────────────────────────────
# Hyperparameter — den (sauberen) Tuning-Cache von run.py wiederverwenden
# ─────────────────────────────────────────────────────────────────────
def resolve_hyperparams(season: str, use_xg: bool) -> tuple[float, float, float]:
    """Lädt getunte (τ, γ, ε) aus dem Tuning-Cache; sonst Paper-Defaults.

    Das Tuning in ``tune.py`` ist leckfrei (History-Saisons VOR der Analyse-
    Saison, disjunkter Train/Holdout-Split) — wir dürfen es bedenkenlos
    wiederverwenden.
    """
    tag = "xg" if use_xg else "goals"
    s = season.replace("/", "_")
    candidates = [
        CACHE_DIR / f"tune_paper_{s}_{tag}.pkl",
        CACHE_DIR / f"tune_single_{s}_{tag}.pkl",
        CACHE_DIR / f"tune_{s}_{tag}.pkl",
    ]
    for c in candidates:
        if c.exists():
            with open(c, "rb") as f:
                tr = pickle.load(f)
            b = tr["best"]
            print(f"  Getunte Hyperparameter aus {c.name}: "
                  f"τ={b['tau']}, γ={b['gamma']}, ε={b['epsilon']}")
            return float(b["tau"]), float(b["gamma"]), float(b["epsilon"])
    print("  Kein Tuning-Cache gefunden → Paper-Defaults τ=100, γ=0.10, ε=0.20 "
          "(Tipp: erst run.py laufen lassen, um zu tunen).")
    return 100.0, 0.10, 0.20


# ─────────────────────────────────────────────────────────────────────
# Warm-Start-Hilfen
# ─────────────────────────────────────────────────────────────────────
def _last_strength_by_name(samples, L) -> dict[str, tuple[float, float]]:
    """Posteriori-Mittel der *letzten* Stärke pro Team (Filter-Schätzung
    am Trainings-Ende), gemappt über den Team-NAMEN (nicht den Index, der
    sich zwischen verschieden großen Präfix-Ligen ändert)."""
    mean_attack, mean_defense, _ = posterior_means(samples)
    out: dict[str, tuple[float, float]] = {}
    for t in range(L.n_teams):
        out[L.teams[t]] = (float(mean_attack[t][-1]),
                           float(mean_defense[t][-1]))
    return out


def _flat_init(L, strengths_by_name: dict[str, tuple[float, float]]
               ) -> tuple[np.ndarray, np.ndarray]:
    """Baut flache Start-Arrays: jeder Knoten eines Teams startet auf dessen
    zuletzt bekannter Stärke. Neue/unbekannte Teams starten bei 0 (Prior)."""
    n_total = int(L.team_start[-1])
    a = np.zeros(n_total, dtype=np.float64)
    d = np.zeros(n_total, dtype=np.float64)
    for t in range(L.n_teams):
        s, e = int(L.team_start[t]), int(L.team_start[t + 1])
        av, dv = strengths_by_name.get(L.teams[t], (0.0, 0.0))
        a[s:e] = av
        d[s:e] = dv
    return a, d


def _home_advantage_by_name(samples, L) -> dict[str, float]:
    means = posterior_home_advantage(samples)
    if len(means) == 0:
        return {team: 0.0 for team in L.teams}
    return {team: float(means[t]) for t, team in enumerate(L.teams)}


def _home_adv_init(L, values_by_name: dict[str, float] | None) -> np.ndarray:
    values = np.array(
        [0.0 if values_by_name is None else values_by_name.get(team, 0.0)
         for team in L.teams],
        dtype=np.float64,
    )
    values -= values.mean()
    return values


# ─────────────────────────────────────────────────────────────────────
# Multi-Chain-Hilfen (Worker muss top-level sein → picklebar für den Pool)
# ─────────────────────────────────────────────────────────────────────
def _chain_worker(args):
    """Fittet EINE MCMC-Kette (läuft im Worker-Prozess)."""
    (L, n_iter, burnin, thin, proposal_sd, seed, init_a, init_d, init_h) = args
    return run_mcmc(
        L, n_iter=n_iter, burnin=burnin, thin=thin,
        proposal_sd=proposal_sd, seed=seed, verbose=False,
        init_attack=init_a, init_defense=init_d,
        init_home_advantage=init_h,
    )


def _compute_rhat(chains: list, n_teams: int) -> float:
    """Max R-hat (Gelman-Rubin) über alle Team-Stärke-Mittel, mehrere Ketten.

    Analog zu ``parallel.run_mcmc_parallel``: pro Team & Komponente wird die
    Kette der Sample-weisen mittleren Stärke gebildet und R-hat berechnet.
    """
    vals = []
    for team in range(n_teams):
        for key in ("attack", "defense"):
            series = [np.array([s[team].mean() for s in r[key]]) for r in chains]
            if min(len(c) for c in series) >= 2:
                vals.append(gelman_rubin(series))
        if chains[0].get("home_advantage"):
            series = [
                np.array([sample[team] for sample in r["home_advantage"]])
                for r in chains
            ]
            if min(len(c) for c in series) >= 2:
                vals.append(gelman_rubin(series))
    return float(np.max(vals)) if vals else float("nan")


# ─────────────────────────────────────────────────────────────────────
# Kern: Walk-Forward-Vorhersagen für eine Saison
# ─────────────────────────────────────────────────────────────────────
def walkforward_predictions(
    df_full: pd.DataFrame,
    season: str,
    *,
    use_xg: bool,
    tau: float, gamma: float, eps: float,
    holdout_frac: float = 0.3,
    base_iter: int = WF_BASE_ITER, base_burnin: int = WF_BASE_BURNIN,
    warm_iter: int = WF_WARM_ITER, warm_burnin: int = WF_WARM_BURNIN,
    thin: int = WF_THIN, proposal_sd: float = PROPOSAL_SD,
    seed: int = SEED, verbose: bool = True,
    n_chains: int = 1, jitter_sd: float = 0.1,
    continuous_xg: bool = USE_CONTINUOUS_XG, phi: float = PHI,
    market_values: dict | None = None, market_kappa: float = 0.0,
    use_team_home_advantage: bool = USE_TEAM_HOME_ADVANTAGE,
    home_adv_prior_sd: float = TEAM_HOME_ADV_PRIOR_SD,
    use_historical_home_prior: bool = USE_HISTORICAL_HOME_PRIOR,
    home_adv_history_seasons: int = HOME_ADV_HISTORY_SEASONS,
    home_adv_history_decay: float = HOME_ADV_HISTORY_DECAY,
    home_adv_history_iter: int = HOME_ADV_HISTORY_ITER,
    home_adv_history_burnin: int = HOME_ADV_HISTORY_BURNIN,
    home_adv_history_thin: int = HOME_ADV_HISTORY_THIN,
) -> dict:
    """
    Liefert out-of-sample Modell-Wahrscheinlichkeiten für die Holdout-Spiele
    der Saison via Refit-pro-Spieltag.

    Args:
        n_chains: Ketten pro Refit. >1 ⇒ parallel + R-hat-Diagnostik.
        jitter_sd: SD der Überdispersion der Ketten-Startpunkte (nur n_chains>1).

    Returns: dict mit
        probs_model (n,3)  — NaN außerhalb des Holdouts
        cutoff             — Index der ersten Holdout-Zeile (Datum-aligned)
        n                  — Anzahl Saison-Spiele
        df_season          — sortierter Saison-DataFrame (reset index)
        rhat_max           — max R-hat über alle Refits (NaN bei n_chains=1)
    """
    df_season = sort_matches(df_full[df_full["Season"] == season])
    n = len(df_season)
    if n == 0:
        raise ValueError(f"Saison '{season}' nicht in den Daten.")

    # Holdout = letzte holdout_frac der Saison, an GANZEN Spieltagen (Daten)
    # ausgerichtet, damit kein Datum train/holdout splittet.
    cutoff_idx0 = int(n * (1.0 - holdout_frac))
    cutoff_date = df_season.loc[cutoff_idx0, "Date"]
    hold_mask = (df_season["Date"] >= cutoff_date).values
    cutoff = int(np.argmax(hold_mask))          # erste Holdout-Zeile
    holdout_dates = sorted(df_season.loc[hold_mask, "Date"].unique())

    if verbose:
        print(f"  Saison {season}: {n} Spiele, Holdout ab {pd.Timestamp(cutoff_date).date()} "
              f"→ {n - cutoff} Spiele über {len(holdout_dates)} Spieltage")
        if continuous_xg:
            print(f"  Beobachtungsmodell: kontinuierliches xG (Gamma, φ={phi:g})")
        else:
            print(f"  Beobachtungsmodell: {'gerundetes xG' if use_xg else 'tatsächliche Tore'} "
                  f"(Trunc-Poisson + Dixon-Coles)")
        print("  Teamspezifischer Heimvorteil: " +
              (f"AN (Prior-SD={home_adv_prior_sd:g})"
               if use_team_home_advantage else "AUS"))

    historical_home_prior: dict[str, float] = {}
    history_info: dict = {"seasons": [], "cache_hits": 0}
    if use_team_home_advantage and use_historical_home_prior:
        historical_home_prior, history_info = estimate_historical_home_prior(
            df_full,
            target_season=season,
            target_teams=sorted(set(df_season["HomeTeam"]) |
                                set(df_season["AwayTeam"])),
            use_xg=use_xg,
            tau=tau, gamma=gamma, epsilon=eps,
            continuous_xg=continuous_xg, phi=phi,
            prior_sd=home_adv_prior_sd,
            n_seasons=home_adv_history_seasons,
            decay=home_adv_history_decay,
            n_iter=home_adv_history_iter,
            burnin=home_adv_history_burnin,
            thin=home_adv_history_thin,
            proposal_sd=proposal_sd,
            seed=seed + 50_000,
            cache_dir=CACHE_DIR / "historical_home",
        )
        if verbose:
            seasons_text = ", ".join(history_info["seasons"]) or "keine"
            nonzero = sum(abs(v) > 1e-12 for v in historical_home_prior.values())
            print(f"  Historischer Heim-Prior: {seasons_text}; "
                  f"{nonzero}/{len(historical_home_prior)} Teams mit Historie "
                  f"(Decay={home_adv_history_decay:g}, "
                  f"Cache-Hits={history_info['cache_hits']})")

    probs_model = np.full((n, 3), np.nan)
    warm: dict[str, tuple[float, float]] | None = None
    warm_home: dict[str, float] | None = None
    rhat_max = float("nan")
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # Mehrere Ketten? → EIN Worker-Pool, der über alle Spieltage offen bleibt
    # (vermeidet wiederholtes Prozess-Spawnen + JIT-Laden, v.a. auf Windows).
    executor = ProcessPoolExecutor(max_workers=n_chains) if n_chains > 1 else None
    try:
        for di, d in enumerate(holdout_dates):
            df_prefix = df_season[df_season["Date"] < d]
            if len(df_prefix) < 10:
                continue  # zu wenig Historie für eine sinnvolle Schätzung

            L = build_league(df_prefix, use_xg=use_xg,
                             tau=tau, gamma=gamma, epsilon=eps,
                             continuous_xg=continuous_xg, phi=phi,
                             team_values=market_values, market_kappa=market_kappa,
                             use_team_home_advantage=use_team_home_advantage,
                             home_adv_prior_sd=home_adv_prior_sd,
                             home_adv_prior_means=historical_home_prior)
            n_total = int(L.team_start[-1])

            # Warm-Start-Basis (zuletzt bekannte Stärken) + Iterationsbudget
            if warm is None:
                base_a = np.zeros(n_total)
                base_d = np.zeros(n_total)
                base_h = np.asarray(L.home_adv_prior_mean, dtype=float).copy()
                n_iter, burnin = base_iter, base_burnin
            else:
                base_a, base_d = _flat_init(L, warm)
                base_h = _home_adv_init(L, warm_home)
                n_iter, burnin = warm_iter, warm_burnin

            rhat_str = ""
            if executor is None:
                # ── Eine Kette (Warm-Start ohne Jitter) ──────────────────
                ia = None if warm is None else base_a
                id_ = None if warm is None else base_d
                ih = (base_h if use_team_home_advantage else None)
                samples = run_mcmc(
                    L, n_iter=n_iter, burnin=burnin, thin=thin,
                    proposal_sd=proposal_sd, seed=seed + di, verbose=False,
                    init_attack=ia, init_defense=id_,
                    init_home_advantage=ih,
                )
                acc = samples["acceptance"]
            else:
                # ── Mehrere Ketten, ÜBERDISPERGIERTE Starts (für R-hat) ──
                args = []
                for ci in range(n_chains):
                    ia = base_a + rng.normal(0.0, jitter_sd, n_total)
                    id_ = base_d + rng.normal(0.0, jitter_sd, n_total)
                    ih = base_h + rng.normal(0.0, jitter_sd, L.n_teams)
                    ih -= ih.mean()
                    args.append((L, n_iter, burnin, thin, proposal_sd,
                                 seed + di * 1000 + ci, ia, id_, ih))
                chains = list(executor.map(_chain_worker, args))
                samples = {"attack": [], "defense": [],
                           "home_advantage": [], "delta": []}
                for r in chains:
                    samples["attack"].extend(r["attack"])
                    samples["defense"].extend(r["defense"])
                    samples["home_advantage"].extend(r["home_advantage"])
                    samples["delta"].extend(r["delta"])
                acc = float(np.mean([r["acceptance"] for r in chains]))
                rh = _compute_rhat(chains, L.n_teams)
                if not np.isnan(rh):
                    rhat_max = rh if np.isnan(rhat_max) else max(rhat_max, rh)
                    rhat_str = f", R̂={rh:.3f}"

            strengths = _last_strength_by_name(samples, L)
            home_advantages = _home_advantage_by_name(samples, L)

            # Spiele dieses Datums vorhersagen (nur Vergangenheit ist eingeflossen)
            games = df_season[df_season["Date"] == d]
            for row_i, row in games.iterrows():
                h, a = row["HomeTeam"], row["AwayTeam"]
                if h not in strengths or a not in strengths:
                    continue
                a_h, d_h = strengths[h]
                a_a, d_a = strengths[a]
                home_adv = home_advantages.get(h, 0.0)
                p_h, p_d, p_a = predict_outcome_probs(
                    a_h, d_h, a_a, d_a, L.c_x, L.c_y,
                    gamma=gamma, eps=eps, max_k=MAX_GOALS,
                    home_advantage=home_adv,
                )
                probs_model[row_i] = (p_h, p_d, p_a)

            warm = strengths   # Warm-Start für den nächsten Spieltag
            warm_home = home_advantages

            if verbose:
                done = di + 1
                print(f"    [{done:>2}/{len(holdout_dates)}] {pd.Timestamp(d).date()}  "
                      f"Präfix={len(df_prefix):>3} Spiele  "
                      f"(acc={acc:.0%}{rhat_str}, {time.time()-t0:.0f}s gesamt)")
    finally:
        if executor is not None:
            executor.shutdown()

    return {
        "probs_model": probs_model,
        "cutoff": cutoff,
        "n": n,
        "df_season": df_season,
        "rhat_max": rhat_max,
        "home_advantages": warm_home or {},
        "historical_home_prior": historical_home_prior,
        "historical_home_info": history_info,
    }


# ─────────────────────────────────────────────────────────────────────
# Saison-Evaluation: Modell vs Buchmacher vs Baselines (ehrlich)
# ─────────────────────────────────────────────────────────────────────
def evaluate_season_walkforward(
    df_full: pd.DataFrame, season: str, *,
    use_xg: bool, tau: float, gamma: float, eps: float,
    holdout_frac: float = 0.3, verbose: bool = True, **wf_kwargs,
) -> dict:
    """Vollständiger ehrlicher Vergleich für eine Saison.

    Returns ein ``comparison``-dict in EXAKT dem Format, das
    ``viz.plot_rps_comparison`` / ``viz.plot_calibration`` erwarten.
    """
    wf = walkforward_predictions(
        df_full, season, use_xg=use_xg, tau=tau, gamma=gamma, eps=eps,
        holdout_frac=holdout_frac, verbose=verbose, **wf_kwargs,
    )
    df_season = wf["df_season"]
    probs_model = wf["probs_model"]
    cutoff, n = wf["cutoff"], wf["n"]

    # Buchmacher + Baselines (gleiche Saison-Indexierung)
    probs_bm = bookmaker_probs_from_df(df_season)
    probs_uni = uniform_baseline(n)
    df_pre = df_full[df_full["Date"] < df_season["Date"].min()]
    probs_emp = (empirical_baseline(df_pre, n) if len(df_pre)
                 else uniform_baseline(n))

    outcomes = np.array([outcome_index(int(r["FTHG"]), int(r["FTAG"]))
                         for _, r in df_season.iterrows()], dtype=int)

    # Fairer Head-to-Head: nur Holdout-Zeilen, in denen SOWOHL Modell ALS
    # AUCH Buchmacher eine Vorhersage haben → identisches n für alle.
    hold = np.zeros(n, dtype=bool)
    hold[cutoff:] = True
    fin_m = np.all(np.isfinite(probs_model), axis=1)
    fin_b = np.all(np.isfinite(probs_bm), axis=1)
    common = hold & fin_m & fin_b

    comparison = {
        "model":     evaluate_predictions(probs_model[common], outcomes[common]),
        "bookmaker": evaluate_predictions(probs_bm[common],    outcomes[common]),
        "empirical": evaluate_predictions(probs_emp[common],   outcomes[common]),
        "uniform":   evaluate_predictions(probs_uni[common],   outcomes[common]),
        # Für die Plots (slicen [cutoff:n] und droppen NaN selbst):
        "probs_model":     probs_model,
        "probs_bookmaker": probs_bm,
        "outcomes":        outcomes,
        "match_ids":       df_season[MATCH_ID_COL].astype(str).to_numpy(),
        "dates":           pd.to_datetime(df_season["Date"]).to_numpy(),
        "home_teams":      df_season["HomeTeam"].astype(str).to_numpy(),
        "away_teams":      df_season["AwayTeam"].astype(str).to_numpy(),
        "home_goals":      df_season["FTHG"].to_numpy(dtype=int),
        "away_goals":      df_season["FTAG"].to_numpy(dtype=int),
        "cutoff":          cutoff,
        "n":               n,
        "n_common":        int(common.sum()),
        "rhat_max":        wf.get("rhat_max"),
        "historical_home_prior": wf.get("historical_home_prior", {}),
        "historical_home_info": wf.get("historical_home_info", {}),
    }
    return comparison


def add_paper_baseline(comparison: dict, df_full: pd.DataFrame, season: str, *,
                       holdout_frac: float, n_chains: int, jitter_sd: float,
                       verbose: bool = True) -> dict:
    """Ergänzt ``comparison`` um das PAPER-STANDARDMODELL (Schlüssel 'paper').

    Identische Walk-Forward-Maschine, aber alle v2-Erweiterungen AUS: Tore
    statt xG, feste Paper-Hyperparameter (τ=100, γ=0.10, ε=0.20), 0-Prior.
    Bewertung auf EXAKT denselben Holdout-Spielen wie das v2-Modell — die
    Refit-Spieltage sind modell-unabhängig (Präfix-Split hängt nur an den
    Daten), also ist probs_paper genau dort endlich, wo probs_model es ist.
    """
    if verbose:
        print("\n  Paper-Standardmodell (Baseline: Tore + Paper-Defaults, "
              "0-Prior) ...")
    wf = walkforward_predictions(
        df_full, season, use_xg=False,
        tau=DEFAULT_TAU, gamma=DEFAULT_GAMMA, eps=DEFAULT_EPSILON,
        holdout_frac=holdout_frac, verbose=verbose,
        n_chains=n_chains, jitter_sd=jitter_sd,
        market_values=None, market_kappa=0.0, continuous_xg=False,
        use_team_home_advantage=False,
    )
    probs_paper = wf["probs_model"]
    out = comparison["outcomes"]
    n, cutoff = comparison["n"], comparison["cutoff"]
    hold = np.zeros(n, dtype=bool)
    hold[cutoff:] = True
    common = (hold
              & np.all(np.isfinite(comparison["probs_model"]), axis=1)
              & np.all(np.isfinite(comparison["probs_bookmaker"]), axis=1)
              & np.all(np.isfinite(probs_paper), axis=1))
    comparison["paper"] = evaluate_predictions(probs_paper[common], out[common])
    comparison["probs_paper"] = probs_paper
    return comparison


def print_comparison(comparison: dict, season: str):
    nc = comparison.get("n_common", comparison["model"].get("n", 0))
    print(f"\n  Ehrliche Walk-Forward-Evaluation — Saison {season}")
    print(f"  Holdout (Modell ∩ Buchmacher): {nc} Spiele")
    print(f"\n  {'Quelle':<24} {'RPS':>10} {'LogLoss':>10} {'Brier':>10}")
    print(f"  {'-'*56}")
    labels = {"model": "Modell (v2)", "paper": "Paper-Standard (2000)",
              "bookmaker": "Buchmacher", "empirical": "Basisrate",
              "uniform": "Uniform"}
    for key in ["model", "paper", "bookmaker", "empirical", "uniform"]:
        if key not in comparison:
            continue
        m = comparison[key]
        print(f"  {labels[key]:<24} {m['rps']:>10.4f} {m['log_loss']:>10.4f} "
              f"{m['brier']:>10.4f}")

    rps_m = comparison["model"]["rps"]
    rps_b = comparison["bookmaker"]["rps"]

    if "paper" in comparison:
        rps_p = comparison["paper"]["rps"]
        if np.isfinite(rps_m) and np.isfinite(rps_p) and rps_p > 0:
            impr = (rps_p - rps_m) / rps_p * 100.0
            verdict_p = "v2 besser" if impr > 0 else "Paper besser"
            print(f"\n  v2-Erweiterungen vs. Paper-Standard: "
                  f"ΔRPS = {rps_p - rps_m:+.4f} ({impr:+.1f} % rel. — {verdict_p})")
    if np.isfinite(rps_m) and np.isfinite(rps_b) and rps_b > 0:
        rel = (rps_b - rps_m) / rps_b * 100.0
        verdict = ("Modell besser" if rel > 0 else "Buchmacher besser")
        # grober Standardfehler der RPS-Differenz (≈ σ_RPS/√n, σ_RPS≈0.21)
        se = 0.21 / np.sqrt(max(1, nc))
        print(f"\n  ΔRPS = {rps_b - rps_m:+.4f}  ({rel:+.1f} % rel. — {verdict})")
        print(f"  grober SE des mittleren RPS ≈ {se:.4f}  → ein Vorsprung muss "
              f"deutlich größer als ~{2*se:.4f} sein, um auf EINER Saison real "
              f"zu wirken.")

    rh = comparison.get("rhat_max")
    if rh is not None and np.isfinite(rh):
        status = "OK (<1.05)" if rh < 1.05 else "WARNUNG ≥1.05 — mehr Iterationen nötig"
        print(f"\n  Max R-hat über alle Refits: {rh:.3f}  [{status}]")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    global OUT
    t_start = time.time()
    season_tag = ANALYSIS_SEASON.replace("/", "_")
    OUT = timestamped_output_dir(f"backtest_{season_tag}")
    print(f"  Output directory: {OUT.resolve()}")
    print("=" * 64)
    print(" EHRLICHER WALK-FORWARD-BUCHMACHER-VERGLEICH")
    print("=" * 64)

    if (USE_CONTINUOUS_XG or USE_REAL_XG) and not USE_XG:
        raise ValueError(
            "USE_CONTINUOUS_XG/USE_REAL_XG brauchen USE_XG=True (sie laufen "
            "auf den xG-Spalten). Bitte USE_XG aktivieren oder die Flags "
            "deaktivieren.")

    # 1) Daten
    df = load_bundesliga(START_YEAR, END_YEAR, with_extras=True)
    df = extract_bookmaker_probs(df)

    # 2) xG-β NUR auf Trainings-Saisons (vor der Analyse-Saison) fitten,
    #    ohne den gemeinsamen Platten-Cache zu überschreiben (Leck #2).
    if USE_XG:
        seasons = sorted(df["Season"].unique())
        train_seasons = [s for s in seasons if s < ANALYSIS_SEASON]
        df_train = df[df["Season"].isin(train_seasons)]
        print(f"  xG-β aus {len(train_seasons)} Trainings-Saisons "
              f"(vor {ANALYSIS_SEASON}) — leckfrei.")
        beta_off, beta_on = fit_xg_weights(df_train, force=True, cache=False)
        df = add_xg_columns(df, beta_off, beta_on)
        if USE_REAL_XG:
            from real_xg import add_real_xg_columns
            df = add_real_xg_columns(df)

    # 3) Hyperparameter: getunt (Cache) oder feste Paper-Defaults
    if USE_TUNED_HYPERPARAMS:
        tau, gamma, eps = resolve_hyperparams(ANALYSIS_SEASON, USE_XG)
    else:
        tau, gamma, eps = DEFAULT_TAU, DEFAULT_GAMMA, DEFAULT_EPSILON
        print(f"  Feste Paper-Hyperparameter (leckfrei): "
              f"τ={tau}, γ={gamma}, ε={eps}")

    # 4) Numba-JIT einmal vorkompilieren (sequenzieller Lauf, kein Pool)
    print("  Numba-JIT vorkompilieren ...", end="", flush=True)
    warmup_jit()
    print(" OK")

    # 5) Walk-Forward-Evaluation
    n_chains = N_CHAINS if USE_MULTI_CHAIN else 1
    if n_chains > 1:
        print(f"  {n_chains} parallele Ketten pro Refit (R-hat-Diagnostik aktiv).")

    market_values = None
    market_kappa = 0.0
    if USE_MARKET_PRIOR:
        from market_value import team_market_values
        market_values = team_market_values(ANALYSIS_SEASON, csv_path=MARKET_VALUE_CSV)
        market_kappa = MARKET_PRIOR_KAPPA
        print(f"  Marktwert-Prior aktiv (κ={market_kappa}, Werte as-of "
              f"Saisonstart = leak-frei): {len(market_values)} Teams.")

    comparison = evaluate_season_walkforward(
        df, ANALYSIS_SEASON, use_xg=USE_XG,
        tau=tau, gamma=gamma, eps=eps, holdout_frac=HOLDOUT_FRAC,
        n_chains=n_chains, jitter_sd=CHAIN_JITTER_SD,
        market_values=market_values, market_kappa=market_kappa,
    )
    if INCLUDE_PAPER_BASELINE:
        add_paper_baseline(comparison, df, ANALYSIS_SEASON,
                           holdout_frac=HOLDOUT_FRAC, n_chains=n_chains,
                           jitter_sd=CHAIN_JITTER_SD)
    print_comparison(comparison, ANALYSIS_SEASON)

    # 6) Plots (viz erst hier importieren → Ketten-Worker brauchen kein matplotlib)
    from viz import plot_rps_comparison, plot_calibration
    print("\n  → RPS-Vergleich-Plot ...")
    plot_rps_comparison(comparison, OUT / "13_rps_vergleich_walkforward.png")
    print("  → Kalibrierungskurven ...")
    plot_calibration(comparison, OUT / "14_kalibrierung_walkforward.png")

    print(f"\n  Fertig in {(time.time()-t_start)/60:.1f} min. "
          f"Plots in {OUT.resolve()}")


if __name__ == "__main__":
    main()
