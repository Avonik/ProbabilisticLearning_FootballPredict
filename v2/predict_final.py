"""
predict_final.py
================
Vorhersage für ein einzelnes Spiel (z.B. Pokalfinale) aus dem MCMC-Cache.

Einstellungen unten unter "=== Konfiguration ===" anpassen.
Aufruf: python predict_final.py
"""

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: nur Datei-Export, kein GUI-Fenster (Tk)
import matplotlib.pyplot as plt
import numpy as np

# UTF-8 stdout (Windows)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from data import load_bundesliga, extract_bookmaker_probs
from xg import fit_xg_weights, add_xg_columns
from model import (
    MAX_GOALS, build_league, compute_lambdas,
    match_likelihood, predict_outcome_probs,
)

# === Konfiguration ===
TEAM1           = "Bayern Munich"
TEAM2           = "Stuttgart"
NEUTRAL_VENUE   = True
ANALYSIS_SEASON = "2025/26"

# Muss zur Konfiguration in run.py passen
USE_XG     = True
USE_TEAM_HOME_ADVANTAGE = True
TEAM_HOME_ADV_PRIOR_SD = 0.15
USE_HISTORICAL_HOME_PRIOR = True
N_CHAINS   = 4
N_ITER     = 500_000
TUNE_MODE  = "paper"

# Pfad relativ zum Skript, nicht zum Arbeitsverzeichnis – sonst wird der
# Cache beim Start aus der Repo-Wurzel nicht gefunden.
CACHE_DIR  = Path(__file__).resolve().parent / "cache_v2"
CACHE      = CACHE_DIR / (
    f"mcmc_{ANALYSIS_SEASON.replace('/','_')}_{N_CHAINS}c_{N_ITER}i_"
    f"{'xg' if USE_XG else 'goals'}"
    f"{'_teamhome' if USE_TEAM_HOME_ADVANTAGE else ''}"
    f"{'_hist3d065' if USE_TEAM_HOME_ADVANTAGE and USE_HISTORICAL_HOME_PRIOR else ''}.pkl"
)
TUNE_CACHE = CACHE_DIR / f"tune_{TUNE_MODE}_{ANALYSIS_SEASON.replace('/','_')}_{'xg' if USE_XG else 'goals'}.pkl"
# =====================


def load_samples(cache: Path, season: str) -> tuple:
    if not cache.exists():
        # Fallback: nach legacy-Caches schauen
        legacy = Path(f"mcmc_cache_{N_CHAINS}chains_{N_ITER}iter.pkl")
        if legacy.exists():
            cache = legacy
        else:
            print(f"Kein Cache gefunden ({cache}). Bitte zuerst run.py ausführen.")
            sys.exit(1)
    print(f"Lade Cache: {cache}")
    with open(cache, "rb") as f:
        samples = pickle.load(f)
    if USE_TEAM_HOME_ADVANTAGE and "home_advantage" not in samples:
        print("Der Cache enthält noch keine teamspezifischen Heimeffekte. "
              "Bitte run.py mit der aktuellen Konfiguration neu ausführen.")
        sys.exit(1)

    df = load_bundesliga(2024, 2026, fallback_synthetic=False, with_extras=True)
    df = extract_bookmaker_probs(df)
    if USE_XG:
        beta_off, beta_on = fit_xg_weights(df)
        df = add_xg_columns(df, beta_off, beta_on)

    df_season = df[df["Season"] == season].copy()
    if df_season.empty:
        season_fb = sorted(df["Season"].unique())[-1]
        print(f"Saison '{season}' nicht gefunden, verwende {season_fb}")
        df_season = df[df["Season"] == season_fb].copy()

    # Falls Tuning-Cache existiert, nutze die getunten Hyperparameter
    tau, gamma, eps = 100.0, 0.10, 0.20
    if TUNE_CACHE.exists():
        with open(TUNE_CACHE, "rb") as f:
            tr = pickle.load(f)
        b = tr["best"]
        tau, gamma, eps = float(b["tau"]), float(b["gamma"]), float(b["epsilon"])
        print(f"Verwende getunte Hyperparameter: tau={tau}, gamma={gamma}, eps={eps}")

    L = build_league(df_season, use_xg=USE_XG,
                     tau=tau, gamma=gamma, epsilon=eps,
                     use_team_home_advantage=USE_TEAM_HOME_ADVANTAGE,
                     home_adv_prior_sd=TEAM_HOME_ADV_PRIOR_SD)
    return samples, L


def predict_match(samples, L, team1: str, team2: str, neutral: bool) -> dict:
    for name in (team1, team2):
        if name not in L.teams:
            print(f"Team '{name}' nicht gefunden.")
            print(f"Verfuegbare Teams: {sorted(L.teams)}")
            sys.exit(1)

    t1, t2 = L.teams.index(team1), L.teams.index(team2)
    c_x, c_y = L.c_x, L.c_y
    if neutral:
        c_x = c_y = (c_x + c_y) / 2

    p1_list, pd_list, p2_list = [], [], []
    score_mat = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))

    home_samples = samples.get("home_advantage")
    for sample_i, (sa, sd) in enumerate(zip(samples["attack"], samples["defense"])):
        a1, d1 = float(sa[t1][-1]), float(sd[t1][-1])
        a2, d2 = float(sa[t2][-1]), float(sd[t2][-1])
        home_adv = (float(home_samples[sample_i][t1])
                    if home_samples is not None and not neutral else 0.0)

        p1, pd, p2 = predict_outcome_probs(
            a1, d1, a2, d2, c_x, c_y,
            gamma=L.gamma, eps=L.epsilon, home_advantage=home_adv,
        )
        p1_list.append(p1)
        pd_list.append(pd)
        p2_list.append(p2)

        lam_x, lam_y = compute_lambdas(
            a1, d1, a2, d2, c_x, c_y,
            gamma=L.gamma, home_advantage=home_adv,
        )
        lam_avg_x = np.exp(c_x)
        lam_avg_y = np.exp(c_y)
        for x in range(MAX_GOALS + 1):
            for y in range(MAX_GOALS + 1):
                score_mat[x, y] += match_likelihood(
                    x, y, lam_x, lam_y, lam_avg_x, lam_avg_y, eps=L.epsilon,
                )

    score_mat /= score_mat.sum()
    return {
        "p1": np.mean(p1_list),
        "pd": np.mean(pd_list),
        "p2": np.mean(p2_list),
        "score_mat": score_mat,
    }


def plot_result(res: dict, team1: str, team2: str, neutral: bool, samples):
    venue = "Neutraler Platz" if neutral else f"{team1} Heim"
    mat = res["score_mat"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={"width_ratios": [2, 3]})

    ax = axes[0]
    labels = [f"{team1}\ngewinnt", "Unent-\nschieden", f"{team2}\ngewinnt"]
    values = [res["p1"], res["pd"], res["p2"]]
    colors = ["#1f77b4", "#aaaaaa", "#d62728"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black",
                  linewidth=0.7, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{val:.1%}",
                ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylim(0, max(values) + 0.12)
    ax.set_ylabel("Wahrscheinlichkeit")
    ax.set_title(f"{team1} vs {team2}\n({venue})", fontsize=12)

    ax = axes[1]
    im = ax.imshow(mat.T, origin="lower", cmap="Blues",
                   vmin=0, vmax=mat.max())
    ax.set_xlabel(f"Tore {team1}", fontsize=11)
    ax.set_ylabel(f"Tore {team2}", fontsize=11)
    ax.set_xticks(range(MAX_GOALS + 1))
    ax.set_yticks(range(MAX_GOALS + 1))
    ax.set_title("Wahrscheinlichste Ergebnisse", fontsize=12)

    for x in range(MAX_GOALS + 1):
        for y in range(MAX_GOALS + 1):
            v = mat[x, y]
            color = "white" if v > mat.max() * 0.5 else "black"
            ax.text(x, y, f"{v*100:.1f}%", ha="center", va="center",
                    fontsize=8, color=color)

    for i in range(MAX_GOALS + 1):
        ax.add_patch(plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor="gray", linewidth=1.5, linestyle="--"
        ))

    fig.colorbar(im, ax=axes[1], shrink=0.8, label="Wahrscheinlichkeit")
    fig.suptitle(f"Bayesianische Spielvorhersage  |  {len(samples['attack'])} Posterior-Samples",
                 fontsize=13, y=1.01)
    fig.tight_layout()

    out_path = Path("output") / f"vorhersage_{team1.replace(' ', '_')}_vs_{team2.replace(' ', '_')}.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Plot gespeichert: {out_path}")
    plt.show()


if __name__ == "__main__":
    samples, L = load_samples(CACHE, ANALYSIS_SEASON)

    print(f"\nVorhersage: {TEAM1} vs {TEAM2}")
    print(f"Austragungsort: {'Neutral' if NEUTRAL_VENUE else TEAM1 + ' Heim'}")
    print(f"Posterior-Samples: {len(samples['attack'])}")

    res = predict_match(samples, L, TEAM1, TEAM2, NEUTRAL_VENUE)

    print(f"\n  P({TEAM1} gewinnt): {res['p1']:.1%}")
    print(f"  P(Unentschieden):   {res['pd']:.1%}")
    print(f"  P({TEAM2} gewinnt): {res['p2']:.1%}")

    top_scores = sorted(
        [(res["score_mat"][x, y], x, y)
         for x in range(MAX_GOALS + 1)
         for y in range(MAX_GOALS + 1)],
        reverse=True,
    )[:5]
    print(f"\n  Wahrscheinlichste Ergebnisse:")
    for prob, x, y in top_scores:
        print(f"    {x}:{y}  ->  {prob:.1%}")

    plot_result(res, TEAM1, TEAM2, NEUTRAL_VENUE, samples)
