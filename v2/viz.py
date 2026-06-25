"""
viz.py
======
Statische Visualisierungen für die Rue-Salvesen-Pipeline (v2).

Bestehende Plots:
  01  Tor-Verteilung
  02  Heimvorteil über Zeit
  03  Brownsche Bewegung
  04  Tor-Modell-Aufbau
  05  DAG-Struktur
  06  MCMC-Trace
  07  Stärken-Evolution
  08  Team-Profile
  09  Rangverteilung (Replay)
  10  Überraschungsspiele

Neu in v2 (xG, Tuning, RPS):
  11  xG vs. Tore (Scatter + Verteilungen)
  12  Hyperparameter-Scan (Heatmap)
  13  RPS-Vergleich Modell / Buchmacher / Baselines
  14  Kalibrierungskurven
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: nur Datei-Export, kein GUI-Fenster (Tk)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Stil
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


# ─────────────────────────────────────────────────────────────────────
# 1. Datenexploration
# ─────────────────────────────────────────────────────────────────────

def plot_goal_distribution(df: pd.DataFrame, save_path):
    """Histogramm der Heim- und Auswärtstore (analog Fig. 1 im Paper)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    max_g = 8

    for ax, col, title, color in [
        (axes[0], "FTHG", "Heimtore", PALETTE[0]),
        (axes[1], "FTAG", "Auswärtstore", PALETTE[1]),
    ]:
        vals = df[col].clip(upper=max_g).values
        bins = np.arange(-0.5, max_g + 1.5, 1)
        counts, _ = np.histogram(vals, bins=bins)
        freqs = counts / counts.sum()
        ax.bar(range(max_g + 1), freqs, color=color, edgecolor="white", alpha=0.9)
        ax.set_xlabel("Tore")
        ax.set_title(f"{title} (n={len(df)} Spiele)")
        mean = df[col].mean()
        ax.axvline(mean, color="black", linestyle="--", alpha=0.6,
                   label=f"⌀ {mean:.2f}")
        ax.legend()
    axes[0].set_ylabel("Relative Häufigkeit")
    fig.suptitle("Verteilung der Tore in der Bundesliga", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_home_advantage_over_time(df: pd.DataFrame, save_path):
    """Heimvorteil pro Saison: Heimsieg-Quote und Tor-Differenz."""
    grp = df.groupby("Season").agg(
        home_win_rate=("FTR", lambda s: (s == "H").mean()),
        draw_rate=("FTR", lambda s: (s == "D").mean()),
        away_win_rate=("FTR", lambda s: (s == "A").mean()),
        avg_home_goals=("FTHG", "mean"),
        avg_away_goals=("FTAG", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(grp["Season"], grp["home_win_rate"], "-o",
                 color=PALETTE[0], label="Heimsieg")
    axes[0].plot(grp["Season"], grp["draw_rate"], "-o",
                 color=PALETTE[2], label="Unentschieden")
    axes[0].plot(grp["Season"], grp["away_win_rate"], "-o",
                 color=PALETTE[1], label="Auswärtssieg")
    axes[0].set_title("Ergebnisanteile pro Saison")
    axes[0].set_ylabel("Anteil")
    axes[0].legend(loc="upper right")
    axes[0].tick_params(axis="x", rotation=70)

    axes[1].plot(grp["Season"], grp["avg_home_goals"], "-o",
                 color=PALETTE[0], label="Heimtore (⌀)")
    axes[1].plot(grp["Season"], grp["avg_away_goals"], "-o",
                 color=PALETTE[1], label="Auswärtstore (⌀)")
    axes[1].fill_between(grp["Season"], grp["avg_home_goals"], grp["avg_away_goals"],
                         alpha=0.15, color="gray", label="Heimvorteil")
    axes[1].set_title("Durchschnittliche Tore pro Spiel")
    axes[1].set_ylabel("Tore")
    axes[1].legend(loc="upper right")
    axes[1].tick_params(axis="x", rotation=70)

    fig.suptitle("Heimvorteil über alle Bundesligasaisons", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# 2. Modell-Intuition
# ─────────────────────────────────────────────────────────────────────

def plot_brownian_motion_examples(save_path, n_paths: int = 6, seed: int = 42,
                                  tau: float = 100.0):
    """Beispielpfade einer Brownschen Bewegung."""
    rng = np.random.default_rng(seed)
    n_steps = 200
    days = np.linspace(0, 270, n_steps)
    sigma2 = 1 / 37

    fig, ax = plt.subplots(figsize=(10, 5))
    for k in range(n_paths):
        path = np.zeros(n_steps)
        for i in range(1, n_steps):
            dt = days[i] - days[i - 1]
            path[i] = path[i - 1] + rng.normal(0, np.sqrt(dt / tau * sigma2))
        ax.plot(days, path, alpha=0.8, color=PALETTE[k % len(PALETTE)],
                label=f"Team {k+1}")

    ax.axhline(0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("Tage seit Saisonbeginn")
    ax.set_ylabel("Angriffsstärke a(t)")
    ax.set_title("Beispielhafte Brownsche Pfade der Angriffsstärke\n"
                 f"(τ={tau:.0f} Tage, σ²={sigma2:.4f}) — so 'wandern' Stärken im Prior")
    ax.legend(ncol=3, loc="lower center")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_tor_modell_intuition(save_path, epsilon: float = 0.2):
    """Zeigt wie λ → Poisson → Dixon-Coles → Mischung das Tormodell formen."""
    from scipy.stats import poisson as scipy_poisson
    from model import dc_correction

    lam_x, lam_y = 1.6, 1.1
    max_g = 6
    x = np.arange(max_g + 1)
    y = np.arange(max_g + 1)

    p_naive = np.outer(scipy_poisson.pmf(x, lam_x), scipy_poisson.pmf(y, lam_y))
    p_dc = p_naive.copy()
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p_dc[i, j] *= dc_correction(i, j, lam_x, lam_y)
    p_dc /= p_dc.sum()
    lam_x_avg, lam_y_avg = 1.5, 1.1
    p_avg = np.outer(scipy_poisson.pmf(x, lam_x_avg), scipy_poisson.pmf(y, lam_y_avg))
    p_avg /= p_avg.sum()
    p_mix = (1 - epsilon) * p_dc + epsilon * p_avg

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, mat, title in zip(
        axes,
        [p_naive / p_naive.sum(), p_dc, p_mix],
        [f"Naives Poisson\n(λ_h={lam_x}, λ_a={lam_y})",
         "+ Dixon-Coles Korrektur\n(0:0 und 1:1 verstärkt)",
         f"+ Mischung mit Durchschnittsspiel\n(ε={epsilon:.2f})"],
    ):
        im = ax.imshow(mat.T, origin="lower", cmap="viridis", aspect="equal")
        ax.set_xlabel("Heimtore")
        ax.set_ylabel("Auswärtstore")
        ax.set_title(title)
        ax.set_xticks(range(max_g + 1))
        ax.set_yticks(range(max_g + 1))
        for i in range(max_g + 1):
            for j in range(max_g + 1):
                ax.text(i, j, f"{mat[i,j]*100:.0f}",
                        ha="center", va="center",
                        color="white" if mat[i, j] < mat.max() * 0.5 else "black",
                        fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle("Tormodell: Schritt für Schritt — Werte in % (gerundet)", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_dag_structure(save_path):
    """Vereinfachter DAG (analog Fig. 2 im Paper)."""
    fig, ax = plt.subplots(figsize=(12, 7))

    teams = ["A", "B", "C", "D"]
    times = [0, 1, 2, 3]
    x_pos = {t: i * 2 for i, t in enumerate(teams)}
    y_pos = {tau: -tau * 2 for tau in times}

    for tau in times:
        for t in teams:
            ax.scatter(x_pos[t], y_pos[tau], s=900, c=PALETTE[teams.index(t)],
                       edgecolors="black", linewidth=1.2, zorder=3)
            ax.text(x_pos[t], y_pos[tau], t, ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white", zorder=4)
        ax.text(-1.5, y_pos[tau], f"t={tau}", ha="right", va="center",
                fontsize=11, fontstyle="italic")

    for tau in times[:-1]:
        for t in teams:
            ax.annotate("", xy=(x_pos[t], y_pos[tau + 1] + 0.4),
                        xytext=(x_pos[t], y_pos[tau] - 0.4),
                        arrowprops=dict(arrowstyle="->", color="gray",
                                        alpha=0.6, lw=1.5))

    match_pairs = [
        (0, "A", "B"), (0, "C", "D"),
        (1, "A", "C"), (1, "B", "D"),
        (2, "A", "D"), (2, "B", "C"),
        (3, "A", "B"), (3, "C", "D"),
    ]
    for tau, t1, t2 in match_pairs:
        mx = (x_pos[t1] + x_pos[t2]) / 2
        my = y_pos[tau] - 1.0
        marker = "X-Y" if tau < 3 else "?-?"
        color = "#f0f0f0" if tau < 3 else "#ffe4b5"
        ax.scatter(mx, my, s=700, c=color, edgecolors="black",
                   linewidth=1, marker="s", zorder=3)
        ax.text(mx, my, marker, ha="center", va="center", fontsize=9)
        for t in (t1, t2):
            ax.annotate("", xy=(mx, my + 0.3),
                        xytext=(x_pos[t], y_pos[tau] - 0.4),
                        arrowprops=dict(arrowstyle="->", color=PALETTE[teams.index(t)],
                                        alpha=0.7, lw=1.2))

    ax.set_xlim(-3, 8)
    ax.set_ylim(-8, 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Modellstruktur als gerichteter azyklischer Graph (DAG)\n"
                 "Pfeile: Eltern-Kind-Beziehungen — alles ist verknüpft, "
                 "daher gemeinsame Inferenz nötig",
                 fontsize=12)

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=14, label="Teamstärken (a, d)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#f0f0f0",
               markeredgecolor="black", markersize=14, label="beobachtetes Spiel"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#ffe4b5",
               markeredgecolor="black", markersize=14, label="vorherzusagendes Spiel"),
    ]
    ax.legend(handles=legend_elems, loc="upper right")

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# 3. MCMC-Diagnostik
# ─────────────────────────────────────────────────────────────────────

def plot_mcmc_trace(samples, L, save_path, n_teams_show: int = 4):
    chains = samples.get("chains") or []
    has_chains = len(chains) > 1

    if has_chains:
        team_ids = list(chains[0]["attack"][0].keys())
        n_per_chain = len(chains[0]["attack"])
        rhat_str = (f"  |  R-hat max = {samples['rhat_max']:.3f}"
                    if "rhat_max" in samples else "")
        header = f"{len(chains)} Ketten à {n_per_chain} Samples{rhat_str}"
    else:
        team_ids = list(samples["attack"][0].keys())
        n_per_chain = len(samples["attack"])
        header = f"{n_per_chain} Samples (Einzelkette)"

    show = team_ids[:n_teams_show]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)

    for k, team in enumerate(show):
        color = PALETTE[k % len(PALETTE)]
        if has_chains:
            for chain in chains:
                a_mean = [s[team].mean() for s in chain["attack"]]
                d_mean = [s[team].mean() for s in chain["defense"]]
                axes[0].plot(a_mean, color=color, alpha=0.45, linewidth=0.8)
                axes[1].plot(d_mean, color=color, alpha=0.45, linewidth=0.8)
            axes[0].plot([], [], color=color, alpha=0.9, linewidth=1.5,
                         label=L.teams[team])
        else:
            a_mean = [s[team].mean() for s in samples["attack"]]
            d_mean = [s[team].mean() for s in samples["defense"]]
            axes[0].plot(a_mean, color=color, alpha=0.85, label=L.teams[team])
            axes[1].plot(d_mean, color=color, alpha=0.85)

    for ax, ylabel in zip(axes, ["Mittlere Angriffsstärke", "Mittlere Abwehrstärke"]):
        ax.axhline(0, color="black", linestyle=":", alpha=0.4)
        ax.set_ylabel(ylabel)

    axes[0].legend(ncol=4, loc="upper right", fontsize=9)
    axes[1].set_xlabel("MCMC-Iteration (pro Kette)")

    fig.suptitle(
        f"MCMC-Konvergenz ({header})\n"
        "Gleiche Farbe = selbes Team, jede Linie = eine Kette",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_strength_evolution(samples, L, save_path, top_k: int = 6):
    from mcmc import posterior_means
    mean_attack, mean_defense, _ = posterior_means(samples)

    overall = []
    for team in range(len(L.teams)):
        overall.append((mean_attack[team].mean() + mean_defense[team].mean(), team))
    overall.sort(reverse=True)
    top_teams = [t for _, t in overall[:top_k]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    for k, team in enumerate(top_teams):
        match_indices = L.team_matches[team]
        dates = L.dates[match_indices]
        color = PALETTE[k % len(PALETTE)]
        axes[0].plot(dates, mean_attack[team], "-o", color=color,
                     markersize=4, label=L.teams[team], alpha=0.9)
        axes[1].plot(dates, mean_defense[team], "-o", color=color,
                     markersize=4, label=L.teams[team], alpha=0.9)

    axes[0].set_title("Posteriori-Angriffsstärke über die Saison")
    axes[0].set_ylabel("a(t)")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].set_title("Posteriori-Abwehrstärke über die Saison")
    axes[1].set_ylabel("d(t)")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle(f"Stärken-Evolution: Top {top_k} Teams", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# 4. Ergebnisse und Retrospektive
# ─────────────────────────────────────────────────────────────────────

def plot_predicted_rankings(samples, L, save_path, n_replays: int = 1000,
                            seed: int = 0):
    """Posteriori-Rangverteilung (analog Fig. 5 im Paper)."""
    from model import predict_outcome_probs
    rng = np.random.default_rng(seed)

    n_teams = len(L.teams)
    rank_counts = np.zeros((n_teams, n_teams), dtype=int)
    n_samples = len(samples["attack"])

    actual_pts = np.zeros(n_teams)
    actual_gd = np.zeros(n_teams)
    for m in range(len(L.home_goals)):
        h, a = L.home_idx[m], L.away_idx[m]
        hg, ag = L.home_goals[m], L.away_goals[m]
        if hg > ag:
            actual_pts[h] += 3
        elif hg < ag:
            actual_pts[a] += 3
        else:
            actual_pts[h] += 1
            actual_pts[a] += 1
        actual_gd[h] += hg - ag
        actual_gd[a] += ag - hg
    actual_rank = np.argsort(np.argsort(-(actual_pts + 0.001 * actual_gd)))

    for _ in range(n_replays):
        s_idx = rng.integers(0, n_samples)
        pts = np.zeros(n_teams)
        gd = np.zeros(n_teams)
        for m in range(len(L.home_goals)):
            h, a = L.home_idx[m], L.away_idx[m]
            lh = L.local_idx[(int(h), m)]
            la = L.local_idx[(int(a), m)]
            a_h = samples["attack"][s_idx][h][lh]
            d_h = samples["defense"][s_idx][h][lh]
            a_a = samples["attack"][s_idx][a][la]
            d_a = samples["defense"][s_idx][a][la]
            p_home, p_draw, p_away = predict_outcome_probs(
                a_h, d_h, a_a, d_a, L.c_x, L.c_y,
                gamma=L.gamma, eps=L.epsilon,
            )
            r = rng.random()
            if r < p_home:
                pts[h] += 3
                gd_delta = 1
            elif r < p_home + p_draw:
                pts[h] += 1
                pts[a] += 1
                gd_delta = 0
            else:
                pts[a] += 3
                gd_delta = -1
            gd[h] += gd_delta
            gd[a] -= gd_delta
        rank = np.argsort(np.argsort(-(pts + 0.001 * gd)))
        for t in range(n_teams):
            rank_counts[t, rank[t]] += 1

    fig, ax = plt.subplots(figsize=(13, 6.5))
    order = np.argsort(actual_rank)
    expected_rank = (rank_counts * np.arange(n_teams)[None, :]).sum(1) / n_replays

    for k, team in enumerate(order):
        cumprob = np.cumsum(rank_counts[team]) / n_replays
        lo = int(np.searchsorted(cumprob, 0.05))
        hi = int(np.searchsorted(cumprob, 0.95))
        ax.plot([k, k], [lo + 1, hi + 1], color="gray", linewidth=2, alpha=0.5)
        ax.plot(k, expected_rank[team] + 1, "o", color=PALETTE[0],
                markersize=9, zorder=4)
        ax.plot(k, actual_rank[team] + 1, "x", color="red",
                markersize=10, markeredgewidth=2, zorder=5)

    ax.set_xticks(range(n_teams))
    ax.set_xticklabels([L.teams[t] for t in order], rotation=70, ha="right")
    ax.set_ylabel("Tabellenplatz")
    ax.invert_yaxis()
    ax.set_title("Retrospektive Rangverteilung\n"
                 "blau = Posteriori-Erwartung, rot = tatsächlicher Rang, "
                 "grau = 90% Intervall")

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[0],
               markersize=10, label="Erwarteter Rang (Posterior)"),
        Line2D([0], [0], marker="x", color="red", markersize=10,
               markeredgewidth=2, linestyle="", label="Tatsächlicher Rang"),
        Line2D([0], [0], color="gray", linewidth=2, label="90% Intervall"),
    ]
    ax.legend(handles=legend, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_surprise_matches(samples, L, df_season, save_path, top_n: int = 10):
    from mcmc import posterior_means
    _, _, p_outlier = posterior_means(samples)

    df = df_season.copy().reset_index(drop=True)
    df["p_outlier"] = p_outlier
    df["result"] = df["FTHG"].astype(str) + ":" + df["FTAG"].astype(str)
    top = df.nlargest(top_n, "p_outlier")

    fig, ax = plt.subplots(figsize=(11, 0.5 * top_n + 1.5))
    y_pos = np.arange(len(top))[::-1]
    colors = plt.cm.Reds(0.4 + 0.5 * top["p_outlier"].values)
    ax.barh(y_pos, top["p_outlier"], color=colors, edgecolor="black", linewidth=0.5)
    labels = [
        f"{r['HomeTeam']} {r['result']} {r['AwayTeam']}  "
        f"({pd.to_datetime(r['Date']).strftime('%d.%m.%y')})"
        for _, r in top.iterrows()
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("P(Ausreißer-Spiel)")
    ax.set_xlim(0, 1)
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(r["p_outlier"] + 0.01, y_pos[i], f"{r['p_outlier']:.2f}",
                va="center", fontsize=9)

    ax.set_title(f"Die {top_n} überraschendsten Spiele dieser Saison\n"
                 "(höchste Posterior-Wahrscheinlichkeit für die Mixture-Komponente δ=1)")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_team_ranking_summary(samples, L, save_path):
    from mcmc import posterior_means
    mean_attack, mean_defense, _ = posterior_means(samples)

    a_avg = np.array([mean_attack[t].mean() for t in range(len(L.teams))])
    d_avg = np.array([mean_defense[t].mean() for t in range(len(L.teams))])

    fig, ax = plt.subplots(figsize=(10, 7.5))
    overall = a_avg + d_avg
    sc = ax.scatter(a_avg, d_avg, c=overall, cmap="RdYlGn", s=240,
                    edgecolors="black", linewidth=0.8, zorder=3)
    for i, name in enumerate(L.teams):
        ax.annotate(name, (a_avg[i], d_avg[i]),
                    fontsize=9, ha="center", va="center",
                    xytext=(0, -16), textcoords="offset points")

    ax.axhline(0, color="black", linestyle=":", alpha=0.4)
    ax.axvline(0, color="black", linestyle=":", alpha=0.4)
    ax.set_xlabel("Mittlere Angriffsstärke (über Saison)")
    ax.set_ylabel("Mittlere Abwehrstärke (über Saison)")
    ax.set_title("Team-Profile: Angriff vs. Abwehr\n"
                 "(rechts oben = stark in beidem, links unten = schwach)")
    fig.colorbar(sc, ax=ax, label="Gesamtstärke")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# 5. Neu in v2 — xG, Tuning, RPS, Kalibrierung
# ─────────────────────────────────────────────────────────────────────

def plot_xg_vs_goals(df_season: pd.DataFrame, save_path,
                     beta_off: float | None = None,
                     beta_on: float | None = None):
    """
    Zweiteilig:
      links  — Scatter xG vs. Tore, je für Heim und Auswärts
      rechts — Verteilung: gerundete xG vs. tatsächliche Tore
    """
    if "xG_home" not in df_season.columns:
        # Falls xG fehlt: leeren Platzhalter speichern
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Keine xG-Daten für diese Saison verfügbar",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return

    has_xg = df_season.get("has_xg", pd.Series([True] * len(df_season))).astype(bool)
    sub = df_season[has_xg].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Scatter xG vs Tore ---
    rng = np.random.default_rng(0)
    jitter_h = rng.normal(0, 0.07, len(sub))
    jitter_a = rng.normal(0, 0.07, len(sub))
    axes[0].scatter(sub["xG_home"], sub["FTHG"] + jitter_h,
                    alpha=0.45, s=22, color=PALETTE[0], label="Heim")
    axes[0].scatter(sub["xG_away"], sub["FTAG"] + jitter_a,
                    alpha=0.45, s=22, color=PALETTE[1], label="Auswärts")
    lo, hi = 0, max(sub["xG_home"].max(), sub["xG_away"].max(),
                    sub["FTHG"].max(), sub["FTAG"].max()) + 0.5
    axes[0].plot([lo, hi], [lo, hi], "--", color="black", alpha=0.5,
                 label="x=y (perfekte Effizienz)")
    axes[0].set_xlabel("Proxy-xG (aus Schüssen)")
    axes[0].set_ylabel("Tatsächliche Tore (mit Jitter)")
    axes[0].set_title("xG-Treffsicherheit pro Spiel")
    axes[0].legend(loc="upper left", fontsize=9)
    # Korrelationen
    r_h = np.corrcoef(sub["xG_home"], sub["FTHG"])[0, 1]
    r_a = np.corrcoef(sub["xG_away"], sub["FTAG"])[0, 1]
    axes[0].text(0.98, 0.02,
                 f"corr Heim:  {r_h:.2f}\ncorr Ausw.:  {r_a:.2f}",
                 transform=axes[0].transAxes, ha="right", va="bottom",
                 fontsize=10, bbox=dict(boxstyle="round", facecolor="white",
                                         alpha=0.8))

    # --- Verteilung: gerundetes xG vs Tore ---
    max_g = 6
    bins = np.arange(-0.5, max_g + 1.5, 1)
    goals_all = np.concatenate([sub["FTHG"].values, sub["FTAG"].values])
    xg_all = np.concatenate([sub["xG_home"].values, sub["xG_away"].values])
    xg_rounded = np.clip(np.round(xg_all), 0, max_g)
    goals_clip = np.clip(goals_all, 0, max_g)

    counts_g, _ = np.histogram(goals_clip, bins=bins)
    counts_x, _ = np.histogram(xg_rounded, bins=bins)
    width = 0.4
    centers = np.arange(max_g + 1)
    axes[1].bar(centers - width/2, counts_g / counts_g.sum(), width,
                color=PALETTE[2], label="Tatsächliche Tore", alpha=0.9)
    axes[1].bar(centers + width/2, counts_x / counts_x.sum(), width,
                color=PALETTE[3], label="round(xG)", alpha=0.9)
    axes[1].set_xlabel("Tore")
    axes[1].set_ylabel("Relative Häufigkeit")
    axes[1].set_title("Verteilung: Tore vs. gerundetes xG")
    axes[1].legend()
    axes[1].set_xticks(centers)

    coef_text = ""
    if beta_off is not None and beta_on is not None:
        coef_text = f"  |  β_off={beta_off:.3f}, β_on={beta_on:.3f}"
    fig.suptitle(f"Proxy-xG-Kalibrierung (n={len(sub)} Spiele){coef_text}",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_hyperparameter_scan(tune_result: dict, save_path):
    """
    Visualisiert das Grid-Search-Ergebnis als Heatmaps.

    Für jede Wahl von ε wird eine τ×γ-Heatmap gezeichnet.
    """
    df = pd.DataFrame(tune_result["all_results"])
    metric = tune_result.get("metric", "score")
    best = tune_result["best"]
    epsilons = sorted(df["epsilon"].unique())

    fig, axes = plt.subplots(1, len(epsilons), figsize=(5 * len(epsilons), 4.5),
                             squeeze=False, sharey=True)
    axes = axes[0]

    vmin = df["score"].min()
    vmax = df["score"].max()

    for i, eps in enumerate(epsilons):
        sub = df[df["epsilon"] == eps]
        pivot = sub.pivot(index="gamma", columns="tau", values="score")
        ax = axes[i]
        im = ax.imshow(pivot.values, origin="lower", cmap="viridis",
                       aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{t:.0f}" for t in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{g:.2f}" for g in pivot.index])
        ax.set_xlabel("τ (Tage)")
        if i == 0:
            ax.set_ylabel("γ (psychologischer Effekt)")
        ax.set_title(f"ε = {eps:.2f}")
        # Werte einzeichnen
        for r, gv in enumerate(pivot.index):
            for c, tv in enumerate(pivot.columns):
                val = pivot.iloc[r, c]
                ax.text(c, r, f"{val:+.3f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if val < (vmin + vmax) / 2 else "black")
        # Bestes Tripel markieren
        if eps == best["epsilon"]:
            tau_pos = list(pivot.columns).index(best["tau"])
            gamma_pos = list(pivot.index).index(best["gamma"])
            ax.plot(tau_pos, gamma_pos, "*", color="red", markersize=22,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=10)

    fig.colorbar(im, ax=axes, label=metric, shrink=0.8)

    # Markiere "innerhalb 1 SE vom Besten" mit weißem Rand —
    # für Paper-Style-Tuning ist das die wichtige Info
    se = tune_result.get("se_approx")
    if se is not None:
        for i, eps in enumerate(epsilons):
            sub = df[df["epsilon"] == eps]
            pivot = sub.pivot(index="gamma", columns="tau", values="score")
            ax = axes[i]
            for r in range(len(pivot.index)):
                for c in range(len(pivot.columns)):
                    val = pivot.iloc[r, c]
                    if best["score"] - val < se:
                        ax.add_patch(plt.Rectangle(
                            (c - 0.5, r - 0.5), 1, 1,
                            fill=False, edgecolor="white", linewidth=2.0,
                            zorder=8,
                        ))

    # Untertitel je nach Tuning-Variante
    if "n_seasons" in tune_result:
        sub_info = (f"Paper-Style: {tune_result['n_seasons']} Saisons, "
                    f"{tune_result['n_holdout_total']} Holdout-Matches, "
                    f"~SE={tune_result.get('se_approx', 0):.4f}, "
                    f"{tune_result.get('n_close_to_best', '?')} Kombis "
                    f"innerhalb 1 SE")
    else:
        sub_info = (f"Single-Season: späte "
                    f"{int((1 - tune_result.get('frac_train', 0.7)) * 100)} % "
                    f"als Holdout")

    fig.suptitle(
        f"Hyperparameter-Tuning  ({metric})  —  {sub_info}\n"
        f"Beste Kombination:  τ={best['tau']:.0f},  γ={best['gamma']:.2f},  "
        f"ε={best['epsilon']:.2f}   →   {metric}={best['score']:+.4f}",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_rps_comparison(comparison: dict, save_path):
    """
    Balkendiagramm mit RPS / Log-Loss / Brier von Modell vs Buchmacher
    vs Baselines auf der Holdout-Hälfte der Saison.
    """
    labels_de = {
        "model": "Modell\n(Rue-Salvesen v2)",
        "paper": "Paper-Standard\n(Rue-Salvesen 2000)",
        "bookmaker": "Buchmacher\n(Pinnacle/Markt)",
        "empirical": "Basisrate\n(historisch)",
        "uniform": "Uniform\n(1/3, 1/3, 1/3)",
    }
    color_map = {"model": PALETTE[0], "paper": PALETTE[1], "bookmaker": PALETTE[3],
                 "empirical": PALETTE[2], "uniform": "#999999"}
    # 'paper' nur zeigen, wenn die Baseline berechnet wurde (optional).
    keys = [k for k in ["model", "paper", "bookmaker", "empirical", "uniform"]
            if k in comparison]
    colors = [color_map[k] for k in keys]

    metrics = [("rps", "RPS  (niedriger = besser)"),
               ("log_loss", "Log-Loss  (niedriger = besser)"),
               ("brier", "Brier  (niedriger = besser)")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (mkey, mtitle) in zip(axes, metrics):
        vals = [comparison[k].get(mkey, np.nan) for k in keys]
        bars = ax.bar([labels_de[k] for k in keys], vals,
                      color=colors, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(mtitle)
        ax.tick_params(axis="x", labelsize=9)

    n = comparison["model"].get("n", 0)
    fig.suptitle(f"Vorhersagequalität auf {n} Holdout-Spielen\n"
                 "(Modell näher am Buchmacher = stark, Modell besser als "
                 "Buchmacher = außergewöhnlich)",
                 y=1.03)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(comparison: dict, save_path):
    """
    Kalibrierungskurven für Heim/Remis/Auswärts: vorhergesagte vs. empirische
    Wahrscheinlichkeit. Perfekt = Diagonale.
    """
    from evaluation import calibration_bins

    probs_model = comparison["probs_model"]
    probs_bm = comparison["probs_bookmaker"]
    outcomes = comparison["outcomes"]
    cutoff = comparison["cutoff"]
    n = comparison["n"]

    sl = slice(cutoff, n)
    cb_model = calibration_bins(probs_model[sl], outcomes[sl], n_bins=8)
    cb_bm = calibration_bins(probs_bm[sl], outcomes[sl], n_bins=8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Heimsieg", "Unentschieden", "Auswärtssieg"]

    for k, ax in enumerate(axes):
        # Diagonale
        ax.plot([0, 1], [0, 1], ":", color="black", alpha=0.5,
                label="perfekte Kalibrierung")
        # Modell
        mdl = cb_model[k]
        valid_m = ~np.isnan(mdl["empirical"])
        ax.plot(mdl["mean_pred"][valid_m], mdl["empirical"][valid_m],
                "-o", color=PALETTE[0], markersize=7, label="Modell")
        # Buchmacher
        bm = cb_bm[k]
        valid_b = ~np.isnan(bm["empirical"])
        ax.plot(bm["mean_pred"][valid_b], bm["empirical"][valid_b],
                "-s", color=PALETTE[3], markersize=6, label="Buchmacher")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Vorhergesagte Wahrscheinlichkeit")
        if k == 0:
            ax.set_ylabel("Empirische Häufigkeit")
        ax.set_title(titles[k])
        if k == 2:
            ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Kalibrierung: Vorhersagen vs. empirische Häufigkeit  "
                 "(Punkte auf Diagonale = gut kalibriert)", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
