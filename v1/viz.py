"""
viz.py
======
Statische Visualisierungen für die Rue-Salvesen-Pipeline.

Alle Plots werden in eine Ausgabedatei gespeichert und können in der
Präsentation verwendet werden.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
        (axes[0], "FTHG", "Home goals", PALETTE[0]),
        (axes[1], "FTAG", "Away goals", PALETTE[1]),
    ]:
        vals = df[col].clip(upper=max_g).values
        bins = np.arange(-0.5, max_g + 1.5, 1)
        counts, _ = np.histogram(vals, bins=bins)
        freqs = counts / counts.sum()
        ax.bar(range(max_g + 1), freqs, color=color, edgecolor="white", alpha=0.9)
        ax.set_xlabel("Goals")
        ax.set_title(f"{title} (n={len(df)} matches)")
        mean = df[col].mean()
        ax.axvline(mean, color="black", linestyle="--", alpha=0.6,
                   label=f"mean {mean:.2f}")
        ax.legend()
    axes[0].set_ylabel("Relative frequency")
    fig.suptitle("Distribution of goals in the Bundesliga 2000–2026", y=1.02)
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
                 color=PALETTE[0], label="Home win")
    axes[0].plot(grp["Season"], grp["draw_rate"], "-o",
                 color=PALETTE[2], label="Draw")
    axes[0].plot(grp["Season"], grp["away_win_rate"], "-o",
                 color=PALETTE[1], label="Away win")
    axes[0].set_title("Result shares per season")
    axes[0].set_ylabel("Share")
    axes[0].legend(loc="upper right")
    axes[0].tick_params(axis="x", rotation=70)

    axes[1].plot(grp["Season"], grp["avg_home_goals"], "-o",
                 color=PALETTE[0], label="Home goals (avg)")
    axes[1].plot(grp["Season"], grp["avg_away_goals"], "-o",
                 color=PALETTE[1], label="Away goals (avg)")
    axes[1].fill_between(grp["Season"], grp["avg_home_goals"], grp["avg_away_goals"],
                         alpha=0.15, color="gray", label="Home advantage")
    axes[1].set_title("Average goals per match")
    axes[1].set_ylabel("Goals")
    axes[1].legend(loc="upper right")
    axes[1].tick_params(axis="x", rotation=70)

    fig.suptitle("Home advantage across 26 Bundesliga seasons", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# 2. Modell-Intuition
# ─────────────────────────────────────────────────────────────────────

def plot_brownian_motion_examples(save_path, n_paths: int = 6, seed: int = 42):
    """Beispielpfade einer Brownschen Bewegung."""
    rng = np.random.default_rng(seed)
    n_steps = 200
    days = np.linspace(0, 270, n_steps)  # ~Saison

    sigma2 = 1 / 37
    tau = 100

    fig, ax = plt.subplots(figsize=(10, 5))
    for k in range(n_paths):
        path = np.zeros(n_steps)
        for i in range(1, n_steps):
            dt = days[i] - days[i - 1]
            path[i] = path[i - 1] + rng.normal(0, np.sqrt(dt / tau * sigma2))
        ax.plot(days, path, alpha=0.8, color=PALETTE[k % len(PALETTE)],
                label=f"Team {k+1}")

    ax.axhline(0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("Days since season start")
    ax.set_ylabel("Attack strength a(t)")
    ax.set_title("Example Brownian paths of attack strength\n"
                 f"(τ={tau} days, σ²={sigma2:.4f}) — how strengths 'wander' under the prior")
    ax.legend(ncol=3, loc="lower center")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_tor_modell_intuition(save_path):
    """Zeigt wie λ → Poisson → Dixon-Coles → Mischung das Tormodell formen."""
    from scipy.stats import poisson as scipy_poisson
    from model import dc_correction, EPSILON

    lam_x, lam_y = 1.6, 1.1
    max_g = 6
    x = np.arange(max_g + 1)
    y = np.arange(max_g + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Naives Poisson (Produkt)
    p_naive = np.outer(scipy_poisson.pmf(x, lam_x), scipy_poisson.pmf(y, lam_y))
    # Mit DC-Korrektur
    p_dc = p_naive.copy()
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p_dc[i, j] *= dc_correction(i, j, lam_x, lam_y)
    p_dc /= p_dc.sum()
    # Mischung mit Durchschnittsspiel (hier λ=1.5, 1.1)
    lam_x_avg, lam_y_avg = 1.5, 1.1
    p_avg = np.outer(scipy_poisson.pmf(x, lam_x_avg), scipy_poisson.pmf(y, lam_y_avg))
    p_avg /= p_avg.sum()
    p_mix = (1 - EPSILON) * p_dc + EPSILON * p_avg

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, mat, title in zip(
        axes,
        [p_naive / p_naive.sum(), p_dc, p_mix],
        [f"Naive Poisson\n(λ_h={lam_x}, λ_a={lam_y})",
         "+ Dixon-Coles correction\n(0:0 and 1:1 boosted)",
         f"+ Mixture with average match\n(ε={EPSILON})"],
    ):
        im = ax.imshow(mat.T, origin="lower", cmap="viridis", aspect="equal")
        ax.set_xlabel("Home goals")
        ax.set_ylabel("Away goals")
        ax.set_title(title)
        ax.set_xticks(range(max_g + 1))
        ax.set_yticks(range(max_g + 1))
        # Werte einzeichnen
        for i in range(max_g + 1):
            for j in range(max_g + 1):
                ax.text(i, j, f"{mat[i,j]*100:.0f}",
                        ha="center", va="center",
                        color="white" if mat[i, j] < mat.max() * 0.5 else "black",
                        fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle("Goal model: step by step — values in % (rounded)", y=1.02)
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

    # Team-Knoten
    for tau in times:
        for t in teams:
            ax.scatter(x_pos[t], y_pos[tau], s=900, c=PALETTE[teams.index(t)],
                       edgecolors="black", linewidth=1.2, zorder=3)
            ax.text(x_pos[t], y_pos[tau], t, ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white", zorder=4)
        ax.text(-1.5, y_pos[tau], f"t={tau}", ha="right", va="center",
                fontsize=11, fontstyle="italic")

    # Zeit-Pfeile (Brownsche Bewegung)
    for tau in times[:-1]:
        for t in teams:
            ax.annotate("", xy=(x_pos[t], y_pos[tau + 1] + 0.4),
                        xytext=(x_pos[t], y_pos[tau] - 0.4),
                        arrowprops=dict(arrowstyle="->", color="gray",
                                        alpha=0.6, lw=1.5))

    # Matches (X-Y Knoten zwischen Teams)
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
        # Pfeile von beiden Teams zum Match
        for t in (t1, t2):
            ax.annotate("", xy=(mx, my + 0.3),
                        xytext=(x_pos[t], y_pos[tau] - 0.4),
                        arrowprops=dict(arrowstyle="->", color=PALETTE[teams.index(t)],
                                        alpha=0.7, lw=1.2))

    ax.set_xlim(-3, 8)
    ax.set_ylim(-8, 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Model structure as a directed acyclic graph (DAG)\n"
                 "Arrows: parent-child relationships — everything is linked, "
                 "so joint inference is required",
                 fontsize=12)

    # Legende
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=14, label="Team strengths (a, d)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#f0f0f0",
               markeredgecolor="black", markersize=14, label="observed match"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#ffe4b5",
               markeredgecolor="black", markersize=14, label="match to predict"),
    ]
    ax.legend(handles=legend_elems, loc="upper right")

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# 3. MCMC-Diagnostik
# ─────────────────────────────────────────────────────────────────────

def plot_mcmc_trace(samples, L, save_path, n_teams_show: int = 4):
    """Trace-Plots der mittleren Stärke — jede Kette als eigene Linie."""
    chains = samples.get("chains") or []
    has_chains = len(chains) > 1

    if has_chains:
        team_ids = list(chains[0]["attack"][0].keys())
        n_per_chain = len(chains[0]["attack"])
        rhat_str = (f"  |  R-hat max = {samples['rhat_max']:.3f}"
                    if "rhat_max" in samples else "")
        header = f"{len(chains)} chains × {n_per_chain} samples{rhat_str}"
    else:
        team_ids = list(samples["attack"][0].keys())
        n_per_chain = len(samples["attack"])
        header = f"{n_per_chain} samples (single chain)"

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

    for ax, ylabel in zip(axes, ["Mean attack strength", "Mean defense strength"]):
        ax.axhline(0, color="black", linestyle=":", alpha=0.4)
        ax.set_ylabel(ylabel)

    axes[0].legend(ncol=4, loc="upper right", fontsize=9)
    axes[1].set_xlabel("MCMC iteration (per chain)")

    fig.suptitle(
        f"MCMC convergence ({header})\n"
        "Same color = same team, each line = one chain"
        " — converged when the chains overlap",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_strength_evolution(samples, L, save_path, top_k: int = 6):
    """Zeitlicher Verlauf der Stärken für die stärksten Teams (analog Fig. 6)."""
    from mcmc import posterior_means
    mean_attack, mean_defense, _ = posterior_means(samples)

    # Stärkstes Team = höchste mittlere (attack - defense_against)
    overall = []
    for team in range(len(L.teams)):
        overall.append((mean_attack[team].mean() + mean_defense[team].mean(), team))
    overall.sort(reverse=True)
    top_teams = [t for _, t in overall[:top_k]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    for k, team in enumerate(top_teams):
        match_indices = L.team_matches[team]
        days = L.match_days[match_indices]
        dates = L.dates[match_indices]
        color = PALETTE[k % len(PALETTE)]
        axes[0].plot(dates, mean_attack[team], "-o", color=color,
                     markersize=4, label=L.teams[team], alpha=0.9)
        axes[1].plot(dates, mean_defense[team], "-o", color=color,
                     markersize=4, label=L.teams[team], alpha=0.9)

    axes[0].set_title("Posterior attack strength over the season")
    axes[0].set_ylabel("a(t)")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].set_title("Posterior defense strength over the season")
    axes[1].set_ylabel("d(t)")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle(f"Strength evolution: top {top_k} teams", y=1.02)
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
    rank_counts = np.zeros((n_teams, n_teams), dtype=int)  # team, rank
    n_samples = len(samples["attack"])

    # Tatsächliche Tabelle
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

    # Replay-Simulation aus Posterior
    for _ in range(n_replays):
        s_idx = rng.integers(0, n_samples)
        pts = np.zeros(n_teams)
        gd = np.zeros(n_teams)
        for m in range(len(L.home_goals)):
            h, a = L.home_idx[m], L.away_idx[m]
            lh = L.local_idx[(h, m)]
            la = L.local_idx[(a, m)]
            a_h = samples["attack"][s_idx][h][lh]
            d_h = samples["defense"][s_idx][h][lh]
            a_a = samples["attack"][s_idx][a][la]
            d_a = samples["defense"][s_idx][a][la]
            p_home, p_draw, p_away = predict_outcome_probs(
                a_h, d_h, a_a, d_a, L.c_x, L.c_y,
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

    # Plot
    fig, ax = plt.subplots(figsize=(13, 6.5))
    order = np.argsort(actual_rank)
    expected_rank = (rank_counts * np.arange(n_teams)[None, :]).sum(1) / n_replays

    for k, team in enumerate(order):
        # 90% Intervall
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
    ax.set_ylabel("Table position")
    ax.invert_yaxis()
    ax.set_title("Retrospective rank distribution\n"
                 "blue = posterior expectation, red = actual rank, "
                 "gray = 90% interval")

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[0],
               markersize=10, label="Expected rank (posterior)"),
        Line2D([0], [0], marker="x", color="red", markersize=10,
               markeredgewidth=2, linestyle="", label="Actual rank"),
        Line2D([0], [0], color="gray", linewidth=2, label="90% interval"),
    ]
    ax.legend(handles=legend, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_surprise_matches(samples, L, df_season, save_path, top_n: int = 10):
    """Top-N überraschendste Spiele (höchste Posterior-Wkt. für δ=1)."""
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
    ax.set_xlabel("P(outlier match)")
    ax.set_xlim(0, 1)
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(r["p_outlier"] + 0.01, y_pos[i], f"{r['p_outlier']:.2f}",
                va="center", fontsize=9)

    ax.set_title(f"The {top_n} most surprising matches of this season\n"
                 "(highest posterior probability for the mixture component δ=1)")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_team_ranking_summary(samples, L, save_path):
    """Mittlere Angriffs- vs. Abwehrstärke aller Teams (Scatter)."""
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
    ax.set_xlabel("Mean attack strength (over season)")
    ax.set_ylabel("Mean defense strength (over season)")
    ax.set_title("Team profiles: attack vs. defense\n"
                 "(top right = strong in both, bottom left = weak)")
    fig.colorbar(sc, ax=ax, label="Overall strength")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
