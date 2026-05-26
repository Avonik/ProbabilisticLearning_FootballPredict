"""
animations.py
=============
Animationen für die Präsentation. Werden als .gif gespeichert.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#17becf"]


def animate_strength_evolution(samples, L, save_path, top_k: int = 6,
                               fps: int = 8):
    """Animiert wie sich Posterior-Stärken über die Saison entwickeln."""
    from mcmc import posterior_means
    mean_attack, mean_defense, _ = posterior_means(samples)

    overall = sorted(range(len(L.teams)),
                     key=lambda t: -(mean_attack[t].mean() + mean_defense[t].mean()))
    top_teams = overall[:top_k]

    # Einheitliche Zeitachse: alle eindeutigen Matchdays
    all_days = np.array(sorted(set(L.match_days)))

    # Interpoliere Stärken auf gemeinsame Zeitachse
    def interpolate(team_id, attr):
        match_indices = L.team_matches[team_id]
        days = L.match_days[match_indices]
        vals = attr[team_id]
        return np.interp(all_days, days, vals)

    attack_interp = {t: interpolate(t, mean_attack) for t in top_teams}
    defense_interp = {t: interpolate(t, mean_defense) for t in top_teams}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    axes[0].set_title("Angriffsstärke a(t)")
    axes[1].set_title("Abwehrstärke d(t)")
    for ax in axes:
        ax.axhline(0, color="black", linestyle=":", alpha=0.4)
        ax.set_xlabel("Tage seit Saisonbeginn")
        ax.grid(alpha=0.3)
        ax.set_xlim(all_days.min(), all_days.max())

    # Y-Achsen-Limits
    a_min = min(min(attack_interp[t]) for t in top_teams) - 0.05
    a_max = max(max(attack_interp[t]) for t in top_teams) + 0.05
    d_min = min(min(defense_interp[t]) for t in top_teams) - 0.05
    d_max = max(max(defense_interp[t]) for t in top_teams) + 0.05
    axes[0].set_ylim(a_min, a_max)
    axes[1].set_ylim(d_min, d_max)

    lines_a, lines_d, dots_a, dots_d = {}, {}, {}, {}
    for k, t in enumerate(top_teams):
        c = PALETTE[k % len(PALETTE)]
        ln_a, = axes[0].plot([], [], "-", color=c, lw=2, label=L.teams[t])
        ln_d, = axes[1].plot([], [], "-", color=c, lw=2)
        pt_a, = axes[0].plot([], [], "o", color=c, markersize=10,
                             markeredgecolor="black")
        pt_d, = axes[1].plot([], [], "o", color=c, markersize=10,
                             markeredgecolor="black")
        lines_a[t] = ln_a
        lines_d[t] = ln_d
        dots_a[t] = pt_a
        dots_d[t] = pt_d

    axes[0].legend(loc="upper left", ncol=2, fontsize=9)
    fig.suptitle("Animierte Stärken-Evolution", fontsize=13)
    fig.tight_layout()

    n_frames = min(80, len(all_days))
    frame_indices = np.linspace(0, len(all_days) - 1, n_frames).astype(int)

    def update(fi):
        i = frame_indices[fi]
        for t in top_teams:
            lines_a[t].set_data(all_days[: i + 1], attack_interp[t][: i + 1])
            lines_d[t].set_data(all_days[: i + 1], defense_interp[t][: i + 1])
            dots_a[t].set_data([all_days[i]], [attack_interp[t][i]])
            dots_d[t].set_data([all_days[i]], [defense_interp[t][i]])
        return list(lines_a.values()) + list(lines_d.values()) + \
               list(dots_a.values()) + list(dots_d.values())

    anim = FuncAnimation(fig, update, frames=n_frames, blit=True, interval=120)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def animate_season_table(L, save_path, fps: int = 6):
    """Animiert die echte Tabelle über die Spieltage."""
    n_teams = len(L.teams)
    teams = L.teams

    # Spielreihenfolge (chronologisch)
    order = np.argsort(L.match_days)
    n_total = len(order)
    n_frames = min(60, n_total)
    checkpoints = np.linspace(1, n_total, n_frames).astype(int)

    fig, ax = plt.subplots(figsize=(8, 9))
    bars = ax.barh(range(n_teams), [0] * n_teams,
                   color=[PALETTE[i % len(PALETTE)] for i in range(n_teams)])
    ax.set_yticks(range(n_teams))
    ax.set_xlabel("Punkte")
    ax.set_xlim(0, 100)
    title = ax.set_title("Tabellenstand")
    ax.invert_yaxis()
    text_labels = [ax.text(0, i, "", va="center", fontsize=10) for i in range(n_teams)]

    def update(fi):
        cutoff = checkpoints[fi]
        pts = np.zeros(n_teams)
        gd = np.zeros(n_teams)
        for k in range(cutoff):
            m = order[k]
            h, a = L.home_idx[m], L.away_idx[m]
            hg, ag = L.home_goals[m], L.away_goals[m]
            if hg > ag:
                pts[h] += 3
            elif hg < ag:
                pts[a] += 3
            else:
                pts[h] += 1
                pts[a] += 1
            gd[h] += hg - ag
            gd[a] += ag - hg
        rank = np.argsort(-(pts + 0.001 * gd))
        for r, t in enumerate(rank):
            bars[r].set_width(pts[t])
            bars[r].set_color(PALETTE[t % len(PALETTE)])
            text_labels[r].set_position((pts[t] + 0.5, r))
            text_labels[r].set_text(f"  {teams[t]} ({int(pts[t])})")
        title.set_text(f"Tabellenstand nach {cutoff} von {n_total} Spielen")
        return list(bars) + text_labels + [title]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def animate_mcmc_walk(samples, L, save_path, team_idx: int = 0,
                      n_frames: int = 60, fps: int = 8):
    """Animiert die MCMC-Trace eines einzelnen Teams."""
    team_attack = [s[team_idx] for s in samples["attack"]]
    team_defense = [s[team_idx] for s in samples["defense"]]
    n_match = len(team_attack[0])
    n_samples = len(team_attack)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].set_title(f"MCMC-Verlauf für {L.teams[team_idx]}: Angriffsstärke")
    axes[1].set_title("Abwehrstärke")
    axes[0].set_ylabel("a")
    axes[1].set_ylabel("d")
    axes[1].set_xlabel("Spielindex (chronologisch)")

    # Ranges
    all_a = np.concatenate(team_attack)
    all_d = np.concatenate(team_defense)
    axes[0].set_ylim(all_a.min() - 0.1, all_a.max() + 0.1)
    axes[1].set_ylim(all_d.min() - 0.1, all_d.max() + 0.1)
    axes[0].set_xlim(0, n_match - 1)

    # Plotelemente
    history_lines_a = []
    history_lines_d = []
    current_a, = axes[0].plot(range(n_match), team_attack[0], "-o",
                              color=PALETTE[0], lw=2.5, markersize=6, zorder=5)
    current_d, = axes[1].plot(range(n_match), team_defense[0], "-o",
                              color=PALETTE[3], lw=2.5, markersize=6, zorder=5)
    axes[0].axhline(0, color="black", linestyle=":", alpha=0.4)
    axes[1].axhline(0, color="black", linestyle=":", alpha=0.4)

    frame_indices = np.linspace(0, n_samples - 1, n_frames).astype(int)

    def update(fi):
        i = frame_indices[fi]
        # Vergangene Samples halbtransparent
        if fi % 3 == 0 and fi > 0:
            ln_a, = axes[0].plot(range(n_match), team_attack[i],
                                 color=PALETTE[0], alpha=0.15, lw=0.6)
            ln_d, = axes[1].plot(range(n_match), team_defense[i],
                                 color=PALETTE[3], alpha=0.15, lw=0.6)
            history_lines_a.append(ln_a)
            history_lines_d.append(ln_d)
        current_a.set_ydata(team_attack[i])
        current_d.set_ydata(team_defense[i])
        fig.suptitle(f"MCMC-Sample {i + 1} / {n_samples}", fontsize=11)
        return [current_a, current_d] + history_lines_a + history_lines_d

    anim = FuncAnimation(fig, update, frames=n_frames, interval=180, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
