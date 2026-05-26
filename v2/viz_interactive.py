"""
viz_interactive.py
==================
Interaktive Plotly-HTML-Versionen der wichtigsten Diagnose-Plots.

Ergänzen (nicht ersetzen) die statischen PNGs:
  07_staerken_evolution.png   →  07_staerken_evolution.html
  06_mcmc_trace.png           →  06_mcmc_trace.html

Bedienung im Browser:
  * Klick auf Legenden-Eintrag      → Trace ein-/ausblenden
  * Doppelklick auf Legenden-Eintrag → Nur diesen Trace zeigen
  * Klick auf Gruppen-Überschrift    → Ganze Gruppe togglen
  * Buttons oben: Schnellauswahl (Top-6 / Alle / Keine)
  * Maus über Linie:  Tooltip mit Datum, Team, Wert
  * Drag im Plot:     Zoom
  * Doppelklick im Plot: Reset
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Farbpalette identisch zu viz.py (matplotlib tab10)
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
    "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
    "#dbdb8d", "#9edae5",
]


def _team_colors(teams: list[str]) -> dict:
    return {t: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(teams)}


# ─────────────────────────────────────────────────────────────────────
# Stärken-Evolution (interaktiv)
# ─────────────────────────────────────────────────────────────────────

def plot_strength_evolution_html(samples, L, save_path,
                                  show_top_k_initially: int = 6):
    """
    Interaktiver Plot der Posterior-Stärken-Verläufe für ALLE Teams.

    Zwei Subplots (oben: Angriff, unten: Abwehr). Jedes Team hat
    eine Legendengruppe → ein Klick togglet beide Linien gemeinsam.
    Initial sichtbar: die ``show_top_k_initially`` stärksten Teams,
    der Rest ist eingeklappt (über die Legende zuschaltbar).
    """
    from mcmc import posterior_means
    mean_attack, mean_defense, _ = posterior_means(samples)

    n_teams = L.n_teams
    overall = sorted(range(n_teams),
                     key=lambda t: -(mean_attack[t].mean() + mean_defense[t].mean()))
    top_set = set(overall[:show_top_k_initially])
    colors = _team_colors(L.teams)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Angriffsstärke a(t)", "Abwehrstärke d(t)"),
        vertical_spacing=0.10,
    )

    for rank, team in enumerate(overall):
        name = L.teams[team]
        match_idx = L.team_matches[team]
        dates = pd.to_datetime(L.dates[match_idx])
        x = list(dates)
        a_vals = mean_attack[team]
        d_vals = mean_defense[team]
        visible = True if team in top_set else "legendonly"
        col = colors[name]

        # Angriff
        fig.add_trace(
            go.Scatter(
                x=x, y=a_vals,
                mode="lines+markers",
                name=name,
                legendgroup=name,
                legendgrouptitle=None,
                line=dict(color=col, width=2),
                marker=dict(size=5),
                visible=visible,
                hovertemplate=(f"<b>{name}</b><br>"
                               "Datum: %{x|%d.%m.%Y}<br>"
                               "a = %{y:.3f}<extra></extra>"),
            ),
            row=1, col=1,
        )
        # Abwehr — gleiche Legendengruppe, kein separater Legendeneintrag
        fig.add_trace(
            go.Scatter(
                x=x, y=d_vals,
                mode="lines+markers",
                name=name,
                legendgroup=name,
                showlegend=False,
                line=dict(color=col, width=2, dash="dot"),
                marker=dict(size=5, symbol="diamond"),
                visible=visible,
                hovertemplate=(f"<b>{name}</b><br>"
                               "Datum: %{x|%d.%m.%Y}<br>"
                               "d = %{y:.3f}<extra></extra>"),
            ),
            row=2, col=1,
        )

    # Null-Linie pro Subplot
    for r in (1, 2):
        fig.add_hline(y=0, line_dash="dot", line_color="gray",
                      opacity=0.4, row=r, col=1)

    # Quick-Select-Buttons
    n_traces = 2 * n_teams
    all_visible = [True] * n_traces
    none_visible = [True] + ["legendonly"] * (n_traces - 1)  # mindestens 1 an
    top_visible = []
    for rank, team in enumerate(overall):
        v = True if team in top_set else "legendonly"
        top_visible.extend([v, v])

    fig.update_layout(
        title=dict(text=("Posterior-Stärken-Evolution  —  "
                         "interaktiv (Legende: Klick = togglen, Doppelklick = isolieren)"),
                   x=0.02),
        height=820,
        hovermode="x unified",
        legend=dict(
            title="Teams (Klick zum Toggle)",
            groupclick="togglegroup",
            tracegroupgap=2,
            itemsizing="constant",
        ),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.5, y=1.10, xanchor="center",
            buttons=[
                dict(label=f"Top {show_top_k_initially}",
                     method="update",
                     args=[{"visible": top_visible}]),
                dict(label="Alle Teams",
                     method="update",
                     args=[{"visible": all_visible}]),
                dict(label="Keine",
                     method="update",
                     args=[{"visible": none_visible}]),
            ],
        )],
        margin=dict(l=60, r=30, t=120, b=60),
    )
    fig.update_xaxes(title_text="Datum", row=2, col=1)
    fig.update_yaxes(title_text="a(t)", row=1, col=1)
    fig.update_yaxes(title_text="d(t)", row=2, col=1)

    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    fig.write_html(save_path, include_plotlyjs="cdn", full_html=True)


# ─────────────────────────────────────────────────────────────────────
# MCMC-Trace (interaktiv, getrennt nach Team & Kette)
# ─────────────────────────────────────────────────────────────────────

def plot_mcmc_trace_html(samples, L, save_path,
                          n_teams_show: int = 6,
                          max_points_per_trace: int = 800):
    """
    Interaktiver Trace-Plot. Pro (Team, Kette) ein eigener Trace.

    Legendenstruktur:
      Gruppe = Team   →   Header-Klick togglet ALLE Ketten eines Teams
      Eintrag = einzelne Kette   →   Klick togglet nur diese Kette

    Initial nur die ``n_teams_show`` stärksten Teams sichtbar.

    Bei langen Ketten (z.B. 18 000 Samples pro Kette) wird auf
    ``max_points_per_trace`` Stützstellen ausgedünnt — das reicht für die
    visuelle Konvergenzdiagnose und hält die HTML-Datei unter ~5 MB.
    """
    from mcmc import posterior_means
    chains = samples.get("chains") or []
    has_chains = len(chains) > 1

    def _thin(values: list[float], max_n: int) -> tuple[list[int], list[float]]:
        n = len(values)
        if n <= max_n:
            return list(range(n)), values
        step = n / max_n
        idx = [int(i * step) for i in range(max_n)]
        return idx, [values[i] for i in idx]

    # Ranking nach Posterior-Mittel (über alle kombinierten Samples)
    mean_attack, mean_defense, _ = posterior_means(samples)
    overall = sorted(range(L.n_teams),
                     key=lambda t: -(mean_attack[t].mean() + mean_defense[t].mean()))
    show_set = set(overall[:n_teams_show])
    colors = _team_colors(L.teams)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Mittlere Angriffsstärke (über Spiele) — pro Sample",
                        "Mittlere Abwehrstärke (über Spiele) — pro Sample"),
        vertical_spacing=0.10,
    )

    # Falls multi-chain: jede Kette als eigener Trace
    if has_chains:
        n_per_chain = len(chains[0]["attack"])
        for team in overall:
            name = L.teams[team]
            col = colors[name]
            initial_vis = True if team in show_set else "legendonly"

            for ci, chain in enumerate(chains):
                a_mean = [float(s[team].mean()) for s in chain["attack"]]
                d_mean = [float(s[team].mean()) for s in chain["defense"]]
                x_iter, a_mean = _thin(a_mean, max_points_per_trace)
                _, d_mean = _thin(d_mean, max_points_per_trace)

                fig.add_trace(
                    go.Scatter(
                        x=x_iter, y=a_mean,
                        mode="lines",
                        name=f"{name} · Kette {ci + 1}",
                        legendgroup=name,
                        legendgrouptitle=dict(text=name) if ci == 0 else None,
                        line=dict(color=col, width=1.2),
                        opacity=0.75,
                        visible=initial_vis,
                        hovertemplate=(f"<b>{name}</b> — Kette {ci + 1}<br>"
                                       "Iter: %{x}<br>"
                                       "ā = %{y:.3f}<extra></extra>"),
                    ),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_iter, y=d_mean,
                        mode="lines",
                        name=f"{name} · Kette {ci + 1} (d)",
                        legendgroup=name,
                        showlegend=False,
                        line=dict(color=col, width=1.2, dash="dot"),
                        opacity=0.75,
                        visible=initial_vis,
                        hovertemplate=(f"<b>{name}</b> — Kette {ci + 1}<br>"
                                       "Iter: %{x}<br>"
                                       "d̄ = %{y:.3f}<extra></extra>"),
                    ),
                    row=2, col=1,
                )
    else:
        # Single-chain Fallback
        n_per_chain = len(samples["attack"])
        for team in overall:
            name = L.teams[team]
            col = colors[name]
            a_mean = [float(s[team].mean()) for s in samples["attack"]]
            d_mean = [float(s[team].mean()) for s in samples["defense"]]
            x_iter, a_mean = _thin(a_mean, max_points_per_trace)
            _, d_mean = _thin(d_mean, max_points_per_trace)
            initial_vis = True if team in show_set else "legendonly"

            fig.add_trace(
                go.Scatter(
                    x=x_iter, y=a_mean,
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    line=dict(color=col, width=1.5),
                    visible=initial_vis,
                    hovertemplate=(f"<b>{name}</b><br>Iter: %{{x}}<br>"
                                   "ā = %{y:.3f}<extra></extra>"),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_iter, y=d_mean,
                    mode="lines",
                    name=name + " (d)",
                    legendgroup=name,
                    showlegend=False,
                    line=dict(color=col, width=1.5, dash="dot"),
                    visible=initial_vis,
                    hovertemplate=(f"<b>{name}</b><br>Iter: %{{x}}<br>"
                                   "d̄ = %{y:.3f}<extra></extra>"),
                ),
                row=2, col=1,
            )

    for r in (1, 2):
        fig.add_hline(y=0, line_dash="dot", line_color="gray",
                      opacity=0.4, row=r, col=1)

    # Quick-Buttons: alle / top-N / keine
    n_traces = len(fig.data)
    all_visible = [True] * n_traces
    none_visible = [True] + ["legendonly"] * (n_traces - 1)

    # top-N: pro Trace prüfen, ob er zu einem top-N-Team gehört
    top_visible = []
    for tr in fig.data:
        team_name = tr.legendgroup
        team_idx = L.teams.index(team_name) if team_name in L.teams else -1
        v = True if team_idx in show_set else "legendonly"
        top_visible.append(v)

    rhat_str = (f"R-hat max = {samples['rhat_max']:.3f}"
                if "rhat_max" in samples else "")
    sub = f"{len(chains)} Ketten à {n_per_chain} Samples  |  {rhat_str}" \
        if has_chains else f"{n_per_chain} Samples (Einzelkette)"

    fig.update_layout(
        title=dict(text=f"MCMC-Trace (interaktiv)  —  {sub}<br>"
                        "<span style='font-size:12px;color:#666'>"
                        "Linke Legendenspalte: Team-Klick togglet alle Ketten. "
                        "Innerhalb einer Gruppe togglet einzelne Ketten.</span>",
                   x=0.02),
        height=820,
        hovermode="closest",
        legend=dict(
            title="Teams · Ketten",
            groupclick="togglegroup",
            tracegroupgap=8,
            itemsizing="constant",
        ),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.5, y=1.10, xanchor="center",
            buttons=[
                dict(label=f"Top {n_teams_show}",
                     method="update",
                     args=[{"visible": top_visible}]),
                dict(label="Alle Teams & Ketten",
                     method="update",
                     args=[{"visible": all_visible}]),
                dict(label="Keine",
                     method="update",
                     args=[{"visible": none_visible}]),
            ],
        )],
        margin=dict(l=60, r=30, t=140, b=60),
    )
    fig.update_xaxes(title_text="MCMC-Iteration (pro Kette)", row=2, col=1)
    fig.update_yaxes(title_text="ā", row=1, col=1)
    fig.update_yaxes(title_text="d̄", row=2, col=1)

    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    fig.write_html(save_path, include_plotlyjs="cdn", full_html=True)


# ─────────────────────────────────────────────────────────────────────
# Bonus: Team-Profile als interaktiver Scatter (alle Teams hover-fähig)
# ─────────────────────────────────────────────────────────────────────

def plot_team_profile_html(samples, L, save_path):
    """
    Interaktiver Angriff-vs-Abwehr-Scatter. Hover zeigt Team + Rang.
    """
    from mcmc import posterior_means
    mean_attack, mean_defense, _ = posterior_means(samples)

    a_avg = np.array([mean_attack[t].mean() for t in range(L.n_teams)])
    d_avg = np.array([mean_defense[t].mean() for t in range(L.n_teams)])
    overall = a_avg + d_avg

    df = pd.DataFrame({
        "team": L.teams, "a": a_avg, "d": d_avg, "overall": overall,
    }).sort_values("overall", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    fig = go.Figure(go.Scatter(
        x=df["a"], y=df["d"], mode="markers+text",
        text=df["team"], textposition="bottom center",
        marker=dict(
            size=18, color=df["overall"], colorscale="RdYlGn",
            showscale=True, colorbar=dict(title="Gesamtstärke"),
            line=dict(color="black", width=0.7),
        ),
        hovertemplate=("<b>%{text}</b><br>"
                       "Angriff a = %{x:.3f}<br>"
                       "Abwehr d = %{y:.3f}<br>"
                       "Gesamtstärke = %{marker.color:.3f}<br>"
                       "Posterior-Rang = %{customdata}<extra></extra>"),
        customdata=df["rank"],
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig.update_layout(
        title="Team-Profile: Angriff vs. Abwehr (interaktiv)",
        xaxis_title="Mittlere Angriffsstärke", yaxis_title="Mittlere Abwehrstärke",
        height=700,
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    fig.write_html(save_path, include_plotlyjs="cdn", full_html=True)
