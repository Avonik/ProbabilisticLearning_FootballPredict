"""
backtest_multiseason.py
=======================
MULTI-SEASON-SIGNIFIKANZSTUDIE — schlägt das Modell die Buchmacher-Schlusslinie
über mehrere Saisons *signifikant*, oder ist „4/5 besser" nur Rauschen?

Eine einzelne Saison (~94 Holdout-Spiele, SE des RPS ≈ 0.02) kann „auf Augenhöhe"
nicht von „kleiner echter Vorsprung" trennen. Dieser Driver:

  1. schleift den ehrlichen Walk-Forward (``backtest.evaluate_season_walkforward``)
     über alle Saisons ≥ 2014/15 (echtes xG verfügbar),
  2. **poolt die Per-Spiel-RPS-Differenzen** (Buchmacher − Modell) über alle
     Saisons,
  3. rechnet einen **gepaarten Wilcoxon-Signed-Rank-Test** + **Bootstrap-95%-CI**
     des mittleren ΔRPS — die eigentliche Signifikanzaussage,
  4. **protokolliert pro Saison** ``bm_source`` (welche Quote!) und ``xg_source``
     (echt/Proxy) → eingebauter Leck-/Fairness-Audit,
  5. optional: **Placebo / Negativ-Kontrolle** (``SCRAMBLE_XG``) — xG wird
     innerhalb jeder Saison verwürfelt. Verschwindet der „Vorsprung", sitzt das
     Signal echt im xG; bleibt er, gibt es einen Pipeline-Leak.

Modell-Einstellungen (Gamma/echtes-xG/φ/Hyperparameter/Ketten/Budget) werden
1:1 aus ``backtest.py`` übernommen — eine einzige Quelle der Wahrheit. Für die
saubere Studie dort ``USE_TUNED_HYPERPARAMS = False`` setzen (leckfrei).

Aufruf:  python backtest_multiseason.py
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from data import load_bundesliga, extract_bookmaker_probs, sort_matches
from xg import fit_xg_weights, add_xg_columns
from real_xg import add_real_xg_columns, UNDERSTAT_FIRST_YEAR
from mcmc import warmup_jit
from evaluation import evaluate_predictions, rps_one
import backtest as bt   # Konfiguration + evaluate_season_walkforward (Quelle der Wahrheit)


# ─────────────────────────────────────────────────────────────────────
# Konfiguration (modellspezifisches kommt aus backtest.py)
# ─────────────────────────────────────────────────────────────────────
SEASONS: list[str] | None = ["2016/17","2017/18","2018/19","2019/20","2020/21","2021/22","2022/23","2023/24","2024/25","2025/26"]
#["2022/23", "2023/24", "2024/25"]
    #["2016/17","2017/18","2018/19","2019/20","2020/21","2021/22","2022/23","2023/24","2024/25","2025/26"]   # None = auto: alle Saisons ≥ 2014/15 im DataFrame.
                                   # Zum Kürzen z.B. ["2022/23", "2023/24", "2024/25"].
SCRAMBLE_XG   = False              # Placebo / Negativ-Kontrolle (xG je Saison verwürfeln)
SCRAMBLE_SEED = 123
BOOTSTRAP_N   = 10_000
BOOTSTRAP_SEED = 0
FIRST_SEASON_LABEL = f"{UNDERSTAT_FIRST_YEAR}/{(UNDERSTAT_FIRST_YEAR + 1) % 100:02d}"  # "2014/15"
OUTPUT_ROOT = Path("output")
MAX_PARALLEL_SEASONS = 2
PAPER_BASELINE_CACHE_VERSION = 2
OUTCOME_LABELS = np.array(["home_win", "draw", "away_win"], dtype=object)


# ─────────────────────────────────────────────────────────────────────
# Hilfen
# ─────────────────────────────────────────────────────────────────────
def _auto_seasons(df: pd.DataFrame) -> list[str]:
    """Alle (vollständigen) Saisons mit echtem xG, d.h. ≥ 2014/15."""
    out = []
    for s in sorted(df["Season"].unique()):
        if s >= FIRST_SEASON_LABEL and (df["Season"] == s).sum() >= 50:
            out.append(s)
    return out


def _scramble_xg(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Placebo: permutiert die (xG_home, xG_away)-PAARE innerhalb jeder Saison.
    Bricht die Verbindung Team-Leistung↔xG, behält aber die xG-Verteilung."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    for _, idx in out.groupby("Season").groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        perm = rng.permutation(len(idx))
        out.loc[idx, "xG_home"] = out.loc[idx, "xG_home"].to_numpy()[perm]
        out.loc[idx, "xG_away"] = out.loc[idx, "xG_away"].to_numpy()[perm]
    return out


def _common_holdout_indices(cmp: dict, *, require_paper: bool) -> np.ndarray:
    """Fair holdout rows where all required probability vectors are finite."""
    pm = cmp["probs_model"]
    pb = cmp["probs_bookmaker"]
    n, cutoff = cmp["n"], cmp["cutoff"]
    hold = np.arange(n) >= cutoff
    fin = np.all(np.isfinite(pm), axis=1) & np.all(np.isfinite(pb), axis=1)
    if require_paper:
        pp = cmp["probs_paper"]
        fin = fin & np.all(np.isfinite(pp), axis=1)
    return np.where(hold & fin)[0]


def _per_match_rps(cmp: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-Spiel-RPS (Modell, Buchmacher) auf dem fairen gemeinsamen Holdout
    (genau die Spiele, auf denen ``evaluate_season_walkforward`` aggregiert)."""
    pm = cmp["probs_model"]
    pb = cmp["probs_bookmaker"]
    out = cmp["outcomes"]
    idx = _common_holdout_indices(cmp, require_paper=False)
    rps_m = np.array([rps_one(pm[i], out[i]) for i in idx])
    rps_b = np.array([rps_one(pb[i], out[i]) for i in idx])
    return idx, rps_m, rps_b


def _per_match_rps3(cmp: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-Spiel-RPS (Modell-v2, Paper-Baseline, Buchmacher) auf dem GEMEINSAMEN
    Holdout (alle drei endlich). Setzt voraus, dass ``cmp['probs_paper']`` da ist
    (via ``backtest.add_paper_baseline``)."""
    pm, pp, pb = cmp["probs_model"], cmp["probs_paper"], cmp["probs_bookmaker"]
    out = cmp["outcomes"]
    idx = _common_holdout_indices(cmp, require_paper=True)
    rps_m = np.array([rps_one(pm[i], out[i]) for i in idx])
    rps_p = np.array([rps_one(pp[i], out[i]) for i in idx])
    rps_b = np.array([rps_one(pb[i], out[i]) for i in idx])
    return idx, rps_m, rps_p, rps_b


def _paired_stats(rps_model: np.ndarray, rps_book: np.ndarray,
                  n_boot: int, seed: int) -> dict:
    """Gepaarte Statistik auf den Per-Spiel-RPS. diff = Buchmacher − Modell
    (positiv = Modell besser)."""
    diff = rps_book - rps_model
    n = len(diff)
    mean_diff = float(diff.mean())
    rel = mean_diff / float(rps_book.mean()) * 100.0

    # Wilcoxon-Signed-Rank (gepaart, zweiseitig)
    try:
        w_stat, w_p = wilcoxon(rps_model, rps_book, alternative="two-sided")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    # Bootstrap-CI des mittleren ΔRPS (gepaart: Spiele resampeln)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = diff[idx].mean()
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    return {
        "n": n, "mean_diff": mean_diff, "rel_pct": rel,
        "wilcoxon_p": float(w_p), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "frac_model_better": float((diff > 0).mean()),
    }


def _season_tag(season: str) -> str:
    return season.replace("/", "_")


def _season_data_fingerprint(df: pd.DataFrame, season: str) -> str:
    cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    sub = sort_matches(df[df["Season"] == season])
    payload_df = sub[cols].copy()
    payload_df["Date"] = pd.to_datetime(payload_df["Date"]).dt.strftime("%Y-%m-%d")
    payload = payload_df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _paper_cache_config(df: pd.DataFrame, season: str, *,
                        holdout_frac: float, n_chains: int,
                        jitter_sd: float) -> dict:
    return {
        "cache_version": PAPER_BASELINE_CACHE_VERSION,
        "model": "paper_baseline",
        "season": season,
        "data_sha256": _season_data_fingerprint(df, season),
        "holdout_frac": float(holdout_frac),
        "n_chains": int(n_chains),
        "jitter_sd": float(jitter_sd),
        "base_iter": int(bt.WF_BASE_ITER),
        "base_burnin": int(bt.WF_BASE_BURNIN),
        "warm_iter": int(bt.WF_WARM_ITER),
        "warm_burnin": int(bt.WF_WARM_BURNIN),
        "thin": int(bt.WF_THIN),
        "proposal_sd": float(bt.PROPOSAL_SD),
        "seed": int(bt.SEED),
        "tau": float(bt.DEFAULT_TAU),
        "gamma": float(bt.DEFAULT_GAMMA),
        "epsilon": float(bt.DEFAULT_EPSILON),
        "use_xg": False,
        "continuous_xg": False,
        "market_prior": False,
    }


def _paper_cache_path(config: dict) -> Path:
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return (bt.CACHE_DIR / "paper_baseline" /
            f"paper_{_season_tag(config['season'])}_{digest}.pkl")


def _load_paper_cache(path: Path, config: dict, n: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("config") != config:
            return None
        probs = np.asarray(payload["probs_paper"], dtype=float)
        if probs.shape != (n, 3):
            return None
        return probs
    except (OSError, EOFError, pickle.PickleError,
            KeyError, TypeError, ValueError) as exc:
        print(f"  Paper-Baseline-Cache ignoriert ({path.name}: {exc})")
        return None


def _save_paper_cache(path: Path, config: dict, probs_paper: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    payload = {"config": config, "probs_paper": np.asarray(probs_paper, dtype=float)}
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _attach_paper_baseline(comparison: dict, probs_paper: np.ndarray) -> dict:
    probs_paper = np.asarray(probs_paper, dtype=float)
    n, cutoff = comparison["n"], comparison["cutoff"]
    if probs_paper.shape != (n, 3):
        raise ValueError(
            f"Paper-Cache shape {probs_paper.shape} passt nicht zu Saison-n={n}."
        )

    out = comparison["outcomes"]
    hold = np.zeros(n, dtype=bool)
    hold[cutoff:] = True
    common = (hold
              & np.all(np.isfinite(comparison["probs_model"]), axis=1)
              & np.all(np.isfinite(comparison["probs_bookmaker"]), axis=1)
              & np.all(np.isfinite(probs_paper), axis=1))
    comparison["paper"] = evaluate_predictions(probs_paper[common], out[common])
    comparison["probs_paper"] = probs_paper
    return comparison


def _add_cached_paper_baseline(comparison: dict, df: pd.DataFrame, season: str, *,
                               holdout_frac: float, n_chains: int,
                               jitter_sd: float, verbose: bool) -> dict:
    config = _paper_cache_config(
        df, season, holdout_frac=holdout_frac,
        n_chains=n_chains, jitter_sd=jitter_sd,
    )
    path = _paper_cache_path(config)
    probs_paper = _load_paper_cache(path, config, comparison["n"])
    if probs_paper is not None:
        print(f"  + Paper-Standard-Baseline aus Cache: {path.name}")
        return _attach_paper_baseline(comparison, probs_paper)

    print("  + Paper-Standard-Baseline (Cache miss, zweiter Walk-Forward) ...")
    wf = bt.walkforward_predictions(
        df, season, use_xg=False,
        tau=bt.DEFAULT_TAU, gamma=bt.DEFAULT_GAMMA, eps=bt.DEFAULT_EPSILON,
        holdout_frac=holdout_frac, verbose=verbose,
        n_chains=n_chains, jitter_sd=jitter_sd,
        market_values=None, market_kappa=0.0, continuous_xg=False,
        use_team_home_advantage=False,
    )
    probs_paper = wf["probs_model"]
    _save_paper_cache(path, config, probs_paper)
    print(f"    Paper-Baseline-Cache geschrieben: {path.name}")
    return _attach_paper_baseline(comparison, probs_paper)


def _supported_parallel_seasons(n_seasons: int, n_chains: int) -> int:
    requested = min(MAX_PARALLEL_SEASONS, n_seasons)
    if requested <= 1:
        return 1

    logical_cores = os.cpu_count()
    needed = requested * max(1, n_chains)
    if logical_cores is None:
        print("  CPU-Check: Kernzahl unbekannt -> Saisons laufen nacheinander.")
        return 1
    if logical_cores < needed:
        print(f"  CPU-Check: {logical_cores} logische Kerne, "
              f"{needed} fuer {requested} parallele Saisons noetig "
              "-> nacheinander.")
        return 1

    print(f"  CPU-Check: {logical_cores} logische Kerne, "
          f"{needed} Worker fuer {requested} parallele Saisons -> OK.")
    return requested


def _run_season_backtest(df: pd.DataFrame, season: str, *,
                         n_chains: int, verbose: bool) -> dict:
    t0 = time.time()
    if bt.USE_TUNED_HYPERPARAMS:
        tau, gamma, eps = bt.resolve_hyperparams(season, bt.USE_XG)
    else:
        tau, gamma, eps = bt.DEFAULT_TAU, bt.DEFAULT_GAMMA, bt.DEFAULT_EPSILON

    if verbose:
        print(f"\n  -- Saison {season}  (tau={tau}, gamma={gamma}, eps={eps}) "
              f"{'-'*28}")

    mv, mk = None, 0.0
    if bt.USE_MARKET_PRIOR:
        from market_value import team_market_values
        mv = team_market_values(season, csv_path=bt.MARKET_VALUE_CSV)
        mk = bt.MARKET_PRIOR_KAPPA

    cmp = bt.evaluate_season_walkforward(
        df, season, use_xg=bt.USE_XG, tau=tau, gamma=gamma, eps=eps,
        holdout_frac=bt.HOLDOUT_FRAC, verbose=verbose,
        n_chains=n_chains, jitter_sd=bt.CHAIN_JITTER_SD,
        market_values=mv, market_kappa=mk,
        use_team_home_advantage=bt.USE_TEAM_HOME_ADVANTAGE,
        home_adv_prior_sd=bt.TEAM_HOME_ADV_PRIOR_SD,
    )

    if bt.INCLUDE_PAPER_BASELINE:
        _add_cached_paper_baseline(
            cmp, df, season, holdout_frac=bt.HOLDOUT_FRAC,
            n_chains=n_chains, jitter_sd=bt.CHAIN_JITTER_SD,
            verbose=verbose,
        )
        idx, rps_m, rps_p, rps_b = _per_match_rps3(cmp)
    else:
        idx, rps_m, rps_b = _per_match_rps(cmp)
        rps_p = None

    out = np.asarray(cmp["outcomes"], dtype=int)[idx]
    probs_v2 = np.asarray(cmp["probs_model"], dtype=float)[idx]
    probs_book = np.asarray(cmp["probs_bookmaker"], dtype=float)[idx]
    match_frame = pd.DataFrame({
        "season": season,
        "team_home_advantage": bool(bt.USE_TEAM_HOME_ADVANTAGE),
        "team_home_prior_sd": float(bt.TEAM_HOME_ADV_PRIOR_SD),
        "match_id": np.asarray(cmp["match_ids"], dtype=object)[idx],
        "date": pd.to_datetime(np.asarray(cmp["dates"])[idx]).strftime("%Y-%m-%d"),
        "home_team": np.asarray(cmp["home_teams"], dtype=object)[idx],
        "away_team": np.asarray(cmp["away_teams"], dtype=object)[idx],
        "home_goals": np.asarray(cmp["home_goals"], dtype=int)[idx],
        "away_goals": np.asarray(cmp["away_goals"], dtype=int)[idx],
        "season_game": idx + 1,
        "holdout_game": np.arange(1, len(rps_m) + 1),
        "actual_result": OUTCOME_LABELS[out],
        "actual_result_idx": out,
        "p_home_v2": probs_v2[:, 0],
        "p_draw_v2": probs_v2[:, 1],
        "p_away_v2": probs_v2[:, 2],
        "p_home_book": probs_book[:, 0],
        "p_draw_book": probs_book[:, 1],
        "p_away_book": probs_book[:, 2],
        "rps_v2": rps_m,
        "rps_book": rps_b,
        "d_book_v2": rps_b - rps_m,
    })
    if rps_p is not None:
        match_frame["rps_paper"] = rps_p
        match_frame["d_paper_v2"] = rps_p - rps_m

    rel = (rps_b.mean() - rps_m.mean()) / rps_b.mean() * 100.0
    row = {
        "season": season, "n": len(rps_m),
        "rps_model": float(rps_m.mean()), "rps_book": float(rps_b.mean()),
        "d_rps": float(rps_b.mean() - rps_m.mean()), "rel": rel,
        "rhat": cmp.get("rhat_max"),
    }
    if rps_p is not None:
        row["rps_paper"] = float(rps_p.mean())
        row["d_paper"] = float(rps_p.mean() - rps_m.mean())

    return {
        "season": season,
        "rps_m": rps_m,
        "rps_b": rps_b,
        "rps_p": rps_p,
        "match_frame": match_frame,
        "row": row,
        "elapsed_min": (time.time() - t0) / 60.0,
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def _setup_plotting():
    """Load matplotlib only at the end so worker processes stay lightweight."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    })
    return plt


def _timestamped_output_dir(prefix: str) -> Path:
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


def _effect_by_season(match_df: pd.DataFrame, seasons: list[str], col: str
                      ) -> pd.DataFrame:
    """Mean + normal-approx CI for match-level differences by season."""
    g = (match_df.groupby("season", observed=False)[col]
         .agg(["mean", "std", "count"]).reset_index())
    g["season"] = pd.Categorical(g["season"], seasons, ordered=True)
    g = g.sort_values("season").reset_index(drop=True)
    g["std"] = g["std"].fillna(0.0)
    g["se"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
    g["ci_lo"] = g["mean"] - 1.96 * g["se"]
    g["ci_hi"] = g["mean"] + 1.96 * g["se"]
    return g


def _annotate_bars(ax, bars, fmt="{:.4f}", pad=0.002):
    for bar in bars:
        h = bar.get_height()
        if not np.isfinite(h):
            continue
        y = h + pad if h >= 0 else h - pad
        va = "bottom" if h >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y, fmt.format(h),
                ha="center", va=va, fontsize=9)


def _plot_rps_overview(season_df: pd.DataFrame, have_paper: bool,
                       stats: dict, stats_vp: dict | None,
                       out_path: Path) -> None:
    plt = _setup_plotting()
    palette = {
        "v2": "#1f77b4",
        "paper": "#ff7f0e",
        "book": "#d62728",
        "delta_book": "#2ca02c",
        "delta_paper": "#9467bd",
    }

    seasons = season_df["season"].tolist()
    x = np.arange(len(seasons))
    rps_keys = [("rps_model", "v2", palette["v2"])]
    if have_paper:
        rps_keys.append(("rps_paper", "Paper baseline", palette["paper"]))
    rps_keys.append(("rps_book", "Bookmaker", palette["book"]))
    width = min(0.25, 0.75 / len(rps_keys))

    fig, axes = plt.subplots(
        2, 1, figsize=(12.5, 8.2), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.1]},
    )
    ax = axes[0]
    offsets = (np.arange(len(rps_keys)) - (len(rps_keys) - 1) / 2) * width
    all_vals = []
    for off, (col, label, color) in zip(offsets, rps_keys):
        vals = season_df[col].to_numpy(dtype=float)
        all_vals.extend(vals[np.isfinite(vals)])
        bars = ax.bar(x + off, vals, width, label=label, color=color,
                      edgecolor="black", linewidth=0.5, alpha=0.92)
        _annotate_bars(ax, bars, "{:.3f}", pad=0.0015)

    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        pad = max(0.006, (hi - lo) * 0.25)
        ax.set_ylim(max(0.0, lo - pad), hi + pad)
    ax.set_ylabel("Mean RPS (lower is better)")
    ax.set_title("Walk-forward predictive quality by season")
    ax.legend(loc="upper left", ncol=len(rps_keys))

    ax2 = axes[1]
    delta_keys = [("d_rps", "Bookmaker - v2", palette["delta_book"])]
    if have_paper:
        delta_keys.append(("d_paper", "Paper - v2", palette["delta_paper"]))
    d_width = min(0.34, 0.65 / len(delta_keys))
    d_offsets = (np.arange(len(delta_keys)) - (len(delta_keys) - 1) / 2) * d_width
    delta_vals = []
    for off, (col, label, color) in zip(d_offsets, delta_keys):
        vals = season_df[col].to_numpy(dtype=float)
        delta_vals.extend(vals[np.isfinite(vals)])
        bars = ax2.bar(x + off, vals, d_width, label=label, color=color,
                       edgecolor="black", linewidth=0.5, alpha=0.9)
        _annotate_bars(ax2, bars, "{:+.4f}", pad=0.0007)
    ax2.axhline(0.0, color="black", linewidth=1.0, alpha=0.75)
    if delta_vals:
        lo, hi = min(delta_vals), max(delta_vals)
        pad = max(0.0015, (hi - lo) * 0.35)
        ax2.set_ylim(lo - pad, hi + pad)
    ax2.set_ylabel("Delta RPS\n(positive = v2 better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(seasons)
    ax2.legend(loc="upper left", ncol=len(delta_keys))

    subtitle = (
        f"Pooled v2 vs bookmaker: Delta={stats['mean_diff']:+.5f}, "
        f"95%-CI [{stats['ci_lo']:+.5f}, {stats['ci_hi']:+.5f}], "
        f"Wilcoxon p={stats['wilcoxon_p']:.4f}"
    )
    if have_paper and stats_vp is not None:
        subtitle += (
            f"\nPooled v2 vs paper: Delta={stats_vp['mean_diff']:+.5f}, "
            f"95%-CI [{stats_vp['ci_lo']:+.5f}, {stats_vp['ci_hi']:+.5f}], "
            f"Wilcoxon p={stats_vp['wilcoxon_p']:.4f}"
        )
    fig.suptitle("Multi-season backtest: RPS overview\n" + subtitle,
                 y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_effect_forest(match_df: pd.DataFrame, seasons: list[str],
                        stats: dict, stats_vp: dict | None,
                        have_paper: bool, out_path: Path) -> None:
    plt = _setup_plotting()
    comparisons = [
        ("d_book_v2", "v2 vs bookmaker", "bookmaker - v2", stats, "#2ca02c"),
    ]
    if have_paper and stats_vp is not None:
        comparisons.append(
            ("d_paper_v2", "v2 vs paper baseline", "paper - v2",
             stats_vp, "#9467bd")
        )

    fig, axes = plt.subplots(1, len(comparisons),
                             figsize=(8.2 * len(comparisons), 5.6),
                             squeeze=False, sharey=False)

    for ax, (col, title, xlab, pooled, color) in zip(axes[0], comparisons):
        g = _effect_by_season(match_df, seasons, col)
        labels = g["season"].astype(str).tolist() + ["pooled"]
        y = np.arange(len(labels))

        for i, row in g.iterrows():
            mean = float(row["mean"])
            lo = float(row["ci_lo"])
            hi = float(row["ci_hi"])
            ax.errorbar(mean, y[i],
                        xerr=[[mean - lo], [hi - mean]],
                        fmt="o", color=color, ecolor=color,
                        elinewidth=2.0, capsize=4, markersize=7, alpha=0.9)
            ax.text(hi + 0.0005, y[i], f"n={int(row['count'])}",
                    va="center", fontsize=9, color="#555555")

        py = len(labels) - 1
        mean = pooled["mean_diff"]
        lo, hi = pooled["ci_lo"], pooled["ci_hi"]
        ax.errorbar(mean, py,
                    xerr=[[mean - lo], [hi - mean]],
                    fmt="D", color="black", ecolor="black",
                    elinewidth=2.4, capsize=5, markersize=8)
        ax.text(hi + 0.0005, py, f"n={pooled['n']}",
                va="center", fontsize=9, color="#333333")

        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.75)
        ax.margins(x=0.18)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(f"Delta RPS ({xlab})\npositive = v2 better")
        ax.set_title(
            f"{title}\nmean={mean:+.5f}, p={pooled['wilcoxon_p']:.4f}, "
            f"{pooled['frac_model_better']*100:.1f}% matches v2 better"
        )

    fig.suptitle("Paired effect sizes across seasons and matches",
                 y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_match_distributions(match_df: pd.DataFrame, seasons: list[str],
                              have_paper: bool, out_path: Path) -> None:
    plt = _setup_plotting()
    comparisons = [
        ("d_book_v2", "Bookmaker - v2", "#2ca02c"),
    ]
    if have_paper:
        comparisons.append(("d_paper_v2", "Paper - v2", "#9467bd"))

    fig, axes = plt.subplots(2, len(comparisons),
                             figsize=(7.2 * len(comparisons), 8.0),
                             squeeze=False)
    for j, (col, label, color) in enumerate(comparisons):
        vals = match_df[col].dropna().to_numpy(dtype=float)
        ax = axes[0, j]
        ax.hist(vals, bins=34, color=color, alpha=0.82,
                edgecolor="white", linewidth=0.7)
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.75,
                   label="no difference")
        ax.axvline(vals.mean(), color="#111111", linestyle="--", linewidth=1.6,
                   label=f"mean {vals.mean():+.4f}")
        ax.set_title(f"Match-level differences: {label}")
        ax.set_xlabel("Delta RPS (positive = v2 better)")
        ax.set_ylabel("Number of matches")
        ax.legend(loc="upper right")

        axb = axes[1, j]
        data = [
            match_df.loc[match_df["season"] == s, col].dropna().to_numpy(dtype=float)
            for s in seasons
        ]
        bp = axb.boxplot(data, labels=seasons, patch_artist=True,
                         showfliers=False, medianprops={"color": "black"})
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.55, edgecolor="black")
        axb.axhline(0.0, color="black", linewidth=1.0, alpha=0.75)
        axb.set_title(f"Season-level spread: {label}")
        axb.set_ylabel("Delta RPS")
        axb.tick_params(axis="x", rotation=25)

    fig.suptitle("Where does the pooled effect come from? Match-level distribution",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_visual_outputs(report_dir: Path,
                          per_season: list[dict], match_df: pd.DataFrame,
                          stats: dict, stats_vp: dict | None,
                          have_paper: bool, seasons: list[str]) -> list[Path]:
    """Write CSVs and presentation-ready PNGs for the report."""
    report_dir.mkdir(parents=True, exist_ok=True)
    season_df = pd.DataFrame(per_season)
    if not match_df.empty:
        match_df = match_df.copy()
        match_df["season"] = pd.Categorical(match_df["season"], seasons, ordered=True)
        match_df = match_df.sort_values("season").reset_index(drop=True)

    season_csv = report_dir / "multiseason_per_season.csv"
    match_csv = report_dir / "multiseason_per_match_rps.csv"
    summary_csv = report_dir / "multiseason_summary.csv"
    season_df.to_csv(season_csv, index=False)
    match_df.to_csv(match_csv, index=False)

    summary_rows = [{
        "comparison": "v2_vs_bookmaker",
        "team_home_advantage": bool(bt.USE_TEAM_HOME_ADVANTAGE),
        "team_home_prior_sd": float(bt.TEAM_HOME_ADV_PRIOR_SD),
        "n": stats["n"],
        "mean_delta_rps": stats["mean_diff"],
        "rel_pct": stats["rel_pct"],
        "ci_lo": stats["ci_lo"],
        "ci_hi": stats["ci_hi"],
        "wilcoxon_p": stats["wilcoxon_p"],
        "frac_v2_better": stats["frac_model_better"],
    }]
    if have_paper and stats_vp is not None:
        summary_rows.append({
            "comparison": "v2_vs_paper",
            "team_home_advantage": bool(bt.USE_TEAM_HOME_ADVANTAGE),
            "team_home_prior_sd": float(bt.TEAM_HOME_ADV_PRIOR_SD),
            "n": stats_vp["n"],
            "mean_delta_rps": stats_vp["mean_diff"],
            "rel_pct": stats_vp["rel_pct"],
            "ci_lo": stats_vp["ci_lo"],
            "ci_hi": stats_vp["ci_hi"],
            "wilcoxon_p": stats_vp["wilcoxon_p"],
            "frac_v2_better": stats_vp["frac_model_better"],
        })
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    paths = [season_csv, match_csv, summary_csv]
    if match_df.empty:
        return paths

    overview_png = report_dir / "15_multiseason_rps_overview.png"
    forest_png = report_dir / "16_multiseason_effect_ci.png"
    dist_png = report_dir / "17_multiseason_match_distributions.png"
    _plot_rps_overview(season_df, have_paper, stats, stats_vp, overview_png)
    _plot_effect_forest(match_df, seasons, stats, stats_vp,
                        have_paper, forest_png)
    _plot_match_distributions(match_df, seasons, have_paper, dist_png)
    paths.extend([overview_png, forest_png, dist_png])
    return paths


def main():
    t_start = time.time()
    print("=" * 72)
    print(" MULTI-SEASON-SIGNIFIKANZSTUDIE — Modell vs. Buchmacher-Schlusslinie")
    print("=" * 72)

    if not bt.USE_XG:
        raise ValueError("Diese Studie braucht USE_XG=True in backtest.py.")
    if bt.USE_TUNED_HYPERPARAMS:
        print("  HINWEIS: backtest.USE_TUNED_HYPERPARAMS=True — für die saubere "
              "Studie besser False (leckfreie feste Paper-Hyperparameter).")

    # 1) Daten + Quoten
    df = load_bundesliga(bt.START_YEAR, bt.END_YEAR, with_extras=True)
    df = extract_bookmaker_probs(df)

    seasons = SEASONS if SEASONS is not None else _auto_seasons(df)
    print(f"  Saisons ({len(seasons)}): {seasons}")

    # 2) xG: Proxy-β leckfrei vor der frühesten Analyse-Saison fitten
    #    (bei echtem xG ohnehin nur Fallback < 2014/15 — wird hier nie gescort).
    earliest = min(seasons)
    df_train = df[df["Season"] < earliest]
    beta_off, beta_on = fit_xg_weights(
        df_train if len(df_train) > 100 else df, force=True, cache=False)
    df = add_xg_columns(df, beta_off, beta_on)
    if bt.USE_REAL_XG:
        df = add_real_xg_columns(df)
    if SCRAMBLE_XG:
        print("  ⚠ PLACEBO-MODUS: xG wird je Saison verwürfelt (Negativ-Kontrolle).")
        df = _scramble_xg(df, SCRAMBLE_SEED)

    # 3) Quellen-Audit pro Saison (welche Quote? echtes/Proxy-xG?)
    print(f"\n  {'Saison':9} {'bm_source (ganze Saison)':<34} {'xg_source':<28}")
    print("  " + "-" * 70)
    for s in seasons:
        sub = df[df["Season"] == s]
        bm = ", ".join(f"{k}={v}" for k, v in sub["bm_source"].value_counts().items())
        xs = ", ".join(f"{k}={v}" for k, v in sub.get(
            "xg_source", pd.Series(["?"] * len(sub))).value_counts().items())
        print(f"  {s:9} {bm:<34} {xs:<28}")

    # 4) JIT + Ketten
    print("\n  Numba-JIT vorkompilieren ...", end="", flush=True)
    warmup_jit()
    print(" OK")
    n_chains = bt.N_CHAINS if bt.USE_MULTI_CHAIN else 1
    season_workers = _supported_parallel_seasons(len(seasons), n_chains)

    # 5) Walk-Forward je Saison, Per-Spiel-RPS poolen
    all_m, all_b, all_p = [], [], []
    per_season = []
    per_match_frames = []
    if season_workers == 1:
        season_results = [
            _run_season_backtest(df, s, n_chains=n_chains, verbose=True)
            for s in seasons
        ]
    else:
        print(f"\n  Starte {season_workers} parallele Saison-Jobs "
              f"(je {n_chains} MCMC-Ketten).")
        for s in seasons:
            print(f"    queued: {s}")
        season_results = []
        with ThreadPoolExecutor(max_workers=season_workers) as executor:
            futures = {
                executor.submit(
                    _run_season_backtest, df, s,
                    n_chains=n_chains, verbose=False,
                ): s
                for s in seasons
            }
            for future in as_completed(futures):
                s = futures[future]
                result = future.result()
                season_results.append(result)
                print(f"    fertig: {s} ({result['elapsed_min']:.1f} min)")

        order = {s: i for i, s in enumerate(seasons)}
        season_results.sort(key=lambda r: order[r["season"]])

    for result in season_results:
        all_m.append(result["rps_m"])
        all_b.append(result["rps_b"])
        if result["rps_p"] is not None:
            all_p.append(result["rps_p"])
        per_match_frames.append(result["match_frame"])
        per_season.append(result["row"])

    rps_model = np.concatenate(all_m)
    rps_book = np.concatenate(all_b)
    match_df = (pd.concat(per_match_frames, ignore_index=True)
                if per_match_frames else pd.DataFrame())
    stats = _paired_stats(rps_model, rps_book, BOOTSTRAP_N, BOOTSTRAP_SEED)

    # v2 vs. Paper-Standardmodell (gepoolt, gepaart): diff = Paper − v2 (>0 = v2 besser)
    have_paper = len(all_p) == len(seasons) and len(all_p) > 0
    rps_paper = np.concatenate(all_p) if have_paper else None
    stats_vp = (_paired_stats(rps_model, rps_paper, BOOTSTRAP_N, BOOTSTRAP_SEED)
                if have_paper else None)

    # 6) Bericht
    print("\n" + "=" * 72)
    print(" ERGEBNIS")
    print("=" * 72)
    if have_paper:
        print(f"\n  {'Saison':9} {'n':>4} {'RPS_v2':>9} {'RPS_Paper':>10} "
              f"{'RPS_Buch':>9} {'Δ(B-v2)':>9} {'Δ(v2-Pap)':>10} {'R-hat':>6}")
        print("  " + "-" * 74)
    else:
        print(f"\n  {'Saison':9} {'n':>4} {'RPS_Modell':>11} {'RPS_Buch':>10} "
              f"{'ΔRPS':>9} {'rel%':>7} {'R-hat':>7}")
        print("  " + "-" * 64)
    n_better = 0
    for r in per_season:
        n_better += int(r["d_rps"] > 0)
        rh = f"{r['rhat']:.2f}" if r["rhat"] is not None and np.isfinite(r["rhat"]) else "—"
        if have_paper:
            print(f"  {r['season']:9} {r['n']:>4} {r['rps_model']:>9.4f} "
                  f"{r['rps_paper']:>10.4f} {r['rps_book']:>9.4f} "
                  f"{r['d_rps']:>+9.4f} {r['d_paper']:>+10.4f} {rh:>6}")
        else:
            print(f"  {r['season']:9} {r['n']:>4} {r['rps_model']:>11.4f} "
                  f"{r['rps_book']:>10.4f} {r['d_rps']:>+9.4f} {r['rel']:>+7.1f} {rh:>7}")

    print("\n  Gepoolt — Modell (v2) vs. Buchmacher:")
    print(f"    Spiele insgesamt:       n = {stats['n']}")
    print(f"    RPS Modell / Buchmacher: {rps_model.mean():.4f} / {rps_book.mean():.4f}")
    print(f"    mittleres ΔRPS (Buch−Modell): {stats['mean_diff']:+.5f} "
          f"({stats['rel_pct']:+.2f} % rel.)")
    print(f"    Bootstrap-95%-CI von ΔRPS:    "
          f"[{stats['ci_lo']:+.5f}, {stats['ci_hi']:+.5f}]")
    print(f"    Wilcoxon-Signed-Rank p:       {stats['wilcoxon_p']:.4f}")
    print(f"    Spiele Modell besser:         {stats['frac_model_better']*100:.1f} %")
    print(f"    Saisons Modell besser:        {n_better}/{len(per_season)}")

    if have_paper:
        n_better_p = sum(int(r.get("d_paper", 0.0) > 0) for r in per_season)
        print("\n  Gepoolt — v2 vs. Paper-Standardmodell (Effekt der Anpassungen):")
        print(f"    RPS v2 / Paper:               {rps_model.mean():.4f} / {rps_paper.mean():.4f}")
        print(f"    mittleres ΔRPS (Paper−v2):    {stats_vp['mean_diff']:+.5f} "
              f"({stats_vp['rel_pct']:+.2f} % rel.)")
        print(f"    Bootstrap-95%-CI von ΔRPS:    "
              f"[{stats_vp['ci_lo']:+.5f}, {stats_vp['ci_hi']:+.5f}]")
        print(f"    Wilcoxon-Signed-Rank p:       {stats_vp['wilcoxon_p']:.4f}")
        print(f"    Spiele v2 besser als Paper:   {stats_vp['frac_model_better']*100:.1f} %")
        print(f"    Saisons v2 besser als Paper:  {n_better_p}/{len(per_season)}")
        sig_vp = ((stats_vp["ci_lo"] > 0 or stats_vp["ci_hi"] < 0)
                  and stats_vp["wilcoxon_p"] < 0.05)
        if sig_vp and stats_vp["mean_diff"] > 0:
            print("    → v2 schlägt das Paper-Modell SIGNIFIKANT (CI schließt 0 aus, p<0.05).")
        elif sig_vp and stats_vp["mean_diff"] < 0:
            print("    → Paper-Modell SIGNIFIKANT besser als v2.")
        else:
            print("    → kein signifikanter Unterschied v2 vs. Paper (CI überdeckt 0).")

    # 7) Urteil
    sig = (stats["ci_lo"] > 0 or stats["ci_hi"] < 0) and stats["wilcoxon_p"] < 0.05
    print("\n  → ", end="")
    if SCRAMBLE_XG:
        if sig and stats["mean_diff"] > 0:
            print("PLACEBO zeigt TROTZDEM einen Vorsprung → PIPELINE-LECK! "
                  "(verwürfeltes xG dürfte nicht helfen.)")
        else:
            print("PLACEBO: kein Vorsprung mit verwürfeltem xG → konsistent mit "
                  "'Signal sitzt echt im xG, kein Leck'.")
    elif sig and stats["mean_diff"] > 0:
        print("SIGNIFIKANTER Vorsprung des Modells (CI schließt 0 aus, p<0.05).")
    elif sig and stats["mean_diff"] < 0:
        print("Buchmacher SIGNIFIKANT besser.")
    else:
        print("KEIN signifikanter Unterschied — auf Augenhöhe mit der "
              "Schlusslinie (CI überdeckt 0).")

    print("\n  Visualisierungen und Tabellen schreiben ...")
    report_prefix = "multiseason_placebo" if SCRAMBLE_XG else "multiseason"
    report_dir = _timestamped_output_dir(report_prefix)
    out_paths = _write_visual_outputs(
        report_dir, per_season, match_df, stats, stats_vp, have_paper, seasons,
    )
    for p in out_paths:
        print(f"    {p}")

    print(f"\n  Fertig in {(time.time()-t_start)/60:.1f} min.")


if __name__ == "__main__":
    main()
