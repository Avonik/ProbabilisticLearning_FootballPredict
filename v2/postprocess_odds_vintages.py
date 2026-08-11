"""
postprocess_odds_vintages.py
============================
Compare a completed multi-season v2 run against weaker bookmaker vintages.

This does not refit the model. It joins the saved v2 holdout probabilities from
``multiseason_per_match_rps.csv`` with local football-data.co.uk odds columns
and evaluates alternative bookmaker baselines such as opening odds.

Usage:
    python postprocess_odds_vintages.py
    python postprocess_odds_vintages.py output/multiseason_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data import MATCH_ID_COL, sort_matches
from scipy.stats import wilcoxon


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_CACHE = SCRIPT_DIR / "data_cache"
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 23


@dataclass(frozen=True)
class OddsSource:
    key: str
    label: str
    columns: tuple[str, str, str]
    note: str


ODDS_SOURCES = [
    OddsSource(
        "pinnacle_open",
        "Pinnacle opening",
        ("PSH", "PSD", "PSA"),
        "Pinnacle non-closing 1X2 odds from football-data.co.uk.",
    ),
    OddsSource(
        "bet365_open",
        "Bet365 opening",
        ("B365H", "B365D", "B365A"),
        "Bet365 non-closing 1X2 odds from football-data.co.uk.",
    ),
    OddsSource(
        "market_avg_open",
        "Market average opening",
        ("AvgH", "AvgD", "AvgA"),
        "Market average non-closing odds; available mainly for newer seasons.",
    ),
    OddsSource(
        "betbrain_avg_open",
        "BetBrain average opening",
        ("BbAvH", "BbAvD", "BbAvA"),
        "Older market-average non-closing odds from BetBrain columns.",
    ),
]
CLOSING_PRIORITY = [
    ("PSCH", "PSCD", "PSCA"),
    ("AvgCH", "AvgCD", "AvgCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("PSH", "PSD", "PSA"),
    ("B365H", "B365D", "B365A"),
]
RAW_ODDS_COLUMNS = sorted({
    col
    for source in ODDS_SOURCES
    for col in source.columns
} | {
    "PSCH", "PSCD", "PSCA",
    "AvgCH", "AvgCD", "AvgCA",
    "B365CH", "B365CD", "B365CA",
})


def _candidate_output_roots() -> list[Path]:
    roots = [SCRIPT_DIR / "output", Path.cwd() / "output"]
    out: list[Path] = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            out.append(root)
            seen.add(resolved)
    return out


def _latest_run_dir() -> Path:
    candidates: list[Path] = []
    for root in _candidate_output_roots():
        candidates.extend(root.glob("multiseason_*/multiseason_per_match_rps.csv"))
    if not candidates:
        roots = ", ".join(str(r) for r in _candidate_output_roots())
        raise FileNotFoundError(
            f"No multiseason_per_match_rps.csv found below: {roots}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime).parent


def _run_dir_from_arg(path_arg: str | None) -> Path:
    if path_arg is None:
        return _latest_run_dir()
    path = Path(path_arg)
    if path.is_file():
        return path.parent
    return path


def _season_start_year(label: str) -> int:
    return int(str(label).split("/")[0])


def _season_code(label: str) -> str:
    year = _season_start_year(label)
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def _read_season_raw(season: str) -> pd.DataFrame:
    path = DATA_CACHE / f"D1_{_season_code(season)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached football-data file: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    if "Div" in df.columns:
        df = df[df["Div"].astype(str).str.strip() == "D1"].copy()
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    df = df.dropna(subset=[c for c in required if c in df.columns]).copy()
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    for col in df.columns:
        if col not in {"Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTR", "HTR"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Season"] = str(season)
    return sort_matches(df)


def _load_match_data(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "multiseason_per_match_rps.csv"
    df = pd.read_csv(csv_path)
    required = {
        "season",
        "match_id",
        "season_game",
        "actual_result_idx",
        "p_home_v2",
        "p_draw_v2",
        "p_away_v2",
        "rps_v2",
        "rps_book",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "This run is missing odds-comparison columns: " + ", ".join(missing)
        )
    return df


def _probs_from_cols(df: pd.DataFrame, cols: tuple[str, str, str]) -> np.ndarray:
    odds = df.loc[:, list(cols)].to_numpy(dtype=float)
    valid = np.all(np.isfinite(odds), axis=1) & np.all(odds > 1.0, axis=1)
    probs = np.full_like(odds, np.nan, dtype=float)
    inv = np.zeros_like(odds, dtype=float)
    inv[valid] = 1.0 / odds[valid]
    probs[valid] = inv[valid] / inv[valid].sum(axis=1, keepdims=True)
    return probs


def _closing_probs_by_priority(raw: pd.DataFrame) -> np.ndarray:
    probs = np.full((len(raw), 3), np.nan, dtype=float)
    for cols in CLOSING_PRIORITY:
        if not all(col in raw.columns for col in cols):
            continue
        candidate = _probs_from_cols(raw, cols)
        fill = np.isnan(probs[:, 0]) & np.all(np.isfinite(candidate), axis=1)
        probs[fill] = candidate[fill]
    return probs


def _join_raw_matches(match_df: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    for season, sub in match_df.groupby("season", sort=False):
        raw = _read_season_raw(str(season))
        raw = raw.reset_index(names="raw_index")
        raw_cols = [MATCH_ID_COL, "raw_index", "Date", "HomeTeam", "AwayTeam",
                    "FTHG", "FTAG"] + [c for c in RAW_ODDS_COLUMNS if c in raw]
        raw_sub = raw[raw_cols].rename(columns={
            MATCH_ID_COL: "match_id",
            "Date": "raw_date",
            "HomeTeam": "raw_home_team",
            "AwayTeam": "raw_away_team",
            "FTHG": "raw_home_goals",
            "FTAG": "raw_away_goals",
        })
        joined = sub.reset_index(drop=True).merge(
            raw_sub, on="match_id", how="left", validate="one_to_one",
            indicator=True,
        )
        missing = joined["_merge"] != "both"
        if missing.any():
            ids = ", ".join(joined.loc[missing, "match_id"].astype(str).head(3))
            raise ValueError(f"Raw odds missing for {missing.sum()} MatchIDs: {ids}")
        joined = joined.drop(columns="_merge")

        identity_checks = [
            ("home_team", "raw_home_team"),
            ("away_team", "raw_away_team"),
            ("home_goals", "raw_home_goals"),
            ("away_goals", "raw_away_goals"),
        ]
        for stored_col, raw_col in identity_checks:
            if stored_col not in joined:
                continue
            if "goals" in stored_col:
                same = np.allclose(
                    pd.to_numeric(joined[stored_col], errors="coerce"),
                    pd.to_numeric(joined[raw_col], errors="coerce"),
                    equal_nan=True,
                )
            else:
                same = np.array_equal(
                    joined[stored_col].astype(str).to_numpy(),
                    joined[raw_col].astype(str).to_numpy(),
                )
            if not same:
                raise ValueError(
                    f"Match identity mismatch in {stored_col} for season {season}."
                )
        joined["matched_season_game"] = joined["raw_index"].to_numpy(dtype=int) + 1
        joined["date"] = joined["raw_date"].dt.strftime("%Y-%m-%d")
        joined["home_team"] = joined["raw_home_team"]
        joined["away_team"] = joined["raw_away_team"]
        joined["home_goals"] = joined["raw_home_goals"]
        joined["away_goals"] = joined["raw_away_goals"]
        joined = joined.drop(columns=[
            "raw_index", "raw_date", "raw_home_team", "raw_away_team",
            "raw_home_goals", "raw_away_goals",
        ])
        for col in RAW_ODDS_COLUMNS:
            if col not in joined:
                joined[col] = np.nan
        chunks.append(joined)
    return pd.concat(chunks, ignore_index=True)


def _probs_from_df(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return df[[f"p_home_{prefix}", f"p_draw_{prefix}", f"p_away_{prefix}"]].to_numpy(
        dtype=float
    )


def _rps_values(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    cum_p0 = probs[:, 0]
    cum_p1 = probs[:, 0] + probs[:, 1]
    cum_o0 = (outcomes == 0).astype(float)
    cum_o1 = (outcomes != 2).astype(float)
    return 0.5 * ((cum_p0 - cum_o0) ** 2 + (cum_p1 - cum_o1) ** 2)


def _bootstrap_ci(diff: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(diff)
    draws = rng.integers(0, n, size=(BOOTSTRAP_N, n))
    means = diff[draws].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _paired_summary(
    name: str,
    label: str,
    rps_v2: np.ndarray,
    rps_odds: np.ndarray,
    note: str,
) -> dict:
    diff = rps_odds - rps_v2
    ci_lo, ci_hi = _bootstrap_ci(diff)
    try:
        _, p_value = wilcoxon(rps_v2, rps_odds, alternative="two-sided")
    except ValueError:
        p_value = float("nan")
    return {
        "source": name,
        "label": label,
        "n": int(len(diff)),
        "rps_v2": float(rps_v2.mean()),
        "rps_odds": float(rps_odds.mean()),
        "delta_odds_minus_v2": float(diff.mean()),
        "rel_pct_vs_odds": float(diff.mean() / rps_odds.mean() * 100.0),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "wilcoxon_p": float(p_value),
        "frac_v2_better": float((diff > 0).mean()),
        "note": note,
    }


def _evaluate_sources(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = joined["actual_result_idx"].to_numpy(dtype=int)
    v2_probs = _probs_from_df(joined, "v2")
    current_probs = _probs_from_df(joined, "book")
    joined = joined.copy()
    joined["rps_v2_recomputed"] = _rps_values(v2_probs, outcomes)
    joined["rps_current_closing"] = _rps_values(current_probs, outcomes)

    source_defs = [
        ("current_closing", "Current closing baseline", None, "Stored baseline from the completed run."),
        *[(s.key, s.label, s.columns, s.note) for s in ODDS_SOURCES],
    ]

    summary_rows = []
    season_rows = []
    for key, label, cols, note in source_defs:
        if cols is None:
            rps_odds_all = joined["rps_current_closing"].to_numpy(dtype=float)
            valid = np.isfinite(rps_odds_all)
        else:
            probs = _probs_from_cols(joined, cols)
            joined[f"p_home_{key}"] = probs[:, 0]
            joined[f"p_draw_{key}"] = probs[:, 1]
            joined[f"p_away_{key}"] = probs[:, 2]
            rps_odds_all = _rps_values(probs, outcomes)
            joined[f"rps_{key}"] = rps_odds_all
            valid = np.isfinite(rps_odds_all)

        if not valid.any():
            continue

        rps_v2 = joined.loc[valid, "rps_v2_recomputed"].to_numpy(dtype=float)
        rps_odds = rps_odds_all[valid]
        summary_rows.append(_paired_summary(key, label, rps_v2, rps_odds, note))

        for season, sub in joined.loc[valid].groupby("season", sort=False):
            sub_rps_v2 = sub["rps_v2_recomputed"].to_numpy(dtype=float)
            if key == "current_closing":
                sub_rps_odds = sub["rps_current_closing"].to_numpy(dtype=float)
            else:
                sub_rps_odds = sub[f"rps_{key}"].to_numpy(dtype=float)
            diff = sub_rps_odds - sub_rps_v2
            season_rows.append({
                "season": season,
                "source": key,
                "label": label,
                "n": int(len(sub)),
                "rps_v2": float(sub_rps_v2.mean()),
                "rps_odds": float(sub_rps_odds.mean()),
                "delta_odds_minus_v2": float(diff.mean()),
                "rel_pct_vs_odds": float(diff.mean() / sub_rps_odds.mean() * 100.0),
                "frac_v2_better": float((diff > 0).mean()),
            })

    return joined, pd.DataFrame(summary_rows), pd.DataFrame(season_rows)


def _plot_results(run_dir: Path, summary: pd.DataFrame, by_season: pd.DataFrame) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    })

    preferred = [
        "current_closing",
        "pinnacle_open",
        "bet365_open",
        "market_avg_open",
        "betbrain_avg_open",
    ]
    plot_summary = (
        summary.set_index("source")
        .loc[[s for s in preferred if s in set(summary["source"])]]
        .reset_index()
    )
    labels = plot_summary["label"].str.replace(" baseline", "", regex=False).tolist()
    x = np.arange(len(plot_summary))

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)

    ax = axes[0]
    width = 0.36
    ax.bar(
        x - width / 2,
        plot_summary["rps_v2"],
        width=width,
        color="#2f7ebc",
        edgecolor="#333333",
        linewidth=0.8,
        label="v2",
    )
    ax.bar(
        x + width / 2,
        plot_summary["rps_odds"],
        width=width,
        color="#d83b3b",
        edgecolor="#333333",
        linewidth=0.8,
        label="odds baseline",
    )
    for xpos, v2, odds in zip(x, plot_summary["rps_v2"], plot_summary["rps_odds"]):
        ax.text(xpos - width / 2, v2, f"{v2:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(xpos + width / 2, odds, f"{odds:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean RPS (lower is better)")
    ax.set_title("v2 against weaker bookmaker vintages")
    ax.legend(loc="upper left")

    ax = axes[1]
    colors = {
        "current_closing": "#555555",
        "pinnacle_open": "#38a169",
        "bet365_open": "#60a5fa",
        "market_avg_open": "#9f7aea",
        "betbrain_avg_open": "#ff8c24",
    }
    for key in preferred:
        sub = by_season[by_season["source"] == key]
        if sub.empty:
            continue
        ax.plot(
            sub["season"],
            sub["delta_odds_minus_v2"],
            marker="o",
            linewidth=1.8,
            label=sub["label"].iloc[0].replace(" baseline", ""),
            color=colors.get(key),
        )
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_ylabel("Delta RPS\n(positive = v2 better)")
    ax.set_xlabel("Season")
    ax.set_title("Season-level effect by odds source")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper left", ncol=2)

    fig.suptitle(
        "Opening odds are joined post-hoc; the model is not refit.",
        fontsize=11,
    )
    path = run_dir / "19_postprocess_odds_vintages.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def run(run_dir: Path) -> dict[str, Path]:
    match_df = _load_match_data(run_dir)
    joined = _join_raw_matches(match_df)
    per_match, summary, by_season = _evaluate_sources(joined)

    per_match_path = run_dir / "postprocess_odds_vintages_per_match.csv"
    summary_path = run_dir / "postprocess_odds_vintages_summary.csv"
    by_season_path = run_dir / "postprocess_odds_vintages_by_season.csv"
    plot_path = _plot_results(run_dir, summary, by_season)

    per_match.to_csv(per_match_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_season.to_csv(by_season_path, index=False)

    return {
        "summary": summary_path,
        "by_season": by_season_path,
        "per_match": per_match_path,
        "plot": plot_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="Run folder or per-match CSV. Defaults to the newest multiseason run.",
    )
    args = parser.parse_args()
    run_dir = _run_dir_from_arg(args.run_dir).resolve()
    print(f"Post-processing run: {run_dir}")
    paths = run(run_dir)
    for label, path in paths.items():
        print(f"  {label}: {path}")

    summary = pd.read_csv(paths["summary"])
    cols = [
        "source",
        "n",
        "rps_v2",
        "rps_odds",
        "delta_odds_minus_v2",
        "rel_pct_vs_odds",
        "wilcoxon_p",
    ]
    print("\nSummary:")
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
