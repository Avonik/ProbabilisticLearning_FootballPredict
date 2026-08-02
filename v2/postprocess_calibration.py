"""
postprocess_calibration.py
==========================
Cheap post-processing experiment for completed multi-season backtests.

It reads ``multiseason_per_match_rps.csv`` from a finished run, calibrates only
the v2 probabilities, and writes CSV/PNG outputs back into that run folder.
Paper and bookmaker values are kept fixed as comparison baselines.

Usage:
    python postprocess_calibration.py
    python postprocess_calibration.py output/multiseason_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


SCRIPT_DIR = Path(__file__).resolve().parent
TEMP_GRID = np.round(np.arange(0.60, 1.81, 0.01), 2)
DRAW_GRID = np.round(np.arange(0.75, 1.31, 0.01), 2)
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 17


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


def _load_match_data(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "multiseason_per_match_rps.csv"
    df = pd.read_csv(csv_path)
    required = {
        "season",
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
            "This run is missing calibration-ready columns: "
            + ", ".join(missing)
        )
    return df


def _probs_from_df(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return df[[f"p_home_{prefix}", f"p_draw_{prefix}", f"p_away_{prefix}"]].to_numpy(
        dtype=float
    )


def _normalize(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, None)
    return probs / probs.sum(axis=1, keepdims=True)


def _rps_values(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    probs = _normalize(probs)
    outcomes = np.asarray(outcomes, dtype=int)
    cum_p0 = probs[:, 0]
    cum_p1 = probs[:, 0] + probs[:, 1]
    cum_o0 = (outcomes == 0).astype(float)
    cum_o1 = (outcomes != 2).astype(float)
    return 0.5 * ((cum_p0 - cum_o0) ** 2 + (cum_p1 - cum_o1) ** 2)


def _temperature_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    q = _normalize(probs) ** (1.0 / float(temperature))
    return _normalize(q)


def _draw_probs(probs: np.ndarray, draw_multiplier: float) -> np.ndarray:
    q = _normalize(probs).copy()
    q[:, 1] *= float(draw_multiplier)
    return _normalize(q)


def _temp_draw_probs(
    probs: np.ndarray, temperature: float, draw_multiplier: float
) -> np.ndarray:
    q = _normalize(probs) ** (1.0 / float(temperature))
    q[:, 1] *= float(draw_multiplier)
    return _normalize(q)


def _fit_temperature(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    best = {"temperature": 1.0, "draw_multiplier": 1.0, "rps": float("inf")}
    for temperature in TEMP_GRID:
        rps = float(_rps_values(_temperature_probs(probs, temperature), outcomes).mean())
        if rps < best["rps"]:
            best = {
                "temperature": float(temperature),
                "draw_multiplier": 1.0,
                "rps": rps,
            }
    return best


def _fit_draw(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    best = {"temperature": 1.0, "draw_multiplier": 1.0, "rps": float("inf")}
    for draw_multiplier in DRAW_GRID:
        rps = float(_rps_values(_draw_probs(probs, draw_multiplier), outcomes).mean())
        if rps < best["rps"]:
            best = {
                "temperature": 1.0,
                "draw_multiplier": float(draw_multiplier),
                "rps": rps,
            }
    return best


def _fit_temp_draw(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    best = {"temperature": 1.0, "draw_multiplier": 1.0, "rps": float("inf")}
    for temperature in TEMP_GRID:
        base = _temperature_probs(probs, temperature)
        for draw_multiplier in DRAW_GRID:
            rps = float(_rps_values(_draw_probs(base, draw_multiplier), outcomes).mean())
            if rps < best["rps"]:
                best = {
                    "temperature": float(temperature),
                    "draw_multiplier": float(draw_multiplier),
                    "rps": rps,
                }
    return best


FITTERS = {
    "temperature": _fit_temperature,
    "draw": _fit_draw,
    "temperature_draw": _fit_temp_draw,
}


def _apply_variant(probs: np.ndarray, variant: str, params: dict) -> np.ndarray:
    if variant == "temperature":
        return _temperature_probs(probs, params["temperature"])
    if variant == "draw":
        return _draw_probs(probs, params["draw_multiplier"])
    if variant == "temperature_draw":
        return _temp_draw_probs(
            probs, params["temperature"], params["draw_multiplier"]
        )
    raise KeyError(variant)


def _season_order(df: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(df["season"].astype(str).tolist()))


def _walkforward_calibration(
    df: pd.DataFrame, probs: np.ndarray, outcomes: np.ndarray, variant: str
) -> tuple[np.ndarray, pd.DataFrame]:
    seasons = _season_order(df)
    calibrated = np.full_like(probs, np.nan, dtype=float)
    rows = []
    fitter = FITTERS[variant]

    for season_idx, season in enumerate(seasons):
        test_mask = df["season"].astype(str).to_numpy() == season
        if season_idx == 0:
            params = {"temperature": 1.0, "draw_multiplier": 1.0, "rps": np.nan}
            train_n = 0
        else:
            train_mask = ~test_mask & df["season"].astype(str).isin(seasons[:season_idx]).to_numpy()
            params = fitter(probs[train_mask], outcomes[train_mask])
            train_n = int(train_mask.sum())

        calibrated[test_mask] = _apply_variant(probs[test_mask], variant, params)
        rps_raw = _rps_values(probs[test_mask], outcomes[test_mask])
        rps_cal = _rps_values(calibrated[test_mask], outcomes[test_mask])
        row = {
            "season": season,
            "variant": f"wf_{variant}",
            "train_n": train_n,
            "temperature": params["temperature"],
            "draw_multiplier": params["draw_multiplier"],
            "rps_raw_v2": float(rps_raw.mean()),
            "rps_calibrated_v2": float(rps_cal.mean()),
            "delta_raw_minus_calibrated": float(rps_raw.mean() - rps_cal.mean()),
            "rel_improvement_vs_raw_pct": float((rps_raw.mean() - rps_cal.mean()) / rps_raw.mean() * 100.0),
        }
        if "rps_book" in df:
            row["rps_book"] = float(df.loc[test_mask, "rps_book"].mean())
            row["delta_book_minus_calibrated"] = float(row["rps_book"] - row["rps_calibrated_v2"])
        if "rps_paper" in df:
            row["rps_paper"] = float(df.loc[test_mask, "rps_paper"].mean())
            row["delta_paper_minus_calibrated"] = float(row["rps_paper"] - row["rps_calibrated_v2"])
        rows.append(row)

    return calibrated, pd.DataFrame(rows)


def _bootstrap_ci(diff: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(diff)
    draws = rng.integers(0, n, size=(BOOTSTRAP_N, n))
    means = diff[draws].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _summary_row(
    name: str,
    rps: np.ndarray,
    raw_rps: np.ndarray,
    book_rps: np.ndarray,
    paper_rps: np.ndarray | None,
    temperature: float | None = None,
    draw_multiplier: float | None = None,
    note: str = "",
) -> dict:
    diff_raw = raw_rps - rps
    ci_lo, ci_hi = _bootstrap_ci(diff_raw)
    try:
        _, p_value = wilcoxon(rps, raw_rps, alternative="two-sided")
    except ValueError:
        p_value = float("nan")

    row = {
        "variant": name,
        "n": len(rps),
        "mean_rps": float(rps.mean()),
        "delta_raw_minus_variant": float(diff_raw.mean()),
        "rel_improvement_vs_raw_pct": float(diff_raw.mean() / raw_rps.mean() * 100.0),
        "ci_lo_raw_minus_variant": ci_lo,
        "ci_hi_raw_minus_variant": ci_hi,
        "wilcoxon_p_vs_raw": float(p_value),
        "delta_book_minus_variant": float(book_rps.mean() - rps.mean()),
        "rel_vs_book_pct": float((book_rps.mean() - rps.mean()) / book_rps.mean() * 100.0),
        "temperature": temperature,
        "draw_multiplier": draw_multiplier,
        "note": note,
    }
    if paper_rps is not None:
        row["delta_paper_minus_variant"] = float(paper_rps.mean() - rps.mean())
        row["rel_vs_paper_pct"] = float((paper_rps.mean() - rps.mean()) / paper_rps.mean() * 100.0)
    return row


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

    fair = summary[summary["variant"].str.startswith("wf_")].copy()
    fair["label"] = fair["variant"].map({
        "wf_temperature": "Temperature",
        "wf_draw": "Draw multiplier",
        "wf_temperature_draw": "Temp + draw",
    })

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    ax = axes[0]
    raw = summary.loc[summary["variant"] == "raw_v2", "mean_rps"].iloc[0]
    book = summary.loc[summary["variant"] == "bookmaker", "mean_rps"].iloc[0]
    labels = ["Raw v2", *fair["label"].tolist(), "Bookmaker"]
    values = [raw, *fair["mean_rps"].tolist(), book]
    colors = ["#2f7ebc", "#60a5fa", "#38a169", "#9f7aea", "#d83b3b"]
    if (summary["variant"] == "paper").any():
        labels.append("Paper")
        values.append(summary.loc[summary["variant"] == "paper", "mean_rps"].iloc[0])
        colors.append("#ff8c24")
    bars = ax.bar(labels, values, color=colors[:len(labels)], edgecolor="#333333", linewidth=0.8)
    ax.axhline(raw, color="#2f7ebc", linestyle="--", linewidth=1.2, alpha=0.75)
    ax.set_ylabel("Mean RPS (lower is better)")
    ax.set_title("Post-processing calibration: v2 only")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}",
                ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    for variant, label, color in [
        ("wf_temperature", "Temperature", "#60a5fa"),
        ("wf_draw", "Draw multiplier", "#38a169"),
        ("wf_temperature_draw", "Temp + draw", "#9f7aea"),
    ]:
        sub = by_season[by_season["variant"] == variant]
        ax.plot(
            sub["season"],
            sub["delta_raw_minus_calibrated"],
            marker="o",
            linewidth=1.8,
            label=label,
            color=color,
        )
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    ax.set_ylabel("Delta RPS vs raw v2\n(positive = calibrated better)")
    ax.set_xlabel("Season")
    ax.set_title("Walk-forward calibration effect by season")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper left", ncol=3)

    fig.suptitle(
        "Calibration is learned only from previous seasons; paper and bookmaker are unchanged.",
        fontsize=11,
    )
    path = run_dir / "18_postprocess_calibration_overview.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def run(run_dir: Path) -> dict[str, Path]:
    df = _load_match_data(run_dir)
    probs = _probs_from_df(df, "v2")
    outcomes = df["actual_result_idx"].to_numpy(dtype=int)
    raw_rps = _rps_values(probs, outcomes)
    book_rps = df["rps_book"].to_numpy(dtype=float)
    paper_rps = (
        df["rps_paper"].to_numpy(dtype=float)
        if "rps_paper" in df.columns
        else None
    )

    calibrated_outputs = {}
    by_season_frames = []
    summary_rows = [
        {
            "variant": "raw_v2",
            "n": len(raw_rps),
            "mean_rps": float(raw_rps.mean()),
            "delta_raw_minus_variant": 0.0,
            "rel_improvement_vs_raw_pct": 0.0,
            "ci_lo_raw_minus_variant": 0.0,
            "ci_hi_raw_minus_variant": 0.0,
            "wilcoxon_p_vs_raw": np.nan,
            "delta_book_minus_variant": float(book_rps.mean() - raw_rps.mean()),
            "rel_vs_book_pct": float((book_rps.mean() - raw_rps.mean()) / book_rps.mean() * 100.0),
            "temperature": 1.0,
            "draw_multiplier": 1.0,
            "note": "Unchanged v2 probabilities.",
        },
        {
            "variant": "bookmaker",
            "n": len(book_rps),
            "mean_rps": float(book_rps.mean()),
            "delta_raw_minus_variant": float(raw_rps.mean() - book_rps.mean()),
            "rel_improvement_vs_raw_pct": float((raw_rps.mean() - book_rps.mean()) / raw_rps.mean() * 100.0),
            "ci_lo_raw_minus_variant": np.nan,
            "ci_hi_raw_minus_variant": np.nan,
            "wilcoxon_p_vs_raw": np.nan,
            "delta_book_minus_variant": 0.0,
            "rel_vs_book_pct": 0.0,
            "temperature": np.nan,
            "draw_multiplier": np.nan,
            "note": "Fixed comparison baseline, not calibrated.",
        },
    ]
    if paper_rps is not None:
        summary_rows[0]["delta_paper_minus_variant"] = float(paper_rps.mean() - raw_rps.mean())
        summary_rows[0]["rel_vs_paper_pct"] = float((paper_rps.mean() - raw_rps.mean()) / paper_rps.mean() * 100.0)
        summary_rows[1]["delta_paper_minus_variant"] = float(paper_rps.mean() - book_rps.mean())
        summary_rows[1]["rel_vs_paper_pct"] = float((paper_rps.mean() - book_rps.mean()) / paper_rps.mean() * 100.0)
        summary_rows.append({
            "variant": "paper",
            "n": len(paper_rps),
            "mean_rps": float(paper_rps.mean()),
            "delta_raw_minus_variant": float(raw_rps.mean() - paper_rps.mean()),
            "rel_improvement_vs_raw_pct": float((raw_rps.mean() - paper_rps.mean()) / raw_rps.mean() * 100.0),
            "ci_lo_raw_minus_variant": np.nan,
            "ci_hi_raw_minus_variant": np.nan,
            "wilcoxon_p_vs_raw": np.nan,
            "delta_book_minus_variant": float(book_rps.mean() - paper_rps.mean()),
            "rel_vs_book_pct": float((book_rps.mean() - paper_rps.mean()) / book_rps.mean() * 100.0),
            "delta_paper_minus_variant": 0.0,
            "rel_vs_paper_pct": 0.0,
            "temperature": np.nan,
            "draw_multiplier": np.nan,
            "note": "Fixed paper baseline, not calibrated.",
        })

    for variant in FITTERS:
        cal_probs, by_season = _walkforward_calibration(df, probs, outcomes, variant)
        cal_rps = _rps_values(cal_probs, outcomes)
        calibrated_outputs[variant] = (cal_probs, cal_rps)
        by_season_frames.append(by_season)
        summary_rows.append(_summary_row(
            f"wf_{variant}",
            cal_rps,
            raw_rps,
            book_rps,
            paper_rps,
            temperature=float(np.nanmedian(by_season["temperature"])),
            draw_multiplier=float(np.nanmedian(by_season["draw_multiplier"])),
            note="Fair walk-forward calibration; first season uses identity.",
        ))

        oracle_params = FITTERS[variant](probs, outcomes)
        oracle_probs = _apply_variant(probs, variant, oracle_params)
        oracle_rps = _rps_values(oracle_probs, outcomes)
        summary_rows.append(_summary_row(
            f"oracle_{variant}",
            oracle_rps,
            raw_rps,
            book_rps,
            paper_rps,
            temperature=oracle_params["temperature"],
            draw_multiplier=oracle_params["draw_multiplier"],
            note="Optimistic upper bound: fitted and evaluated on all matches.",
        ))

    summary = pd.DataFrame(summary_rows)
    by_season_out = pd.concat(by_season_frames, ignore_index=True)

    predictions = df.copy()
    predictions["rps_v2_recomputed"] = raw_rps
    for variant, (cal_probs, cal_rps) in calibrated_outputs.items():
        suffix = variant.replace("temperature", "temp")
        predictions[f"p_home_cal_{suffix}"] = cal_probs[:, 0]
        predictions[f"p_draw_cal_{suffix}"] = cal_probs[:, 1]
        predictions[f"p_away_cal_{suffix}"] = cal_probs[:, 2]
        predictions[f"rps_cal_{suffix}"] = cal_rps

    summary_path = run_dir / "postprocess_calibration_summary.csv"
    by_season_path = run_dir / "postprocess_calibration_by_season.csv"
    predictions_path = run_dir / "postprocess_calibrated_predictions.csv"
    summary.to_csv(summary_path, index=False)
    by_season_out.to_csv(by_season_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    plot_path = _plot_results(run_dir, summary, by_season_out)

    return {
        "summary": summary_path,
        "by_season": by_season_path,
        "predictions": predictions_path,
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
        "variant",
        "mean_rps",
        "delta_raw_minus_variant",
        "rel_improvement_vs_raw_pct",
        "delta_book_minus_variant",
    ]
    if "delta_paper_minus_variant" in summary:
        cols.append("delta_paper_minus_variant")
    print("\nSummary:")
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
