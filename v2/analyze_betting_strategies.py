"""Leak-aware betting-strategy analysis for a completed multi-season run.

The first six seasons are used for strategy discovery and the remaining four
as a locked chronological test. ROI always uses raw decimal odds (including
the bookmaker margin); normalized inverse odds are used only to measure the
model-market probability gap.

Price variants:
  * closing: same source priority as the model evaluation;
  * opening: Pinnacle opening, falling back to Bet365 opening;
  * opening_mean: mean available opening quote across Pinnacle and Bet365;
  * open_close_mid: geometric open/close midpoint (sensitivity only, not a
    guaranteed executable market price).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data import MATCH_ID_COL, _ODDS_PRIORITY, load_bundesliga, sort_matches
from evaluation import rps_one


OUTCOME_LABELS = np.array(["H", "D", "A"], dtype=object)
EDGE_GRID = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
GAP_GRID = (0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15)
KINDS = (
    "any", "H", "D", "A", "market_favorite", "model_favorite",
    "agree_favorite", "disagree_model_pick", "fav_odds<2",
    "mid_2-3.5", "dog>=3.5",
)
FIXED_RULES = (
    ("favorite_gap_04", "market_favorite", "gap", 0.04),
    ("model_favorite_gap_05", "model_favorite", "gap", 0.05),
    ("any_gap_10", "any", "gap", 0.10),
    ("away_gap_10", "A", "gap", 0.10),
    ("underdog_gap_10", "dog>=3.5", "gap", 0.10),
    ("draw_edge_0", "D", "edge", 0.0),
    ("draw_edge_025", "D", "edge", 0.025),
)


def _complete_triplet(df: pd.DataFrame, cols: tuple[str, str, str]) -> np.ndarray:
    if not all(col in df.columns for col in cols):
        return np.zeros(len(df), dtype=bool)
    values = df[list(cols)].to_numpy(dtype=float)
    return np.isfinite(values).all(axis=1) & (values > 1.0).all(axis=1)


def _priority_odds(
    df: pd.DataFrame,
    sources: tuple[tuple[str, str, str, str], ...],
) -> tuple[np.ndarray, np.ndarray]:
    odds = np.full((len(df), 3), np.nan)
    labels = np.full(len(df), "", dtype=object)
    for home, draw, away, label in sources:
        cols = (home, draw, away)
        mask = np.isnan(odds[:, 0]) & _complete_triplet(df, cols)
        odds[mask] = df.loc[mask, list(cols)].to_numpy(dtype=float)
        labels[mask] = label
    return odds, labels


def _mean_opening_odds(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    source_arrays = []
    source_names = []
    for cols, name in (
        (("PSH", "PSD", "PSA"), "Pinnacle"),
        (("B365H", "B365D", "B365A"), "Bet365"),
    ):
        values = np.full((len(df), 3), np.nan)
        mask = _complete_triplet(df, cols)
        values[mask] = df.loc[mask, list(cols)].to_numpy(dtype=float)
        source_arrays.append(values)
        source_names.append((mask, name))
    stacked = np.stack(source_arrays)
    with np.errstate(invalid="ignore"):
        odds = np.nanmean(stacked, axis=0)
    labels = np.full(len(df), "", dtype=object)
    for row in range(len(df)):
        used = [name for mask, name in source_names if mask[row]]
        labels[row] = "+".join(used) + " opening mean" if used else ""
    return odds, labels


def build_price_variants(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    closing, closing_source = _priority_odds(df, tuple(_ODDS_PRIORITY))
    opening, opening_source = _priority_odds(df, (
        ("PSH", "PSD", "PSA", "Pinnacle Opening"),
        ("B365H", "B365D", "B365A", "Bet365 Opening"),
    ))
    opening_mean, opening_mean_source = _mean_opening_odds(df)
    midpoint = np.sqrt(opening_mean * closing)
    midpoint_source = np.where(
        np.isfinite(midpoint).all(axis=1), "Geometric Open/Close Midpoint", "",
    )
    return {
        "closing": (closing, closing_source),
        "opening": (opening, opening_source),
        "opening_mean": (opening_mean, opening_mean_source),
        "open_close_mid": (midpoint, midpoint_source),
    }


def _long_frame(matches: pd.DataFrame, odds: np.ndarray,
                source: np.ndarray) -> pd.DataFrame:
    model = matches[["p_home_v2", "p_draw_v2", "p_away_v2"]].to_numpy(float)
    inv = 1.0 / odds
    market = inv / inv.sum(axis=1, keepdims=True)
    actual = matches["actual_result_idx"].to_numpy(int)
    market_pick = market.argmax(axis=1)
    model_pick = model.argmax(axis=1)
    parts = []
    for outcome, label in enumerate(OUTCOME_LABELS):
        part = pd.DataFrame({
            "match_id": matches["match_id"].astype(str),
            "season": matches["season"].astype(str),
            "date": matches["date"].astype(str),
            "selection": label,
            "outcome_idx": outcome,
            "p_model": model[:, outcome],
            "p_market": market[:, outcome],
            "odds": odds[:, outcome],
            "win": actual == outcome,
            "market_favorite": market_pick == outcome,
            "model_favorite": model_pick == outcome,
            "agree_favorite": ((market_pick == outcome)
                               & (model_pick == outcome)),
            "disagree_model_pick": ((model_pick == outcome)
                                    & (market_pick != outcome)),
            "odds_source": source,
        })
        part["gap"] = part["p_model"] - part["p_market"]
        part["edge"] = part["p_model"] * part["odds"] - 1.0
        part["profit"] = np.where(part["win"], part["odds"] - 1.0, -1.0)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _kind_mask(frame: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "any":
        return np.ones(len(frame), dtype=bool)
    if kind in ("H", "D", "A"):
        return frame["selection"].eq(kind).to_numpy()
    if kind in ("market_favorite", "model_favorite", "agree_favorite",
                "disagree_model_pick"):
        return frame[kind].to_numpy(bool)
    if kind == "fav_odds<2":
        return frame["odds"].lt(2.0).to_numpy()
    if kind == "mid_2-3.5":
        return (frame["odds"].ge(2.0) & frame["odds"].lt(3.5)).to_numpy()
    if kind == "dog>=3.5":
        return frame["odds"].ge(3.5).to_numpy()
    raise ValueError(f"Unknown strategy kind: {kind}")


def select_bets(frame: pd.DataFrame, kind: str, signal: str,
                threshold: float) -> pd.DataFrame:
    selected = frame[_kind_mask(frame, kind)
                     & frame[signal].ge(threshold)].copy()
    if selected.empty:
        return selected
    return (selected.sort_values(["match_id", signal], ascending=[True, False])
            .drop_duplicates("match_id"))


def _metrics(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {
            "n": 0, "roi": np.nan, "profit": 0.0, "hit_rate": np.nan,
            "avg_odds": np.nan, "profitable_seasons": 0,
        }
    season_profit = bets.groupby("season")["profit"].sum()
    return {
        "n": int(len(bets)),
        "roi": float(bets["profit"].mean()),
        "profit": float(bets["profit"].sum()),
        "hit_rate": float(bets["win"].mean()),
        "avg_odds": float(bets["odds"].mean()),
        "profitable_seasons": int((season_profit > 0).sum()),
    }


def _block_ci(bets: pd.DataFrame, seed: int, n_boot: int = 20_000) -> tuple[float, float]:
    if bets.empty:
        return np.nan, np.nan
    blocks = [group["profit"].to_numpy()
              for _, group in bets.groupby(["season", "date"], sort=False)]
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for index in range(n_boot):
        chosen = rng.integers(0, len(blocks), len(blocks))
        estimates[index] = np.concatenate([blocks[i] for i in chosen]).mean()
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def _accuracy_segments(matches: pd.DataFrame, odds: np.ndarray,
                       variant: str, dev_seasons: list[str]) -> list[dict]:
    model = matches[["p_home_v2", "p_draw_v2", "p_away_v2"]].to_numpy(float)
    inv = 1.0 / odds
    market = inv / inv.sum(axis=1, keepdims=True)
    actual = matches["actual_result_idx"].to_numpy(int)
    bins = pd.cut(market.max(axis=1), [0, .45, .55, .65, 1.01], right=False,
                  labels=["weak <45%", "45-55%", "55-65%", "strong >=65%"])
    rows = []
    for split, split_mask in (
        ("development", matches["season"].isin(dev_seasons).to_numpy()),
        ("locked_test", ~matches["season"].isin(dev_seasons).to_numpy()),
    ):
        for label in bins.categories:
            mask = split_mask & (bins == label)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            rps_model = np.mean([rps_one(model[i], actual[i]) for i in indices])
            rps_market = np.mean([rps_one(market[i], actual[i]) for i in indices])
            rows.append({
                "price_variant": variant, "split": split,
                "favorite_bin": str(label), "n": int(len(indices)),
                "rps_model": float(rps_model),
                "rps_market": float(rps_market),
                "delta_market_minus_model": float(rps_market - rps_model),
                "accuracy_model": float((model[indices].argmax(axis=1)
                                          == actual[indices]).mean()),
                "accuracy_market": float((market[indices].argmax(axis=1)
                                           == actual[indices]).mean()),
            })
    return rows


def analyze_variant(matches: pd.DataFrame, variant: str, odds: np.ndarray,
                    source: np.ndarray, dev_seasons: list[str],
                    seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
    matches = matches.loc[valid].reset_index(drop=True)
    odds = odds[valid]
    source = source[valid]
    long = _long_frame(matches, odds, source)
    development = long[long["season"].isin(dev_seasons)]
    locked_test = long[~long["season"].isin(dev_seasons)]

    grid_rows = []
    for kind in KINDS:
        for signal, thresholds in (("edge", EDGE_GRID), ("gap", GAP_GRID)):
            for threshold in thresholds:
                dev_bets = select_bets(development, kind, signal, threshold)
                if len(dev_bets) < 30:
                    continue
                test_bets = select_bets(locked_test, kind, signal, threshold)
                dev_metrics = _metrics(dev_bets)
                test_metrics = _metrics(test_bets)
                grid_rows.append({
                    "price_variant": variant, "kind": kind,
                    "signal": signal, "threshold": threshold,
                    **{f"dev_{key}": value for key, value in dev_metrics.items()},
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                })
    grid = pd.DataFrame(grid_rows)
    # Development-only ranking: shrink small samples and reward season breadth.
    grid["dev_selection_score"] = (
        grid["dev_profit"] / (grid["dev_n"] + 50)
        + 0.01 * grid["dev_profitable_seasons"]
    )

    fixed_rows = []
    for rule_name, kind, signal, threshold in FIXED_RULES:
        for split, frame in (("development", development),
                             ("locked_test", locked_test)):
            selected = select_bets(frame, kind, signal, threshold)
            row = {
                "price_variant": variant, "rule": rule_name,
                "kind": kind, "signal": signal, "threshold": threshold,
                "split": split, **_metrics(selected),
            }
            if split == "locked_test":
                row["ci95_low"], row["ci95_high"] = _block_ci(
                    selected, seed=seed,
                )
            else:
                row["ci95_low"], row["ci95_high"] = np.nan, np.nan
            fixed_rows.append(row)

    accuracy = pd.DataFrame(
        _accuracy_segments(matches, odds, variant, dev_seasons),
    )
    return grid, pd.DataFrame(fixed_rows), accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predictions", nargs="?",
        default="output/multiseason_20260811_115310/multiseason_per_match_rps.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    predictions = pd.read_csv(predictions_path)
    if not predictions["team_home_advantage"].eq(False).all():
        raise ValueError("Expected best-v2 run with team_home_advantage=False.")
    seasons = list(dict.fromkeys(predictions["season"].astype(str)))
    if len(seasons) < 8:
        raise ValueError("At least eight seasons are required for the split.")
    dev_seasons = seasons[:6]
    test_seasons = seasons[6:]

    start_year = int(seasons[0].split("/")[0])
    end_year = int(seasons[-1].split("/")[0]) + 1
    raw = sort_matches(load_bundesliga(start_year, end_year, with_extras=True))
    merged = predictions.merge(
        raw, left_on="match_id", right_on=MATCH_ID_COL,
        how="left", validate="one_to_one", suffixes=("", "_raw"),
    )
    variants = build_price_variants(merged)

    all_grid = []
    all_fixed = []
    all_accuracy = []
    coverage = {}
    for index, (variant, (odds, source)) in enumerate(variants.items()):
        valid = np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
        coverage[variant] = int(valid.sum())
        grid, fixed, accuracy = analyze_variant(
            merged, variant, odds, source, dev_seasons, args.seed + index,
        )
        all_grid.append(grid)
        all_fixed.append(fixed)
        all_accuracy.append(accuracy)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / f"betting_strategy_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    grid = pd.concat(all_grid, ignore_index=True)
    fixed = pd.concat(all_fixed, ignore_index=True)
    accuracy = pd.concat(all_accuracy, ignore_index=True)
    grid.to_csv(output_dir / "strategy_grid.csv", index=False)
    fixed.to_csv(output_dir / "fixed_strategy_summary.csv", index=False)
    accuracy.to_csv(output_dir / "favorite_accuracy_segments.csv", index=False)

    best = (grid.sort_values(
        ["price_variant", "kind", "dev_selection_score", "dev_roi"],
        ascending=[True, True, False, False],
    ).groupby(["price_variant", "kind"], as_index=False).first())
    best.to_csv(output_dir / "best_per_family.csv", index=False)
    metadata = {
        "model": "best_v2_no_teamhome",
        "predictions": str(predictions_path),
        "development_seasons": dev_seasons,
        "locked_test_seasons": test_seasons,
        "coverage": coverage,
        "roi": "flat one-unit stakes at raw decimal odds",
        "market_probability": "normalized inverse odds",
        "midpoint_warning": "open_close_mid is a sensitivity price, not an observed executable quote",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    columns = [
        "price_variant", "rule", "split", "n", "roi", "profit",
        "profitable_seasons", "avg_odds", "ci95_low", "ci95_high",
    ]
    print(fixed[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nOutput: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
