"""Paired, season-blocked ablation test for team-specific home advantage.

Run the multi-season backtest once with ``USE_TEAM_HOME_ADVANTAGE=False`` and
once with it enabled, then call:

    python compare_home_advantage_runs.py BASELINE_RUN TEAMHOME_RUN

Arguments may be run directories or their ``multiseason_per_match_rps.csv``.
Positive delta means that the team-home model has the lower (better) RPS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _csv_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_file() else path / "multiseason_per_match_rps.csv"


def compare_runs(baseline_path: str | Path, teamhome_path: str | Path,
                 *, n_boot: int = 10_000, seed: int = 23
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(_csv_path(baseline_path))
    teamhome = pd.read_csv(_csv_path(teamhome_path))
    required = {"match_id", "season", "rps_v2"}
    for label, frame in (("baseline", baseline), ("team-home", teamhome)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} run missing columns: {', '.join(missing)}")
        if frame["match_id"].duplicated().any():
            raise ValueError(f"{label} run contains duplicate match_id values.")

    joined = baseline[["match_id", "season", "rps_v2"]].merge(
        teamhome[["match_id", "season", "rps_v2"]],
        on="match_id", how="inner", validate="one_to_one",
        suffixes=("_baseline", "_teamhome"),
    )
    if len(joined) != len(baseline) or len(joined) != len(teamhome):
        raise ValueError(
            "Runs do not contain the same MatchIDs: "
            f"baseline={len(baseline)}, team-home={len(teamhome)}, common={len(joined)}"
        )
    if not (joined["season_baseline"] == joined["season_teamhome"]).all():
        raise ValueError("Season mismatch for identical MatchIDs.")
    joined["season"] = joined.pop("season_baseline")
    joined = joined.drop(columns="season_teamhome")
    joined["delta_rps_baseline_minus_teamhome"] = (
        joined["rps_v2_baseline"] - joined["rps_v2_teamhome"]
    )

    diff = joined["delta_rps_baseline_minus_teamhome"].to_numpy(dtype=float)
    try:
        _, p_value = wilcoxon(
            joined["rps_v2_baseline"], joined["rps_v2_teamhome"],
            alternative="two-sided",
        )
    except ValueError:
        p_value = float("nan")

    seasons = joined["season"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(seasons, size=len(seasons), replace=True)
        values = [
            joined.loc[joined["season"] == season,
                       "delta_rps_baseline_minus_teamhome"].to_numpy()
            for season in sampled
        ]
        boot[b] = np.concatenate(values).mean()
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    summary = pd.DataFrame([{
        "comparison": "teamhome_vs_baseline",
        "n": len(joined),
        "seasons": len(seasons),
        "rps_baseline": joined["rps_v2_baseline"].mean(),
        "rps_teamhome": joined["rps_v2_teamhome"].mean(),
        "delta_baseline_minus_teamhome": diff.mean(),
        "ci_lo_season_block": ci_lo,
        "ci_hi_season_block": ci_hi,
        "wilcoxon_p": p_value,
        "frac_matches_teamhome_better": float((diff > 0).mean()),
    }])
    by_season = (joined.groupby("season", sort=False)
                 .agg(n=("match_id", "size"),
                      rps_baseline=("rps_v2_baseline", "mean"),
                      rps_teamhome=("rps_v2_teamhome", "mean"),
                      delta_baseline_minus_teamhome=(
                          "delta_rps_baseline_minus_teamhome", "mean"))
                 .reset_index())
    return summary, by_season


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_run")
    parser.add_argument("teamhome_run")
    args = parser.parse_args()
    summary, by_season = compare_runs(args.baseline_run, args.teamhome_run)

    output_dir = _csv_path(args.teamhome_run).parent
    summary_path = output_dir / "home_advantage_ablation_summary.csv"
    season_path = output_dir / "home_advantage_ablation_by_season.csv"
    summary.to_csv(summary_path, index=False)
    by_season.to_csv(season_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nWritten: {summary_path}\n         {season_path}")


if __name__ == "__main__":
    main()
