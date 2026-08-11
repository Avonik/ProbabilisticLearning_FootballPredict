"""Medium-sized paired ablation for the historical team-home prior.

Larger than ``quick_test_historical_home.py`` but deliberately much cheaper
than the ten-season production study: three seasons, two chains, reduced MCMC
budgets. Results are written to a timestamped output directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import backtest as bt
from data import load_bundesliga
from evaluation import outcome_index, rps_one
from market_value import team_market_values
from mcmc import warmup_jit
from real_xg import add_real_xg_columns
from xg import add_xg_columns, fit_xg_weights


SEASONS = ["2022/23", "2023/24", "2024/25"]
HOLDOUT_FRAC = 0.30
N_CHAINS = 2
BASE_ITER = 6_000
BASE_BURNIN = 1_500
WARM_ITER = 3_000
WARM_BURNIN = 500
THIN = 10
HISTORY_ITER = 6_000
HISTORY_BURNIN = 1_500
SEED = 2026
BOOTSTRAP_N = 20_000


def _scored_rows(wf: dict, variant: str, season: str) -> pd.DataFrame:
    df = wf["df_season"]
    probs = np.asarray(wf["probs_model"], dtype=float)
    idx = np.where((np.arange(len(df)) >= wf["cutoff"])
                   & np.all(np.isfinite(probs), axis=1))[0]
    outcomes = np.array([
        outcome_index(int(df.loc[i, "FTHG"]), int(df.loc[i, "FTAG"]))
        for i in idx
    ], dtype=int)
    return pd.DataFrame({
        "season": season,
        "date": pd.to_datetime(df.loc[idx, "Date"]).dt.strftime("%Y-%m-%d"),
        "match_id": df.loc[idx, "MatchID"].astype(str).to_numpy(),
        "variant": variant,
        "outcome": outcomes,
        "p_home": probs[idx, 0],
        "p_draw": probs[idx, 1],
        "p_away": probs[idx, 2],
        "rps": [rps_one(probs[i], outcomes[j]) for j, i in enumerate(idx)],
    })


def _matchday_block_ci(paired: pd.DataFrame) -> tuple[float, float]:
    """Bootstrap whole matchdays, retaining within-matchday dependence."""
    blocks = [group["delta_rps"].to_numpy()
              for _, group in paired.groupby(["season", "date"], sort=False)]
    rng = np.random.default_rng(SEED)
    boot = np.empty(BOOTSTRAP_N)
    for b in range(BOOTSTRAP_N):
        chosen = rng.integers(0, len(blocks), len(blocks))
        values = np.concatenate([blocks[i] for i in chosen])
        boot[b] = values.mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    output_dir = bt.timestamped_output_dir("medium_historical_home")
    print(f"Output: {output_dir.resolve()}")
    print(f"Seasons: {SEASONS}; holdout={HOLDOUT_FRAC:.0%}; "
          f"chains={N_CHAINS}; MCMC={BASE_ITER}/{WARM_ITER}")

    df = load_bundesliga(2014, 2025, with_extras=True)
    earliest = min(SEASONS)
    xg_train = df[df["Season"] < earliest]
    beta_off, beta_on = fit_xg_weights(xg_train, force=True, cache=False)
    df = add_xg_columns(df, beta_off, beta_on)
    df = add_real_xg_columns(df)
    warmup_jit()

    frames: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    for season in SEASONS:
        values = team_market_values(season, csv_path=bt.MARKET_VALUE_CSV)
        common = dict(
            use_xg=bt.USE_XG, tau=bt.DEFAULT_TAU,
            gamma=bt.DEFAULT_GAMMA, eps=bt.DEFAULT_EPSILON,
            holdout_frac=HOLDOUT_FRAC, verbose=True,
            base_iter=BASE_ITER, base_burnin=BASE_BURNIN,
            warm_iter=WARM_ITER, warm_burnin=WARM_BURNIN, thin=THIN,
            proposal_sd=bt.PROPOSAL_SD, seed=SEED, n_chains=N_CHAINS,
            jitter_sd=bt.CHAIN_JITTER_SD,
            continuous_xg=bt.USE_CONTINUOUS_XG, phi=bt.PHI,
            market_values=values, market_kappa=bt.MARKET_PRIOR_KAPPA,
            use_team_home_advantage=True,
            home_adv_prior_sd=bt.TEAM_HOME_ADV_PRIOR_SD,
            home_adv_history_seasons=3,
            home_adv_history_decay=bt.HOME_ADV_HISTORY_DECAY,
            home_adv_history_iter=HISTORY_ITER,
            home_adv_history_burnin=HISTORY_BURNIN,
            home_adv_history_thin=THIN,
        )

        for variant, use_history in (("current_only", False),
                                     ("historical", True)):
            print(f"\n=== {season}: {variant} ===")
            wf = bt.walkforward_predictions(
                df, season, use_historical_home_prior=use_history, **common,
            )
            frames.append(_scored_rows(wf, variant, season))
            diagnostics.append({
                "season": season, "variant": variant,
                "rhat_max": wf["rhat_max"],
                "historical_seasons": ",".join(
                    wf["historical_home_info"].get("seasons", [])),
            })

    predictions = pd.concat(frames, ignore_index=True)
    current = predictions[predictions["variant"] == "current_only"]
    historical = predictions[predictions["variant"] == "historical"]
    paired = current.merge(
        historical, on=["season", "date", "match_id", "outcome"],
        suffixes=("_current", "_historical"), validate="one_to_one",
    )
    paired["delta_rps"] = paired["rps_current"] - paired["rps_historical"]
    paired["mean_abs_probability_shift"] = np.mean(np.abs(
        paired[["p_home_current", "p_draw_current", "p_away_current"]].to_numpy()
        - paired[["p_home_historical", "p_draw_historical",
                  "p_away_historical"]].to_numpy()
    ), axis=1)

    ci_lo, ci_hi = _matchday_block_ci(paired)
    try:
        _, p_value = wilcoxon(
            paired["rps_current"], paired["rps_historical"],
            alternative="two-sided",
        )
    except ValueError:
        p_value = float("nan")

    by_season = (paired.groupby("season", sort=False)
                 .agg(n=("match_id", "size"),
                      rps_current=("rps_current", "mean"),
                      rps_historical=("rps_historical", "mean"),
                      delta_rps=("delta_rps", "mean"),
                      mean_abs_probability_shift=(
                          "mean_abs_probability_shift", "mean"))
                 .reset_index())
    summary = {
        "seasons": SEASONS,
        "n": int(len(paired)),
        "rps_current": float(paired["rps_current"].mean()),
        "rps_historical": float(paired["rps_historical"].mean()),
        "delta_rps_current_minus_historical": float(paired["delta_rps"].mean()),
        "matchday_block_bootstrap_ci95": [ci_lo, ci_hi],
        "wilcoxon_p": float(p_value),
        "fraction_historical_better": float((paired["delta_rps"] > 0).mean()),
        "mean_abs_probability_shift": float(
            paired["mean_abs_probability_shift"].mean()),
        "config": {
            "holdout_frac": HOLDOUT_FRAC, "n_chains": N_CHAINS,
            "base_iter": BASE_ITER, "warm_iter": WARM_ITER,
            "history_iter": HISTORY_ITER, "history_seasons": 3,
            "history_decay": bt.HOME_ADV_HISTORY_DECAY,
        },
    }

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    paired.to_csv(output_dir / "paired_results.csv", index=False)
    by_season.to_csv(output_dir / "by_season.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir / "diagnostics.csv", index=False)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("MEDIUM TEST RESULT (positive delta = historical prior better)")
    print("=" * 72)
    print(by_season.to_string(index=False))
    print(f"\nN={summary['n']}")
    print(f"RPS current-only = {summary['rps_current']:.6f}")
    print(f"RPS historical   = {summary['rps_historical']:.6f}")
    print(f"Delta            = {summary['delta_rps_current_minus_historical']:+.6f}")
    print(f"95% block-CI     = [{ci_lo:+.6f}, {ci_hi:+.6f}]")
    print(f"Wilcoxon p       = {p_value:.6f}")
    print(f"Historical better per match = "
          f"{summary['fraction_historical_better']:.1%}")
    print(f"Mean abs probability shift  = "
          f"{summary['mean_abs_probability_shift']:.2%}")
    print(f"Output: {output_dir.resolve()}")
    print("NOTE: reduced MCMC budget; stronger than smoke test, not final evidence.")


if __name__ == "__main__":
    main()
