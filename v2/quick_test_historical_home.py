"""Small one-season ablation for the historical team-home prior.

This deliberately uses one chain, a 10% holdout and small MCMC budgets. It is
a smoke/direction test, not a replacement for the full multi-season study.
"""

from __future__ import annotations

import numpy as np

import backtest as bt
from data import load_bundesliga
from evaluation import outcome_index, rps_one
from real_xg import add_real_xg_columns
from xg import add_xg_columns, fit_xg_weights


SEASON = "2024/25"


def _mean_rps(wf: dict) -> tuple[float, np.ndarray]:
    df = wf["df_season"]
    probs = wf["probs_model"]
    idx = np.where((np.arange(len(df)) >= wf["cutoff"])
                   & np.all(np.isfinite(probs), axis=1))[0]
    outcomes = np.array([
        outcome_index(int(df.loc[i, "FTHG"]), int(df.loc[i, "FTAG"]))
        for i in idx
    ])
    return float(np.mean([rps_one(probs[i], outcomes[j])
                          for j, i in enumerate(idx)])), idx


def main() -> None:
    df = load_bundesliga(2014, 2025, with_extras=True)
    train = df[df["Season"] < SEASON]
    beta_off, beta_on = fit_xg_weights(train, force=True, cache=False)
    df = add_xg_columns(df, beta_off, beta_on)
    df = add_real_xg_columns(df)

    common = dict(
        use_xg=bt.USE_XG, tau=bt.DEFAULT_TAU, gamma=bt.DEFAULT_GAMMA,
        eps=bt.DEFAULT_EPSILON, holdout_frac=0.10, verbose=True,
        base_iter=3_000, base_burnin=800,
        warm_iter=1_500, warm_burnin=300, thin=5,
        proposal_sd=bt.PROPOSAL_SD, seed=123, n_chains=1,
        continuous_xg=bt.USE_CONTINUOUS_XG, phi=bt.PHI,
        market_values=None, market_kappa=0.0,
        use_team_home_advantage=True,
        home_adv_prior_sd=bt.TEAM_HOME_ADV_PRIOR_SD,
        home_adv_history_seasons=3, home_adv_history_decay=0.65,
        home_adv_history_iter=3_000, home_adv_history_burnin=800,
        home_adv_history_thin=5,
    )

    print("\n=== Current-season-only team effect ===")
    current = bt.walkforward_predictions(
        df, SEASON, use_historical_home_prior=False, **common,
    )
    print("\n=== Historical prior + current-season update ===")
    historical = bt.walkforward_predictions(
        df, SEASON, use_historical_home_prior=True, **common,
    )

    rps_current, idx_current = _mean_rps(current)
    rps_history, idx_history = _mean_rps(historical)
    idx = np.intersect1d(idx_current, idx_history)
    shift = float(np.mean(np.abs(
        historical["probs_model"][idx] - current["probs_model"][idx]
    )))
    print("\n=== QUICK RESULT (not inferential) ===")
    print(f"Season: {SEASON}; scored matches: {len(idx)}")
    print(f"RPS current-only : {rps_current:.6f}")
    print(f"RPS historical   : {rps_history:.6f}")
    print(f"Delta (current - historical; positive=historical better): "
          f"{rps_current - rps_history:+.6f}")
    print(f"Mean absolute probability shift: {shift:.4%}")
    print(f"Historical seasons: {historical['historical_home_info']['seasons']}")
    print("WARNING: short one-chain run; use only as smoke/direction check.")


if __name__ == "__main__":
    main()
