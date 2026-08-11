from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

V2_DIR = Path(__file__).resolve().parents[1]
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from model import build_league, compute_lambdas, predict_outcome_probs
from mcmc import posterior_home_advantage, run_mcmc
from historical_home import combine_season_effects


class TeamHomeAdvantageTests(unittest.TestCase):
    def test_symmetric_effect_changes_goal_ratio_not_total_log_level(self) -> None:
        base_h, base_a = compute_lambdas(0.1, -0.1, 0.0, 0.05, 0.4, 0.1)
        plus_h, plus_a = compute_lambdas(
            0.1, -0.1, 0.0, 0.05, 0.4, 0.1, home_advantage=0.2,
        )
        self.assertAlmostEqual(np.log(plus_h / base_h), 0.1, places=12)
        self.assertAlmostEqual(np.log(plus_a / base_a), -0.1, places=12)
        self.assertAlmostEqual(plus_h * plus_a, base_h * base_a, places=12)

    def test_positive_effect_increases_home_win_probability(self) -> None:
        base = predict_outcome_probs(0, 0, 0, 0, 0.3, 0.1)
        boosted = predict_outcome_probs(
            0, 0, 0, 0, 0.3, 0.1, home_advantage=0.3,
        )
        self.assertGreater(boosted[0], base[0])
        self.assertLess(boosted[2], base[2])
        self.assertAlmostEqual(sum(boosted), 1.0, places=12)

    def test_sampler_preserves_centering_and_toggle(self) -> None:
        teams = ["A", "B", "C", "D"]
        rows = []
        date = pd.Timestamp("2024-08-01")
        for repeat in range(2):
            for h in teams:
                for a in teams:
                    if h == a:
                        continue
                    rows.append({
                        "Season": "2024/25",
                        "Date": date,
                        "HomeTeam": h,
                        "AwayTeam": a,
                        "FTHG": 3 if h == "A" else 1,
                        "FTAG": 0 if h == "A" else 1,
                    })
                    date += pd.Timedelta(days=1)
        df = pd.DataFrame(rows)

        enabled = build_league(df, use_team_home_advantage=True)
        samples = run_mcmc(
            enabled, n_iter=180, burnin=60, thin=10,
            proposal_sd=0.05, seed=11, verbose=False,
        )
        effects = np.stack(samples["home_advantage"])
        np.testing.assert_allclose(effects.sum(axis=1), 0.0, atol=1e-12)
        self.assertGreater(float(np.abs(effects).max()), 0.0)
        self.assertEqual(posterior_home_advantage(samples).shape, (4,))

        disabled = build_league(df, use_team_home_advantage=False)
        disabled_samples = run_mcmc(
            disabled, n_iter=80, burnin=20, thin=10,
            proposal_sd=0.05, seed=11, verbose=False,
        )
        np.testing.assert_array_equal(
            np.stack(disabled_samples["home_advantage"]), 0.0,
        )

    def test_historical_prior_mapping_and_promoted_team(self) -> None:
        df = pd.DataFrame([
            {"Season": "2024/25", "Date": "2024-08-01", "HomeTeam": "A",
             "AwayTeam": "B", "FTHG": 1, "FTAG": 0},
            {"Season": "2024/25", "Date": "2024-08-02", "HomeTeam": "C",
             "AwayTeam": "A", "FTHG": 1, "FTAG": 1},
        ])
        league = build_league(
            df, use_team_home_advantage=True,
            home_adv_prior_means={"A": 0.20, "B": -0.10},
        )
        mapped = dict(zip(league.teams, league.home_adv_prior_mean))
        self.assertAlmostEqual(mapped["A"], 0.15)
        self.assertAlmostEqual(mapped["B"], -0.15)
        self.assertEqual(mapped["C"], 0.0)
        self.assertAlmostEqual(float(league.home_adv_prior_mean.sum()), 0.0)

    def test_recency_weighting_and_missing_history(self) -> None:
        combined = combine_season_effects(
            [{"A": 0.30, "B": -0.30}, {"A": 0.10, "B": -0.10}],
            ["A", "B", "Promoted"], decay=0.5,
        )
        self.assertAlmostEqual(combined["A"], (0.5 * 0.30 + 0.10) / 1.5)
        self.assertAlmostEqual(combined["B"], -(0.5 * 0.30 + 0.10) / 1.5)
        self.assertEqual(combined["Promoted"], 0.0)

    def test_mcmc_seed_is_reproducible(self) -> None:
        df = pd.DataFrame([
            {"Season": "2024/25", "Date": f"2024-08-{day:02d}",
             "HomeTeam": "A" if day % 2 else "B",
             "AwayTeam": "B" if day % 2 else "A",
             "FTHG": day % 3, "FTAG": (day + 1) % 3}
            for day in range(1, 9)
        ])
        league = build_league(df, use_team_home_advantage=True)
        first = run_mcmc(
            league, n_iter=80, burnin=20, thin=10,
            proposal_sd=0.05, seed=99, verbose=False,
        )
        second = run_mcmc(
            league, n_iter=80, burnin=20, thin=10,
            proposal_sd=0.05, seed=99, verbose=False,
        )
        np.testing.assert_array_equal(
            np.stack(first["home_advantage"]),
            np.stack(second["home_advantage"]),
        )


if __name__ == "__main__":
    unittest.main()
