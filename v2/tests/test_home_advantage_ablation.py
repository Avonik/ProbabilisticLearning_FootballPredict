from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

V2_DIR = Path(__file__).resolve().parents[1]
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from compare_home_advantage_runs import compare_runs


class HomeAdvantageAblationTests(unittest.TestCase):
    def test_comparison_is_paired_by_match_id(self) -> None:
        baseline = pd.DataFrame({
            "match_id": ["m1", "m2", "m3", "m4"],
            "season": ["s1", "s1", "s2", "s2"],
            "rps_v2": [0.30, 0.20, 0.25, 0.15],
        })
        improved = pd.DataFrame({
            "match_id": ["m4", "m2", "m1", "m3"],
            "season": ["s2", "s1", "s1", "s2"],
            "rps_v2": [0.10, 0.15, 0.20, 0.20],
        })
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_path = tmp_path / "baseline.csv"
            improved_path = tmp_path / "improved.csv"
            baseline.to_csv(base_path, index=False)
            improved.to_csv(improved_path, index=False)
            summary, by_season = compare_runs(
                base_path, improved_path, n_boot=100, seed=1,
            )

        self.assertAlmostEqual(
            summary.loc[0, "delta_baseline_minus_teamhome"], 0.0625,
        )
        self.assertEqual(by_season["n"].sum(), 4)


if __name__ == "__main__":
    unittest.main()
