from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

V2_DIR = Path(__file__).resolve().parents[1]
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from data import MATCH_ID_COL, add_match_ids, sort_matches
import postprocess_odds_vintages as odds_post


class MatchIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matches = pd.DataFrame({
            "Season": ["2024/25", "2024/25"],
            "Date": pd.to_datetime(["2025-03-01", "2025-03-01"]),
            "HomeTeam": ["Beta", "Alpha"],
            "AwayTeam": ["Alpha", "Beta"],
            "FTHG": [1, 3],
            "FTAG": [2, 0],
        })

    def test_match_ids_and_order_do_not_depend_on_input_order(self) -> None:
        first = sort_matches(self.matches)
        second = sort_matches(self.matches.iloc[::-1].reset_index(drop=True))
        self.assertEqual(first[MATCH_ID_COL].tolist(), second[MATCH_ID_COL].tolist())
        self.assertEqual(first["HomeTeam"].tolist(), ["Alpha", "Beta"])

    def test_duplicate_natural_key_is_rejected(self) -> None:
        duplicate = pd.concat([self.matches, self.matches.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "not unique"):
            add_match_ids(duplicate)

    def test_odds_postprocess_joins_by_match_id_not_equal_odds(self) -> None:
        raw = sort_matches(self.matches)
        raw["PSCH"] = [2.0, 2.0]
        raw["PSCD"] = [3.5, 3.5]
        raw["PSCA"] = [4.0, 4.0]

        # Deliberately reverse rows while retaining the exact natural keys.
        saved = pd.DataFrame({
            "season": raw["Season"].iloc[::-1].to_numpy(),
            "match_id": raw[MATCH_ID_COL].iloc[::-1].to_numpy(),
            "home_team": raw["HomeTeam"].iloc[::-1].to_numpy(),
            "away_team": raw["AwayTeam"].iloc[::-1].to_numpy(),
            "home_goals": raw["FTHG"].iloc[::-1].to_numpy(),
            "away_goals": raw["FTAG"].iloc[::-1].to_numpy(),
            "p_home_book": [0.5, 0.5],
            "p_draw_book": [0.3, 0.3],
            "p_away_book": [0.2, 0.2],
        })
        with patch.object(odds_post, "_read_season_raw", return_value=raw):
            joined = odds_post._join_raw_matches(saved)

        self.assertEqual(joined["home_team"].tolist(), saved["home_team"].tolist())
        np.testing.assert_allclose(joined["PSCH"], [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
