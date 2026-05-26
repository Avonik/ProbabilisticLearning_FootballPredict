"""
data.py
=======
Lädt historische Bundesligadaten von football-data.co.uk.
Cached die Daten lokal, damit man nicht ständig neu downloaden muss.

Saisoncode-Konvention: Saison 2000/01 -> "0001", 2023/24 -> "2324", usw.
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    print("Bitte installieren: pip install requests")
    sys.exit(1)


CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

BASE_URL = "https://www.football-data.co.uk/mmz4281"


def _season_code(year: int) -> str:
    """2000 -> '0001'  (Saison 2000/01)"""
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def _season_label(year: int) -> str:
    """2000 -> '2000/01'"""
    return f"{year}/{(year + 1) % 100:02d}"


def download_season(year: int, force: bool = False) -> pd.DataFrame | None:
    """Lädt eine Saison; nutzt Cache wenn vorhanden."""
    code = _season_code(year)
    cache_file = CACHE_DIR / f"D1_{code}.csv"

    if cache_file.exists() and not force:
        return pd.read_csv(cache_file)

    url = f"{BASE_URL}/{code}/D1.csv"
    print(f"  Lade Saison {_season_label(year)} ... ", end="", flush=True)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.to_csv(cache_file, index=False)
        print(f"OK ({len(df)} Spiele)")
        return df
    except Exception as e:
        print(f"FEHLER: {e}")
        return None


def _generate_synthetic_season(year: int, n_teams: int = 18) -> pd.DataFrame:
    """Fallback: synthetische Daten, falls Download fehlschlägt."""
    rng = np.random.default_rng(year)
    teams = [f"Team{i:02d}" for i in range(n_teams)]
    true_attack = rng.normal(0, 0.3, n_teams)
    true_defense = rng.normal(0, 0.3, n_teams)

    rows = []
    start_date = pd.Timestamp(f"{year}-08-15")
    for matchday in range(2 * (n_teams - 1)):
        date = start_date + pd.Timedelta(days=7 * matchday)
        order = rng.permutation(n_teams)
        for i in range(0, n_teams, 2):
            h, a = order[i], order[i + 1]
            lam_h = np.exp(0.4 + true_attack[h] - true_defense[a])
            lam_a = np.exp(0.1 + true_attack[a] - true_defense[h])
            hg = rng.poisson(lam_h)
            ag = rng.poisson(lam_a)
            rows.append({
                "Date": date, "HomeTeam": teams[h], "AwayTeam": teams[a],
                "FTHG": hg, "FTAG": ag,
                "FTR": "H" if hg > ag else ("A" if ag > hg else "D"),
            })
    return pd.DataFrame(rows)


def load_bundesliga(start_year: int = 2000, end_year: int = 2026,
                    fallback_synthetic: bool = True) -> pd.DataFrame:
    """
    Lädt mehrere Bundesligasaisons und vereinheitlicht sie.

    Returns DataFrame mit Spalten:
        Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, Season
    """
    print(f"Lade Bundesligadaten {start_year}-{end_year}...")
    all_dfs = []
    keep_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

    for year in range(start_year, end_year):
        df = download_season(year)
        if df is None:
            if fallback_synthetic:
                print(f"  -> Verwende synthetische Daten für {_season_label(year)}")
                df = _generate_synthetic_season(year)
            else:
                continue
        else:
            present = [c for c in keep_cols if c in df.columns]
            df = df[present].copy()
            df["Date"] = pd.to_datetime(df["Date"], format="mixed",
                                       dayfirst=True, errors="coerce")
            df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])

        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce").astype("Int64")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["FTHG", "FTAG"]).copy()
        df["FTHG"] = df["FTHG"].astype(int)
        df["FTAG"] = df["FTAG"].astype(int)
        df["Season"] = _season_label(year)
        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("Keine Daten geladen!")

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values("Date").reset_index(drop=True)
    print(f"Gesamt: {len(result)} Spiele aus {result['Season'].nunique()} Saisons")
    return result


if __name__ == "__main__":
    df = load_bundesliga(2020, 2024)
    print(df.head())
    print(df.tail())
