"""
data.py
=======
Lädt historische Bundesligadaten von football-data.co.uk.

Änderungen v2:
  * Optional zusätzliche Spalten: Schussdaten (HS, AS, HST, AST)
    und Buchmacher-Schlussquoten (PSCH, PSCD, PSCA, B365CH, ...).
  * ``load_bundesliga(..., with_extras=True)`` behält diese Spalten,
    damit xG-Berechnung (xg.py) und Buchmacher-Vergleich (evaluation.py)
    darauf zugreifen können.

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

# football-data.co.uk hat erst ab Saison 1993/94 Daten für die Bundesliga (D1).
# Frühere Anfragen führen zu 404 — wir überspringen sie still.
FIRST_AVAILABLE_YEAR = 1993

# Basisspalten — auf jeden Fall behalten
_BASE_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# Zusatzspalten — Schussdaten + Buchmacher-Schlussquoten + Eröffnungsquoten
_EXTRA_COLS = [
    # Schüsse
    "HS", "AS", "HST", "AST",
    # Buchmacher: Schlussquoten (closing) — bevorzugt für RPS-Vergleich
    "PSCH", "PSCD", "PSCA",     # Pinnacle Closing
    "B365CH", "B365CD", "B365CA",  # Bet365 Closing
    "AvgCH", "AvgCD", "AvgCA",  # Markt-Durchschnitt Closing
    # Buchmacher: Eröffnungsquoten — Fallback für ältere Saisons
    "PSH", "PSD", "PSA",
    "B365H", "B365D", "B365A",
]


def _season_code(year: int) -> str:
    """2000 -> '0001'  (Saison 2000/01)"""
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def _season_label(year: int) -> str:
    """2000 -> '2000/01'"""
    return f"{year}/{(year + 1) % 100:02d}"


def _read_csv_lenient(text: str) -> pd.DataFrame:
    """
    Liest CSV; bei kaputten Zeilen (z.B. eingebettete Kommas in
    den Saisons 2002/03 und 2003/04) werden diese Zeilen übersprungen
    statt einen kompletten Parse-Fehler zu werfen.
    """
    try:
        return pd.read_csv(StringIO(text))
    except Exception:
        return pd.read_csv(StringIO(text), on_bad_lines="skip",
                           engine="python")


def _clean_d1_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtert eine football-data.co.uk-CSV auf echte Bundesliga-1-Spiele.

    Frühe Saisons (z.B. 1993/94, 1996/97, 1999/00) enthalten Trailing-
    Junk-Zeilen mit leerem Div oder leeren Pflichtspalten. Hier behalten
    wir nur Zeilen mit Div == "D1" und vollständigen Pflichtdaten.
    """
    if "Div" in df.columns:
        df = df[df["Div"].astype(str).str.strip() == "D1"]
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    have = [c for c in required if c in df.columns]
    if have:
        df = df.dropna(subset=have)
    return df.reset_index(drop=True)


def download_season(year: int, force: bool = False) -> pd.DataFrame | None:
    """Lädt eine Saison; nutzt Cache wenn vorhanden.

    Gibt None zurück für:
      - Saisons vor FIRST_AVAILABLE_YEAR (1993) — existieren nicht
      - Echte HTTP-Fehler (404, Timeout, ...)
    """
    if year < FIRST_AVAILABLE_YEAR:
        print(f"  Überspringe Saison {_season_label(year)} "
              f"(football-data.co.uk hat erst ab {FIRST_AVAILABLE_YEAR}/94 Daten)")
        return None

    code = _season_code(year)
    cache_file = CACHE_DIR / f"D1_{code}.csv"

    if cache_file.exists() and not force:
        text = cache_file.read_text(encoding="utf-8", errors="replace")
        return _clean_d1_rows(_read_csv_lenient(text))

    url = f"{BASE_URL}/{code}/D1.csv"
    print(f"  Lade Saison {_season_label(year)} ... ", end="", flush=True)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = _clean_d1_rows(_read_csv_lenient(resp.text))
        # Cache speichern (auch wenn lenient eingelesen wurde)
        cache_file.write_text(resp.text, encoding="utf-8")
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


def load_bundesliga(start_year: int = 1993, end_year: int = 2026,
                    fallback_synthetic: bool = False,
                    with_extras: bool = True) -> pd.DataFrame:
    """
    Lädt mehrere Bundesligasaisons und vereinheitlicht sie.

    Args:
        start_year: Erste Saison (inklusiv). Default: 1993 — erste auf
            football-data.co.uk verfügbare Saison.
        end_year: Letzte Saison (EXKLUSIV).
        fallback_synthetic: Wenn True, werden für nicht ladbare Saisons
            synthetische Daten generiert. Default: False, weil synthetische
            Saisons die historischen Analysen (Heimvorteil über Zeit, xG-
            Kalibrierung) verzerren würden. Saisons vor 1993 oder mit
            kaputten CSVs werden still übersprungen.
        with_extras: Wenn True, werden Schussdaten und Buchmacherquoten
            zusätzlich behalten (siehe _EXTRA_COLS).

    Returns:
        DataFrame mit Spalten:
            Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, Season
            [+ HS, AS, HST, AST, PSCH/D/A, B365CH/D/A, AvgCH/D/A, ...]
    """
    print(f"Lade Bundesligadaten {start_year}-{end_year}...")
    all_dfs = []
    keep_cols = _BASE_COLS + (_EXTRA_COLS if with_extras else [])
    n_skipped = 0
    n_synthetic = 0

    for year in range(start_year, end_year):
        df = download_season(year)
        if df is None:
            if fallback_synthetic:
                print(f"  -> Verwende synthetische Daten für {_season_label(year)}")
                df = _generate_synthetic_season(year)
                n_synthetic += 1
            else:
                n_skipped += 1
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

        # Numerische Konversion der Extra-Spalten (silent NaN bei Fehler)
        for col in _EXTRA_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Season"] = _season_label(year)
        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("Keine Daten geladen!")

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values("Date").reset_index(drop=True)
    extra = ""
    if n_skipped:
        extra += f", {n_skipped} übersprungen"
    if n_synthetic:
        extra += f", {n_synthetic} synthetisch"
    print(f"Gesamt: {len(result)} Spiele aus "
          f"{result['Season'].nunique()} Saisons{extra}")
    return result


# ─────────────────────────────────────────────────────────────────────
# Buchmacherquoten extrahieren — bevorzugt Schluss-, sonst Eröffnungsquoten
# ─────────────────────────────────────────────────────────────────────

_ODDS_PRIORITY = [
    ("PSCH", "PSCD", "PSCA", "Pinnacle Closing"),
    ("AvgCH", "AvgCD", "AvgCA", "Markt-Avg Closing"),
    ("B365CH", "B365CD", "B365CA", "Bet365 Closing"),
    ("PSH", "PSD", "PSA", "Pinnacle"),
    ("B365H", "B365D", "B365A", "Bet365"),
]


def extract_bookmaker_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wählt pro Zeile die beste verfügbare Quotenquelle und konvertiert
    Quoten zu vergleichbaren Wahrscheinlichkeiten (Overround entfernt).

    Fügt Spalten ``bm_pH``, ``bm_pD``, ``bm_pA``, ``bm_source`` hinzu.
    Zeilen ohne brauchbare Quoten bleiben NaN.
    """
    out = df.copy()
    n = len(out)
    pH = np.full(n, np.nan)
    pD = np.full(n, np.nan)
    pA = np.full(n, np.nan)
    src = np.full(n, "", dtype=object)

    for h_col, d_col, a_col, name in _ODDS_PRIORITY:
        if not all(c in out.columns for c in (h_col, d_col, a_col)):
            continue
        # Nur Zeilen, die noch keine Quoten haben
        mask = np.isnan(pH) & out[h_col].notna() & out[d_col].notna() & out[a_col].notna()
        if not mask.any():
            continue
        h_odd = out.loc[mask, h_col].astype(float).values
        d_odd = out.loc[mask, d_col].astype(float).values
        a_odd = out.loc[mask, a_col].astype(float).values
        inv_h = 1.0 / h_odd
        inv_d = 1.0 / d_odd
        inv_a = 1.0 / a_odd
        total = inv_h + inv_d + inv_a
        pH[mask] = inv_h / total
        pD[mask] = inv_d / total
        pA[mask] = inv_a / total
        src[mask] = name

    out["bm_pH"] = pH
    out["bm_pD"] = pD
    out["bm_pA"] = pA
    out["bm_source"] = src
    return out


if __name__ == "__main__":
    df = load_bundesliga(2020, 2024)
    print(df.head())
    print()
    print("Mit Quoten:")
    df = extract_bookmaker_probs(df)
    print(df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
              "bm_pH", "bm_pD", "bm_pA", "bm_source"]].head())
