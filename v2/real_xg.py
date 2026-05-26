"""
real_xg.py
==========
Echte Match-xG (Understat via ``soccerdata``) für die Bundesliga.

Hintergrund
-----------
``xg.py`` baut einen groben xG-*Proxy* aus den Schusszählern von
football-data.co.uk (β·Schüsse). Korrelation zu echtem xG nur ~0.85 — und
der φ-Sweep im Walk-Forward zeigt: das Modell darf dem Proxy nur begrenzt
vertrauen (φ-Optimum ~5), weil es sonst dessen Rauschen überfittet.

Echtes xG (Understat, StatsBomb-nah) ist auf Schussqualität (Position,
Winkel, …) kalibriert und ab Saison **2014/15** verfügbar. Dieses Modul holt
es über ``soccerdata`` (das selbst cached: ``~/soccerdata/data/Understat``)
und legt es per (Saison-Startjahr, Heim, Auswärts) auf den football-data-
DataFrame. Spiele ohne echtes xG (vor 2014/15 oder nicht gematcht) behalten
den Proxy/Tore-Fallback aus ``xg.py``.

Kein Leck: echtes xG wird *gemessen* (kein gefitteter Koeffizient), und im
Walk-Forward fließt das xG eines Spiels erst in die Stärken künftiger
Spieltage ein — nie in die Vorhersage des Spiels selbst.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNDERSTAT_FIRST_YEAR = 2014   # erste Saison mit Understat-xG (2014/15)
UNDERSTAT_LEAGUE = "GER-Bundesliga"

# football-data.co.uk-Name  →  Understat-Name (geprüft über 2014/15–2025/26).
_FD_TO_US: dict[str, str] = {
    "Augsburg":            "Augsburg",
    "Bayern Munich":       "Bayern Munich",
    "Bielefeld":           "Arminia Bielefeld",
    "Bochum":              "Bochum",
    "Darmstadt":           "Darmstadt",
    "Dortmund":            "Borussia Dortmund",
    "Ein Frankfurt":       "Eintracht Frankfurt",
    "FC Koln":             "FC Cologne",
    "Fortuna Dusseldorf":  "Fortuna Duesseldorf",
    "Freiburg":            "Freiburg",
    "Greuther Furth":      "Greuther Fuerth",
    "Hamburg":             "Hamburger SV",
    "Hannover":            "Hannover 96",
    "Heidenheim":          "FC Heidenheim",
    "Hertha":              "Hertha Berlin",
    "Hoffenheim":          "Hoffenheim",
    "Holstein Kiel":       "Holstein Kiel",
    "Ingolstadt":          "Ingolstadt",
    "Leverkusen":          "Bayer Leverkusen",
    "M'gladbach":          "Borussia M.Gladbach",
    "Mainz":               "Mainz 05",
    "Nurnberg":            "Nuernberg",
    "Paderborn":           "Paderborn",
    "RB Leipzig":          "RasenBallsport Leipzig",
    "Schalke 04":          "Schalke 04",
    "St Pauli":            "St. Pauli",
    "Stuttgart":           "VfB Stuttgart",
    "Union Berlin":        "Union Berlin",
    "Werder Bremen":       "Werder Bremen",
    "Wolfsburg":           "Wolfsburg",
}


def _season_start_year(ts) -> int:
    """Saison-Startjahr aus einem Spieldatum (Bundesliga läuft Aug–Mai)."""
    ts = pd.Timestamp(ts)
    return ts.year if ts.month >= 7 else ts.year - 1


def _soccerdata_season_tag(start_year: int) -> str:
    """2024 → '2425' (eindeutiges soccerdata-Format; vermeidet die
    Mehrdeutigkeit von 4-stelligen Startjahren wie '2021')."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def fetch_understat_xg(start_years: list[int], verbose: bool = True) -> pd.DataFrame:
    """Holt Understat-Match-xG für die gegebenen Saison-Startjahre.

    Returns: DataFrame mit Spalten
        start_year, home_team, away_team, home_xg, away_xg   (nur gespielte Spiele)
    """
    import logging
    import soccerdata as sd
    logging.getLogger("soccerdata").setLevel(logging.WARNING)

    seasons = [_soccerdata_season_tag(y) for y in start_years]
    if verbose:
        print(f"  Hole echte xG (Understat) für Saisons {seasons} ...")
    us = sd.Understat(leagues=UNDERSTAT_LEAGUE, seasons=seasons)
    sched = us.read_schedule().reset_index()

    sched = sched[sched["is_result"] == True].copy()       # noqa: E712
    sched["home_xg"] = pd.to_numeric(sched["home_xg"], errors="coerce")
    sched["away_xg"] = pd.to_numeric(sched["away_xg"], errors="coerce")
    sched["start_year"] = sched["date"].apply(_season_start_year)
    out = (sched[["start_year", "home_team", "away_team", "home_xg", "away_xg"]]
           .dropna(subset=["home_xg", "away_xg"])
           .drop_duplicates(["start_year", "home_team", "away_team"]))
    return out


def add_real_xg_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Überschreibt ``xG_home``/``xG_away`` mit echten Understat-xG, wo
    verfügbar (Saison ≥ 2014/15 und Team-Paar gematcht). Setzt ``has_xg=True``
    dort und ergänzt ``xg_source`` ∈ {"real","proxy","goals"}.

    Erwartet, dass ``xg.add_xg_columns`` bereits lief (Proxy-xG als Fallback).
    """
    if "xG_home" not in df.columns:
        raise ValueError("add_real_xg_columns braucht vorher xg.add_xg_columns "
                         "(Proxy-xG als Fallback).")
    out = df.copy()

    sy = out["Season"].str.slice(0, 4).astype(int).to_numpy()
    need_years = sorted({int(y) for y in sy if y >= UNDERSTAT_FIRST_YEAR})
    if not need_years:
        out["xg_source"] = np.where(out.get("has_xg", False), "proxy", "goals")
        return out

    us = fetch_understat_xg(need_years, verbose=verbose)
    lut_h = us.set_index(["start_year", "home_team", "away_team"])["home_xg"]
    lut_a = us.set_index(["start_year", "home_team", "away_team"])["away_xg"]

    h_us = out["HomeTeam"].map(_FD_TO_US).to_numpy()
    a_us = out["AwayTeam"].map(_FD_TO_US).to_numpy()
    keys = pd.MultiIndex.from_arrays([sy, h_us, a_us])
    real_h = lut_h.reindex(keys).to_numpy(dtype=float)
    real_a = lut_a.reindex(keys).to_numpy(dtype=float)
    matched = np.isfinite(real_h) & np.isfinite(real_a)

    proxy_has = out.get("has_xg", pd.Series(False, index=out.index)).to_numpy()
    out["xg_source"] = np.where(matched, "real",
                                np.where(proxy_has, "proxy", "goals"))
    out["xG_home"] = np.where(matched, real_h, out["xG_home"].to_numpy())
    out["xG_away"] = np.where(matched, real_a, out["xG_away"].to_numpy())
    out["has_xg"] = proxy_has | matched

    if verbose:
        post14 = sy >= UNDERSTAT_FIRST_YEAR
        n14 = int(post14.sum())
        nmatch = int((matched & post14).sum())
        cov = 100.0 * nmatch / max(1, n14)
        print(f"  Echte xG gematcht: {nmatch}/{n14} Spiele ab 2014/15 "
              f"({cov:.1f}% Abdeckung).")
        miss = out.loc[post14 & ~matched, ["HomeTeam", "AwayTeam"]]
        if len(miss):
            bad = sorted(set(miss["HomeTeam"]) | set(miss["AwayTeam"]))
            print(f"  WARNUNG: {len(miss)} Spiele ohne echtes xG. "
                  f"Beteiligte Teams (Mapping prüfen): {bad}")
    return out


if __name__ == "__main__":
    # Smoke: Abdeckung über alle Saisons ab 2014/15.
    from data import load_bundesliga
    from xg import fit_xg_weights, add_xg_columns
    df = load_bundesliga(1993, 2026, with_extras=True)
    bo, bn = fit_xg_weights(df, force=True, cache=False)
    df = add_xg_columns(df, bo, bn)
    df = add_real_xg_columns(df)
    print(df[df["xg_source"] == "real"]
          [["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
            "xG_home", "xG_away", "xg_source"]].head(6).to_string())
