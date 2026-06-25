"""
market_value.py
===============
Informativer Prior aus Team-KADERMARKTWERTEN (v2, Option A) — LEAK-FREI.

Statt die Initial-Teamstärke auf N(0, σ²) zu zentrieren, wird sie auf
κ·z(log Kaderwert) zentriert — z = pro-Liga z-standardisierter log-Kaderwert.
Weil der Stärke-Random-Walk den Mittelwert erhält, ist damit die GESAMTE
Saison-Trajektorie a-priori auf dem Kaderwert-Baseline verankert (Unsicherheit
wächst über die Saison, die Daten überschreiben sie).

LEAK-FREIE Datenquelle
----------------------
Transfermarkt-Marktwerte zu einem STICHTAG VOR Saisonbeginn (Pfad-Form, die
das historische Datum tatsächlich auf die Werte anwendet):

  .../{liga}/marktwerteverein/wettbewerb/{CODE}/stichtag/{YYYY-MM-DD}/plus/1

Vorgehen je Saison:
  1. Korrekte 18 Saison-Clubs (club_id) von der saison_id-Startseite holen.
  2. Marktwerte zum Stichtag aus L1+L2+L3 holen (ein Club kann zum Stichtag
     in einer tieferen Liga gestanden haben → alle drei abdecken).
  3. Join über club_id. So liefert die Seite zwar die *aktuelle* Liga-
     Mannschaft, aber durch den id-Join zählt nur der historische Wert der
     echten Saison-Clubs.

Der so gespeicherte Wert je Saison ist bereits der Stand VOR dem 1. Spieltag
→ kein Leak, kein Lag nötig. ``team_market_values(S)`` gibt direkt den Wert
für S zurück.

Namensabbildung Transfermarkt → football-data.co.uk über eine NORMALISIERTE
Form (ascii, ohne Ziffern/Akzente), damit Schreibvarianten robust matchen.
"""

from __future__ import annotations

import re
import time
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = "data_cache/transfermarkt_squad_values.csv"
_REQUIRED_COLS = {"Season", "ClubTM", "MarketValueEUR"}

_UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/126.0 Safari/537.36"),
       "Accept-Language": "en-US,en;q=0.9,de;q=0.8"}
_BASE = "https://www.transfermarkt.com"

# Aktuelle Wettbewerbe, in denen (ehemalige) BL-Clubs heute stehen können.
_COMPETITIONS = [("bundesliga", "L1"), ("2-bundesliga", "L2"), ("3-liga", "L3")]


@dataclass(frozen=True)
class SeasonConfig:
    season: str          # "2023/24"  (football-data-Format)
    season_id: int       # 2023
    season_start: str    # erster Spieltag
    cutoff_date: str     # Stichtag VOR dem ersten Spieltag (leak-frei)


# Stichtage liegen bewusst VOR dem jeweiligen Saisonstart (leak-frei).
SEASONS: list[SeasonConfig] = [
    SeasonConfig("2014/15", 2014, "2014-08-22", "2014-07-10"),
    SeasonConfig("2015/16", 2015, "2015-08-14", "2015-08-01"),
    SeasonConfig("2016/17", 2016, "2016-08-26", "2016-08-15"),
    SeasonConfig("2017/18", 2017, "2017-08-18", "2017-08-15"),
    SeasonConfig("2018/19", 2018, "2018-08-24", "2018-08-15"),
    SeasonConfig("2019/20", 2019, "2019-08-16", "2019-08-15"),
    SeasonConfig("2020/21", 2020, "2020-09-18", "2020-09-15"),
    SeasonConfig("2021/22", 2021, "2021-08-13", "2021-08-01"),
    SeasonConfig("2022/23", 2022, "2022-08-05", "2022-08-01"),
    SeasonConfig("2023/24", 2023, "2023-08-18", "2023-08-15"),
    SeasonConfig("2024/25", 2024, "2024-08-23", "2024-08-15"),
    SeasonConfig("2025/26", 2025, "2025-08-22", "2025-08-15"),
]


# ─────────────────────────────────────────────────────────────────────
# Namensabbildung Transfermarkt → football-data.co.uk
# ─────────────────────────────────────────────────────────────────────
def _norm(name: str) -> str:
    """Robuster Schlüssel: ascii, nur Kleinbuchstaben (ohne Ziffern/Akzente)."""
    s = unicodedata.normalize("NFKD", str(name))
    s = s.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z]", "", s.lower())


_TM_TO_FD: dict[str, str] = {
    _norm("Bayern Munich"):              "Bayern Munich",
    _norm("Bayer 04 Leverkusen"):        "Leverkusen",
    _norm("RB Leipzig"):                 "RB Leipzig",
    _norm("RasenBallsport Leipzig"):     "RB Leipzig",
    _norm("Borussia Dortmund"):          "Dortmund",
    _norm("VfB Stuttgart"):              "Stuttgart",
    _norm("Eintracht Frankfurt"):        "Ein Frankfurt",
    _norm("VfL Wolfsburg"):              "Wolfsburg",
    _norm("SC Freiburg"):                "Freiburg",
    _norm("TSG 1899 Hoffenheim"):        "Hoffenheim",
    _norm("Borussia Monchengladbach"):   "M'gladbach",
    _norm("1.FC Union Berlin"):          "Union Berlin",
    _norm("1.FSV Mainz 05"):             "Mainz",
    _norm("FC Augsburg"):                "Augsburg",
    _norm("SV Werder Bremen"):           "Werder Bremen",
    _norm("1.FC Koln"):                  "FC Koln",
    _norm("1.FC Cologne"):               "FC Koln",
    _norm("1.FC Heidenheim 1846"):       "Heidenheim",
    _norm("VfL Bochum"):                 "Bochum",
    _norm("SV Darmstadt 98"):            "Darmstadt",
    _norm("FC Schalke 04"):              "Schalke 04",
    _norm("Hertha BSC"):                 "Hertha",
    _norm("Hamburger SV"):               "Hamburg",
    _norm("Hannover 96"):                "Hannover",
    _norm("FC Ingolstadt 04"):           "Ingolstadt",
    _norm("FC St. Pauli"):               "St Pauli",
    _norm("Fortuna Dusseldorf"):         "Fortuna Dusseldorf",
    _norm("1.FC Nurnberg"):              "Nurnberg",
    _norm("1.FC Nuremberg"):             "Nurnberg",
    _norm("SC Paderborn 07"):            "Paderborn",
    _norm("Arminia Bielefeld"):          "Bielefeld",
    _norm("DSC Arminia Bielefeld"):      "Bielefeld",
    _norm("SpVgg Greuther Furth"):       "Greuther Furth",
    _norm("Holstein Kiel"):              "Holstein Kiel",
}


def tm_to_fd(club_tm: str) -> str | None:
    return _TM_TO_FD.get(_norm(club_tm))


# ─────────────────────────────────────────────────────────────────────
# Scraping
# ─────────────────────────────────────────────────────────────────────
def _http_get(url: str, retries: int = 4, delay: float = 2.0):
    from bs4 import BeautifulSoup
    last = None
    for attempt in range(retries):
        try:
            raw = urllib.request.urlopen(
                urllib.request.Request(url, headers=_UA), timeout=30).read()
            try:
                html = raw.decode("utf-8")
            except UnicodeDecodeError:
                html = raw.decode("windows-1252", "replace")
            time.sleep(1.1)
            return BeautifulSoup(html, "html.parser")
        except Exception as e:        # noqa: BLE001 (Netzfehler tolerieren)
            last = e
            time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"GET fehlgeschlagen ({retries}x): {url} ({last})")


def _club_id(href: str) -> str | None:
    m = re.search(r"/verein/(\d+)", href or "")
    return m.group(1) if m else None


def _parse_value_eur(text: str) -> float | None:
    """'€427.75m' / '€2.05bn' / '€850k' / '€-23.30m' → EUR float (oder None)."""
    if not text:
        return None
    m = re.search(r"€\s*(-?[\d,.]+)\s*(bn|m|k)?", text.replace("\xa0", " "), re.I)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    mult = {"bn": 1e9, "m": 1e6, "k": 1e3, "": 1.0}[(m.group(2) or "").lower()]
    return num * mult


def _participants(season_id: int) -> dict[str, str]:
    """{club_id -> TM-Clubname} der 18 Saison-Clubs (saison_id-Startseite)."""
    soup = _http_get(f"{_BASE}/bundesliga/startseite/wettbewerb/L1/saison_id/{season_id}")
    out: dict[str, str] = {}
    for row in soup.select("table.items tbody tr"):
        link = row.find("a", href=re.compile(r"/startseite/verein/\d+"))
        if not link:
            continue
        cid = _club_id(link.get("href", ""))
        name = link.get("title") or link.get_text(strip=True)
        if cid and name:
            out[cid] = name
    return out


def _cutoff_values(cutoff_date: str) -> dict[str, float]:
    """{club_id -> Marktwert zum Stichtag} aus L1+L2+L3 (Pfad-Form-Stichtag).

    Erste Geldzelle einer Zeile = Gesamtmarktwert.
    """
    out: dict[str, float] = {}
    for slug, code in _COMPETITIONS:
        url = (f"{_BASE}/{slug}/marktwerteverein/wettbewerb/{code}/"
               f"stichtag/{cutoff_date}/plus/1")
        soup = _http_get(url)
        for row in soup.select("table.items tbody tr"):
            link = row.find("a", href=re.compile(r"/startseite/verein/\d+"))
            if not link:
                continue
            cid = _club_id(link.get("href", ""))
            if not cid:
                continue
            for cell in row.find_all("td"):
                txt = cell.get_text(" ", strip=True)
                if re.search(r"€\s*-?[\d,.]+", txt):
                    v = _parse_value_eur(txt)
                    if v is not None:
                        out[cid] = v
                    break
    return out


def fetch_and_cache(csv_path: str | Path = DEFAULT_CSV,
                    force: bool = False) -> pd.DataFrame:
    """Scrapt alle Saisons leak-frei (Stichtag) und schreibt den Cache."""
    path = Path(csv_path)
    if path.exists() and not force:
        print(f"  Cache existiert: {path} (force=True zum Neuladen)")
        return load_market_values(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cutoff_cache: dict[str, dict[str, float]] = {}
    rows, misses = [], []
    for sc in SEASONS:
        parts = _participants(sc.season_id)
        if len(parts) != 18:
            print(f"  ⚠ {sc.season}: {len(parts)} Clubs (≠18) auf saison_id-Seite")
        if sc.cutoff_date not in cutoff_cache:
            cutoff_cache[sc.cutoff_date] = _cutoff_values(sc.cutoff_date)
        vals = cutoff_cache[sc.cutoff_date]
        n_ok = 0
        for cid, name in parts.items():
            v = vals.get(cid)
            if v is not None:
                rows.append({"Season": sc.season, "ClubTM": name,
                             "MarketValueEUR": v})
                n_ok += 1
            else:
                misses.append((sc.season, name, cid))
        print(f"  {sc.season} @ {sc.cutoff_date}: {n_ok}/{len(parts)} mit Wert")
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  → {len(df)} Zeilen geschrieben nach {path}")
    if misses:
        print(f"  ⚠ Ohne Stichtag-Wert (heute < 3. Liga?): {misses}")
    return df


# ─────────────────────────────────────────────────────────────────────
# Laden + Verwendung
# ─────────────────────────────────────────────────────────────────────
def load_market_values(csv_path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Marktwert-Cache nicht gefunden: {path}. "
            f"Erst `python market_value.py` laufen lassen (scraped + cached).")
    df = pd.read_csv(path, encoding="utf-8")
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path} fehlen Spalten: {sorted(missing)}")
    df = df.dropna(subset=["MarketValueEUR"]).copy()
    df["MarketValueEUR"] = df["MarketValueEUR"].astype(float)
    return df[df["MarketValueEUR"] > 0.0]


def team_market_values(season: str,
                       csv_path: str | Path = DEFAULT_CSV) -> dict[str, float]:
    """{football-data-Teamname -> Kaderwert} als (leak-freier) PRIOR für ``season``.

    Der CSV-Wert je Saison ist bereits der Stichtag-Stand VOR dem 1. Spieltag
    → kein Lag nötig. Nicht abgebildete/fehlende Clubs → in build_league z=0.
    """
    df = load_market_values(csv_path)
    sub = df[df["Season"] == season]
    out: dict[str, float] = {}
    for _, row in sub.iterrows():
        fd = tm_to_fd(row["ClubTM"])
        if fd is not None:
            out[fd] = float(row["MarketValueEUR"])
    return out


def standardized_baselines(values: dict[str, float],
                           teams: list[str],
                           kappa: float) -> np.ndarray:
    """κ·z(log Kaderwert), ausgerichtet an ``teams``. Spiegelt die Inline-
    Logik in ``model.build_league`` (für Diagnose-Plots). Fehlend/≤0 ⇒ 0."""
    raw = np.array([values.get(t, np.nan) for t in teams], dtype=np.float64)
    out = np.zeros(len(teams), dtype=np.float64)
    if kappa == 0.0:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        logv = np.log(raw)
    mask = np.isfinite(logv)
    if int(mask.sum()) >= 2:
        mu = float(logv[mask].mean())
        sd = float(logv[mask].std())
        if sd > 0.0:
            out[mask] = kappa * (logv[mask] - mu) / sd
    return out


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    force = "--force" in sys.argv
    df = fetch_and_cache(force=force)
    print("\n  Abdeckung & Namens-Mapping:")
    unmapped = set()
    for s in sorted(df["Season"].unique()):
        sub = df[df["Season"] == s]
        mapped = team_market_values(s)
        miss = [c for c in sub["ClubTM"] if tm_to_fd(c) is None]
        unmapped.update(miss)
        top = max(mapped, key=mapped.get) if mapped else "—"
        print(f"  {s}: {len(sub):>2} Clubs, {len(mapped):>2} gemappt, "
              f"teuerstes={top}" + (f"  UNGEMAPPT={miss}" if miss else ""))
    if unmapped:
        print(f"\n  ⚠ Noch nicht in _TM_TO_FD: {sorted(unmapped)}")
    else:
        print("\n  ✓ Alle Clubs sauber auf football-data-Namen abgebildet.")
