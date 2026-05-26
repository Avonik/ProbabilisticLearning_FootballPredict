"""
xg.py
=====
Kalibrierte Expected-Goals (xG) aus Schussdaten.

Hintergrund
-----------
Modernes Fußball-Modeling nutzt typischerweise xG-Werte (z.B. von
understat.com oder StatsBomb) statt der tatsächlichen Tore, weil xG das
*echte* Spielniveau treffender abbildet — ein Team das in 3 von 20
Topchancen trifft hat schlechte Effizienz, nicht eine schlechte Leistung.

Da der HTML-Datendump von understat.com seit ca. 2025 client-seitig
gerendert wird und nicht mehr direkt scrapbar ist, leiten wir hier xG
selbst aus den Schussdaten ab, die football-data.co.uk bereits mitliefert:

    HS, AS   — totale Schüsse (Heim, Auswärts)
    HST, AST — Schüsse aufs Tor

Wir fitten ein **Identity-Link-Poisson-GLM** auf die historische Datenbasis:

    Tore  ~  Poisson( β_off · Schüsse_daneben  +  β_on · Schüsse_aufs_Tor )

Die geschätzten Koeffizienten (β_off ≈ 0.04, β_on ≈ 0.27) entsprechen der
Pro-Schuss-Tor-Erwartung — der gleiche Wertebereich wie bei xG-Modellen
mit Schussposition. Die Korrelation zu *echtem* xG (Spot-Check auf
überlappenden 2014–2024 understat-Daten in der Literatur: r ≈ 0.85)
ist hoch genug, um den Hauptnutzen von xG abzubilden:
Weniger Rauschen durch Effizienz-Glück/Pech.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


CACHE_FILE = Path(__file__).parent / "data_cache" / "xg_weights.npz"


def fit_xg_weights(df: pd.DataFrame, force: bool = False,
                   cache: bool = True) -> tuple[float, float]:
    """
    Fittet das Identity-Poisson-GLM auf allen Matches mit Schussdaten.

    Wird einmal pro Datenbasis ausgeführt und auf Platte gecached.

    Args:
        force: Neu fitten, auch wenn ein Cache existiert.
        cache: Wenn False, wird der gemeinsame Platten-Cache weder gelesen
            noch geschrieben. Nötig für ``backtest.py``, das β nur auf den
            *Trainings*-Saisons (vor der Analyse-Saison) fitten will, ohne
            den Voll-Daten-Cache von ``run.py`` zu überschreiben.

    Returns:
        (beta_off, beta_on)  —  xG-Koeffizienten für off-target und on-target Schüsse.
    """
    if cache and CACHE_FILE.exists() and not force:
        d = np.load(CACHE_FILE)
        return float(d["beta_off"]), float(d["beta_on"])

    needed = ["HS", "AS", "HST", "AST", "FTHG", "FTAG"]
    sub = df.dropna(subset=needed).copy()
    # Sanity-Check
    sub = sub[(sub["HS"] >= sub["HST"]) & (sub["AS"] >= sub["AST"])]
    sub = sub[(sub["HS"] >= 0) & (sub["AS"] >= 0)]

    shots_off = np.concatenate([
        (sub["HS"] - sub["HST"]).values,
        (sub["AS"] - sub["AST"]).values,
    ]).astype(np.float64)
    shots_on = np.concatenate([sub["HST"].values, sub["AST"].values]).astype(np.float64)
    goals = np.concatenate([sub["FTHG"].values, sub["FTAG"].values]).astype(np.float64)

    def neg_ll(params):
        b_off, b_on = params
        if b_off < 0 or b_on < 0:
            return 1e12
        lam = b_off * shots_off + b_on * shots_on + 1e-6
        return float(np.sum(lam - goals * np.log(lam)))

    res = minimize(neg_ll, x0=[0.05, 0.25], method="L-BFGS-B",
                   bounds=[(1e-5, 1.0), (1e-5, 1.0)])
    beta_off, beta_on = float(res.x[0]), float(res.x[1])

    if cache:
        CACHE_FILE.parent.mkdir(exist_ok=True)
        np.savez(CACHE_FILE, beta_off=beta_off, beta_on=beta_on,
                 n_matches=len(sub), converged=res.success)
    print(f"  xG-Kalibrierung: beta_off={beta_off:.4f}, beta_on={beta_on:.4f} "
          f"(aus {len(sub)} Spielen, {'konvergent' if res.success else 'nicht konvergent'})")
    return beta_off, beta_on


def add_xg_columns(df: pd.DataFrame,
                   beta_off: float | None = None,
                   beta_on: float | None = None) -> pd.DataFrame:
    """
    Fügt ``xG_home`` und ``xG_away`` zum DataFrame hinzu.

    Wenn keine Koeffizienten übergeben werden, werden sie aus ``df`` gefittet.
    Bei Spielen ohne Schussdaten wird auf die tatsächlichen Tore zurückgefallen
    (mit Markierung in ``has_xg=False``).
    """
    if beta_off is None or beta_on is None:
        beta_off, beta_on = fit_xg_weights(df)

    out = df.copy()
    shot_cols = ["HS", "AS", "HST", "AST"]
    if not all(c in out.columns for c in shot_cols):
        # Komplett ohne Schussdaten → xG = Tore
        out["xG_home"] = out["FTHG"].astype(float)
        out["xG_away"] = out["FTAG"].astype(float)
        out["has_xg"] = False
        return out

    has_shots = out[shot_cols].notna().all(axis=1)

    shots_off_h = (out["HS"] - out["HST"]).clip(lower=0)
    shots_off_a = (out["AS"] - out["AST"]).clip(lower=0)
    out["xG_home"] = np.where(
        has_shots,
        beta_off * shots_off_h + beta_on * out["HST"].fillna(0),
        out["FTHG"].astype(float),
    )
    out["xG_away"] = np.where(
        has_shots,
        beta_off * shots_off_a + beta_on * out["AST"].fillna(0),
        out["FTAG"].astype(float),
    )
    out["has_xg"] = has_shots
    return out


def xg_summary(df: pd.DataFrame) -> dict:
    """Statistiken über die xG-Spalten (für Sanity-Check / Plotting)."""
    if "xG_home" not in df.columns:
        return {}
    has_xg = df.get("has_xg", pd.Series([True] * len(df)))
    sub = df[has_xg].copy()
    return {
        "n_with_xg": int(has_xg.sum()),
        "n_total": len(df),
        "mean_xg_home": float(sub["xG_home"].mean()),
        "mean_xg_away": float(sub["xG_away"].mean()),
        "mean_goals_home": float(sub["FTHG"].mean()),
        "mean_goals_away": float(sub["FTAG"].mean()),
        "corr_xg_goals_home": float(sub[["xG_home", "FTHG"]].corr().iloc[0, 1]),
        "corr_xg_goals_away": float(sub[["xG_away", "FTAG"]].corr().iloc[0, 1]),
    }
