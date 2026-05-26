"""
backtest_multiseason.py
=======================
MULTI-SEASON-SIGNIFIKANZSTUDIE — schlägt das Modell die Buchmacher-Schlusslinie
über mehrere Saisons *signifikant*, oder ist „4/5 besser" nur Rauschen?

Eine einzelne Saison (~94 Holdout-Spiele, SE des RPS ≈ 0.02) kann „auf Augenhöhe"
nicht von „kleiner echter Vorsprung" trennen. Dieser Driver:

  1. schleift den ehrlichen Walk-Forward (``backtest.evaluate_season_walkforward``)
     über alle Saisons ≥ 2014/15 (echtes xG verfügbar),
  2. **poolt die Per-Spiel-RPS-Differenzen** (Buchmacher − Modell) über alle
     Saisons,
  3. rechnet einen **gepaarten Wilcoxon-Signed-Rank-Test** + **Bootstrap-95%-CI**
     des mittleren ΔRPS — die eigentliche Signifikanzaussage,
  4. **protokolliert pro Saison** ``bm_source`` (welche Quote!) und ``xg_source``
     (echt/Proxy) → eingebauter Leck-/Fairness-Audit,
  5. optional: **Placebo / Negativ-Kontrolle** (``SCRAMBLE_XG``) — xG wird
     innerhalb jeder Saison verwürfelt. Verschwindet der „Vorsprung", sitzt das
     Signal echt im xG; bleibt er, gibt es einen Pipeline-Leak.

Modell-Einstellungen (Gamma/echtes-xG/φ/Hyperparameter/Ketten/Budget) werden
1:1 aus ``backtest.py`` übernommen — eine einzige Quelle der Wahrheit. Für die
saubere Studie dort ``USE_TUNED_HYPERPARAMS = False`` setzen (leckfrei).

Aufruf:  python backtest_multiseason.py
"""

from __future__ import annotations

import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from data import load_bundesliga, extract_bookmaker_probs
from xg import fit_xg_weights, add_xg_columns
from real_xg import add_real_xg_columns, UNDERSTAT_FIRST_YEAR
from mcmc import warmup_jit
from evaluation import rps_one
import backtest as bt   # Konfiguration + evaluate_season_walkforward (Quelle der Wahrheit)


# ─────────────────────────────────────────────────────────────────────
# Konfiguration (modellspezifisches kommt aus backtest.py)
# ─────────────────────────────────────────────────────────────────────
SEASONS: list[str] | None = ["2016/17","2017/18","2018/19","2019/20","2020/21","2021/22","2022/23","2023/24","2024/25","2025/26"]   # None = auto: alle Saisons ≥ 2014/15 im DataFrame.
                                   # Zum Kürzen z.B. ["2022/23", "2023/24", "2024/25"].
SCRAMBLE_XG   = False              # Placebo / Negativ-Kontrolle (xG je Saison verwürfeln)
SCRAMBLE_SEED = 123
BOOTSTRAP_N   = 10_000
BOOTSTRAP_SEED = 0
FIRST_SEASON_LABEL = f"{UNDERSTAT_FIRST_YEAR}/{(UNDERSTAT_FIRST_YEAR + 1) % 100:02d}"  # "2014/15"


# ─────────────────────────────────────────────────────────────────────
# Hilfen
# ─────────────────────────────────────────────────────────────────────
def _auto_seasons(df: pd.DataFrame) -> list[str]:
    """Alle (vollständigen) Saisons mit echtem xG, d.h. ≥ 2014/15."""
    out = []
    for s in sorted(df["Season"].unique()):
        if s >= FIRST_SEASON_LABEL and (df["Season"] == s).sum() >= 50:
            out.append(s)
    return out


def _scramble_xg(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Placebo: permutiert die (xG_home, xG_away)-PAARE innerhalb jeder Saison.
    Bricht die Verbindung Team-Leistung↔xG, behält aber die xG-Verteilung."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    for _, idx in out.groupby("Season").groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        perm = rng.permutation(len(idx))
        out.loc[idx, "xG_home"] = out.loc[idx, "xG_home"].to_numpy()[perm]
        out.loc[idx, "xG_away"] = out.loc[idx, "xG_away"].to_numpy()[perm]
    return out


def _per_match_rps(cmp: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-Spiel-RPS (Modell, Buchmacher) auf dem fairen gemeinsamen Holdout
    (genau die Spiele, auf denen ``evaluate_season_walkforward`` aggregiert)."""
    pm = cmp["probs_model"]
    pb = cmp["probs_bookmaker"]
    out = cmp["outcomes"]
    n, cutoff = cmp["n"], cmp["cutoff"]
    hold = np.arange(n) >= cutoff
    fin = np.all(np.isfinite(pm), axis=1) & np.all(np.isfinite(pb), axis=1)
    common = hold & fin
    rps_m = np.array([rps_one(pm[i], out[i]) for i in np.where(common)[0]])
    rps_b = np.array([rps_one(pb[i], out[i]) for i in np.where(common)[0]])
    return rps_m, rps_b


def _paired_stats(rps_model: np.ndarray, rps_book: np.ndarray,
                  n_boot: int, seed: int) -> dict:
    """Gepaarte Statistik auf den Per-Spiel-RPS. diff = Buchmacher − Modell
    (positiv = Modell besser)."""
    diff = rps_book - rps_model
    n = len(diff)
    mean_diff = float(diff.mean())
    rel = mean_diff / float(rps_book.mean()) * 100.0

    # Wilcoxon-Signed-Rank (gepaart, zweiseitig)
    try:
        w_stat, w_p = wilcoxon(rps_model, rps_book, alternative="two-sided")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    # Bootstrap-CI des mittleren ΔRPS (gepaart: Spiele resampeln)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = diff[idx].mean()
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    return {
        "n": n, "mean_diff": mean_diff, "rel_pct": rel,
        "wilcoxon_p": float(w_p), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "frac_model_better": float((diff > 0).mean()),
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    print("=" * 72)
    print(" MULTI-SEASON-SIGNIFIKANZSTUDIE — Modell vs. Buchmacher-Schlusslinie")
    print("=" * 72)

    if not bt.USE_XG:
        raise ValueError("Diese Studie braucht USE_XG=True in backtest.py.")
    if bt.USE_TUNED_HYPERPARAMS:
        print("  HINWEIS: backtest.USE_TUNED_HYPERPARAMS=True — für die saubere "
              "Studie besser False (leckfreie feste Paper-Hyperparameter).")

    # 1) Daten + Quoten
    df = load_bundesliga(bt.START_YEAR, bt.END_YEAR, with_extras=True)
    df = extract_bookmaker_probs(df)

    seasons = SEASONS if SEASONS is not None else _auto_seasons(df)
    print(f"  Saisons ({len(seasons)}): {seasons}")

    # 2) xG: Proxy-β leckfrei vor der frühesten Analyse-Saison fitten
    #    (bei echtem xG ohnehin nur Fallback < 2014/15 — wird hier nie gescort).
    earliest = min(seasons)
    df_train = df[df["Season"] < earliest]
    beta_off, beta_on = fit_xg_weights(
        df_train if len(df_train) > 100 else df, force=True, cache=False)
    df = add_xg_columns(df, beta_off, beta_on)
    if bt.USE_REAL_XG:
        df = add_real_xg_columns(df)
    if SCRAMBLE_XG:
        print("  ⚠ PLACEBO-MODUS: xG wird je Saison verwürfelt (Negativ-Kontrolle).")
        df = _scramble_xg(df, SCRAMBLE_SEED)

    # 3) Quellen-Audit pro Saison (welche Quote? echtes/Proxy-xG?)
    print(f"\n  {'Saison':9} {'bm_source (ganze Saison)':<34} {'xg_source':<28}")
    print("  " + "-" * 70)
    for s in seasons:
        sub = df[df["Season"] == s]
        bm = ", ".join(f"{k}={v}" for k, v in sub["bm_source"].value_counts().items())
        xs = ", ".join(f"{k}={v}" for k, v in sub.get(
            "xg_source", pd.Series(["?"] * len(sub))).value_counts().items())
        print(f"  {s:9} {bm:<34} {xs:<28}")

    # 4) JIT + Ketten
    print("\n  Numba-JIT vorkompilieren ...", end="", flush=True)
    warmup_jit()
    print(" OK")
    n_chains = bt.N_CHAINS if bt.USE_MULTI_CHAIN else 1

    # 5) Walk-Forward je Saison, Per-Spiel-RPS poolen
    all_m, all_b = [], []
    per_season = []
    for s in seasons:
        if bt.USE_TUNED_HYPERPARAMS:
            tau, gamma, eps = bt.resolve_hyperparams(s, bt.USE_XG)
        else:
            tau, gamma, eps = bt.DEFAULT_TAU, bt.DEFAULT_GAMMA, bt.DEFAULT_EPSILON
        print(f"\n  ── Saison {s}  (τ={tau}, γ={gamma}, ε={eps}) "
              f"{'─'*28}")
        cmp = bt.evaluate_season_walkforward(
            df, s, use_xg=bt.USE_XG, tau=tau, gamma=gamma, eps=eps,
            holdout_frac=bt.HOLDOUT_FRAC, verbose=True,
            n_chains=n_chains, jitter_sd=bt.CHAIN_JITTER_SD,
        )
        rps_m, rps_b = _per_match_rps(cmp)
        all_m.append(rps_m)
        all_b.append(rps_b)
        rel = (rps_b.mean() - rps_m.mean()) / rps_b.mean() * 100.0
        per_season.append({
            "season": s, "n": len(rps_m),
            "rps_model": float(rps_m.mean()), "rps_book": float(rps_b.mean()),
            "d_rps": float(rps_b.mean() - rps_m.mean()), "rel": rel,
            "rhat": cmp.get("rhat_max"),
        })

    rps_model = np.concatenate(all_m)
    rps_book = np.concatenate(all_b)
    stats = _paired_stats(rps_model, rps_book, BOOTSTRAP_N, BOOTSTRAP_SEED)

    # 6) Bericht
    print("\n" + "=" * 72)
    print(" ERGEBNIS")
    print("=" * 72)
    print(f"\n  {'Saison':9} {'n':>4} {'RPS_Modell':>11} {'RPS_Buch':>10} "
          f"{'ΔRPS':>9} {'rel%':>7} {'R-hat':>7}")
    print("  " + "-" * 64)
    n_better = 0
    for r in per_season:
        n_better += int(r["d_rps"] > 0)
        rh = f"{r['rhat']:.2f}" if r["rhat"] is not None and np.isfinite(r["rhat"]) else "—"
        print(f"  {r['season']:9} {r['n']:>4} {r['rps_model']:>11.4f} "
              f"{r['rps_book']:>10.4f} {r['d_rps']:>+9.4f} {r['rel']:>+7.1f} {rh:>7}")

    print("\n  Gepoolt (alle Spiele zusammen):")
    print(f"    Spiele insgesamt:       n = {stats['n']}")
    print(f"    RPS Modell / Buchmacher: {rps_model.mean():.4f} / {rps_book.mean():.4f}")
    print(f"    mittleres ΔRPS (Buch−Modell): {stats['mean_diff']:+.5f} "
          f"({stats['rel_pct']:+.2f} % rel.)")
    print(f"    Bootstrap-95%-CI von ΔRPS:    "
          f"[{stats['ci_lo']:+.5f}, {stats['ci_hi']:+.5f}]")
    print(f"    Wilcoxon-Signed-Rank p:       {stats['wilcoxon_p']:.4f}")
    print(f"    Spiele Modell besser:         {stats['frac_model_better']*100:.1f} %")
    print(f"    Saisons Modell besser:        {n_better}/{len(per_season)}")

    # 7) Urteil
    sig = (stats["ci_lo"] > 0 or stats["ci_hi"] < 0) and stats["wilcoxon_p"] < 0.05
    print("\n  → ", end="")
    if SCRAMBLE_XG:
        if sig and stats["mean_diff"] > 0:
            print("PLACEBO zeigt TROTZDEM einen Vorsprung → PIPELINE-LECK! "
                  "(verwürfeltes xG dürfte nicht helfen.)")
        else:
            print("PLACEBO: kein Vorsprung mit verwürfeltem xG → konsistent mit "
                  "'Signal sitzt echt im xG, kein Leck'.")
    elif sig and stats["mean_diff"] > 0:
        print("SIGNIFIKANTER Vorsprung des Modells (CI schließt 0 aus, p<0.05).")
    elif sig and stats["mean_diff"] < 0:
        print("Buchmacher SIGNIFIKANT besser.")
    else:
        print("KEIN signifikanter Unterschied — auf Augenhöhe mit der "
              "Schlusslinie (CI überdeckt 0).")

    print(f"\n  Fertig in {(time.time()-t_start)/60:.1f} min.")


if __name__ == "__main__":
    main()
