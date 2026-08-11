"""Leak-free historical priors for team-specific home advantage.

Each completed season before the target season is fitted independently. The
posterior team effects are combined with exponential recency weights and used
as prior means in the target season. Per-season fits are cached because the
same historical season is reused by many target seasons.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mcmc import posterior_home_advantage, run_mcmc
from model import build_league


CACHE_VERSION = 2  # v2 seeds Numba's RNG explicitly
_FIT_LOCK = threading.Lock()


def _data_fingerprint(df: pd.DataFrame) -> str:
    cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    for col in ("xG_home", "xG_away"):
        if col in df.columns:
            cols.append(col)
    payload = (df[cols].copy()
               .sort_values(["Date", "HomeTeam", "AwayTeam"], kind="stable")
               .reset_index(drop=True))
    payload["Date"] = pd.to_datetime(payload["Date"]).dt.strftime("%Y-%m-%d")
    return hashlib.sha256(payload.to_csv(index=False).encode("utf-8")).hexdigest()


def _cache_config(df: pd.DataFrame, season: str, **settings) -> dict:
    return {
        "version": CACHE_VERSION,
        "season": season,
        "data_sha256": _data_fingerprint(df),
        **settings,
    }


def _cache_path(cache_dir: Path, config: dict) -> Path:
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    tag = str(config["season"]).replace("/", "_")
    return cache_dir / f"home_{tag}_{digest}.pkl"


def _load_cache(path: Path, config: dict) -> dict[str, float] | None:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if payload.get("config") != config:
            return None
        return {str(k): float(v) for k, v in payload["effects"].items()}
    except (OSError, EOFError, pickle.PickleError, KeyError, TypeError, ValueError):
        return None


def _save_cache(path: Path, config: dict, effects: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{time.time_ns()}.tmp")
    with open(tmp, "wb") as handle:
        pickle.dump({"config": config, "effects": effects}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _fit_historical_season(
    df_season: pd.DataFrame,
    season: str,
    *,
    use_xg: bool,
    tau: float,
    gamma: float,
    epsilon: float,
    continuous_xg: bool,
    phi: float,
    prior_sd: float,
    n_iter: int,
    burnin: int,
    thin: int,
    proposal_sd: float,
    seed: int,
    cache_dir: Path,
) -> tuple[dict[str, float], bool]:
    settings = {
        "use_xg": bool(use_xg), "tau": float(tau),
        "gamma": float(gamma), "epsilon": float(epsilon),
        "continuous_xg": bool(continuous_xg), "phi": float(phi),
        "prior_sd": float(prior_sd), "n_iter": int(n_iter),
        "burnin": int(burnin), "thin": int(thin),
        "proposal_sd": float(proposal_sd), "seed": int(seed),
    }
    config = _cache_config(df_season, season, **settings)
    path = _cache_path(cache_dir, config)

    cached = _load_cache(path, config)
    if cached is not None:
        return cached, True

    # Serialise cache misses. Besides avoiding duplicate work, this keeps the
    # Numba/NumPy RNG setup deterministic when outer season jobs use threads.
    with _FIT_LOCK:
        cached = _load_cache(path, config)
        if cached is not None:
            return cached, True
        league = build_league(
            df_season, use_xg=use_xg, tau=tau, gamma=gamma,
            epsilon=epsilon, continuous_xg=continuous_xg, phi=phi,
            use_team_home_advantage=True, home_adv_prior_sd=prior_sd,
        )
        samples = run_mcmc(
            league, n_iter=n_iter, burnin=burnin, thin=thin,
            proposal_sd=proposal_sd, seed=seed, verbose=False,
        )
        means = posterior_home_advantage(samples)
        effects = {
            team: float(means[i]) for i, team in enumerate(league.teams)
        }
        _save_cache(path, config, effects)
        return effects, False


def combine_season_effects(
    season_effects: list[dict[str, float]],
    target_teams: list[str],
    decay: float,
) -> dict[str, float]:
    """Combine oldest-to-newest season estimates with recency weights.

    Missing recent seasons attenuate a returning team's prior toward zero.
    Teams absent from every historical season receive exactly zero.
    """
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be in (0, 1].")
    if not season_effects:
        return {team: 0.0 for team in target_teams}

    n = len(season_effects)
    weights = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=float)
    denominator = float(weights.sum())
    return {
        team: float(sum(w * effects.get(team, 0.0)
                        for w, effects in zip(weights, season_effects)) /
                    denominator)
        for team in target_teams
    }


def estimate_historical_home_prior(
    df_full: pd.DataFrame,
    *,
    target_season: str,
    target_teams: list[str],
    use_xg: bool,
    tau: float,
    gamma: float,
    epsilon: float,
    continuous_xg: bool,
    phi: float,
    prior_sd: float,
    n_seasons: int,
    decay: float,
    n_iter: int,
    burnin: int,
    thin: int,
    proposal_sd: float,
    seed: int,
    cache_dir: Path,
) -> tuple[dict[str, float], dict]:
    if n_seasons < 1:
        raise ValueError("n_seasons must be at least 1.")
    available = sorted(str(s) for s in df_full["Season"].dropna().unique()
                       if str(s) < target_season)
    selected = available[-n_seasons:]

    effects_by_season: list[dict[str, float]] = []
    cache_hits = 0
    for season in selected:
        df_season = df_full[df_full["Season"] == season].copy()
        season_seed = seed + int(season.split("/")[0])
        effects, hit = _fit_historical_season(
            df_season, season, use_xg=use_xg, tau=tau, gamma=gamma,
            epsilon=epsilon, continuous_xg=continuous_xg, phi=phi,
            prior_sd=prior_sd, n_iter=n_iter, burnin=burnin, thin=thin,
            proposal_sd=proposal_sd, seed=season_seed, cache_dir=cache_dir,
        )
        effects_by_season.append(effects)
        cache_hits += int(hit)

    combined = combine_season_effects(effects_by_season, target_teams, decay)
    return combined, {
        "seasons": selected,
        "cache_hits": cache_hits,
        "decay": float(decay),
        "n_iter": int(n_iter),
    }
