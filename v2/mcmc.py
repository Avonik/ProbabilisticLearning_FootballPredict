"""
mcmc.py
=======
Numba-beschleunigter Metropolis-Hastings-Sampler für Rue-Salvesen (v2).

Algorithmus identisch zu v1 (Single-Site M-H mit Time-Prior aus
Brownscher Bewegung + Mixture-Indikator δ), aber der innere Schleifenkern
ist als ``@njit``-Funktion JIT-kompiliert. Typischer Speedup: 50–100×.

Public API:
  - ``run_mcmc(L, ...)``    →  dict mit ``attack``, ``defense``, ``delta``,
    ``acceptance``         (Format identisch zu v1, von viz.py konsumiert)
  - ``posterior_means(samples)``  →  identisch zu v1
  - ``warmup_jit()``        →  triggert die JIT-Kompilation,
    damit parallele Worker den Cache nutzen können

Implementation:
  Die State-Variablen (attack, defense) werden als *flache* numpy-Arrays
  geführt, segmentiert durch ``team_start``. Das Mapping zurück auf die
  dict-API von League erfolgt erst beim Speichern der Samples.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from model import (
    League,
    DEFAULT_TAU, DEFAULT_GAMMA, DEFAULT_EPSILON, PRIOR_VAR, MAX_GOALS,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# Log(k!) für k=0..MAX_GOALS, als modul-Konstante
_LOG_FACT = np.array([
    0.0,                              # log 0!
    0.0,                              # log 1!
    np.log(2.0),                      # log 2!
    np.log(2.0) + np.log(3.0),
    np.log(2.0) + np.log(3.0) + np.log(4.0),
    np.log(2.0) + np.log(3.0) + np.log(4.0) + np.log(5.0),
], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────
# Numba-kompilierte Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _poisson_pmf(k, lam):
    if lam <= 0.0:
        return 1.0 if k == 0 else 0.0
    return np.exp(k * np.log(lam) - lam - _LOG_FACT[k])


@njit(cache=True, fastmath=True)
def _poisson_tail(max_k, lam):
    if lam <= 0.0:
        return 0.0
    s = 0.0
    log_lam = np.log(lam)
    for k in range(max_k):
        s += np.exp(k * log_lam - lam - _LOG_FACT[k])
    rem = 1.0 - s
    return rem if rem > 1e-12 else 1e-12


@njit(cache=True, fastmath=True)
def _trunc_poisson_pmf(k, lam, max_k):
    if k < 0 or k > max_k:
        return 0.0
    if k < max_k:
        return _poisson_pmf(k, lam)
    return _poisson_tail(max_k, lam)


@njit(cache=True, fastmath=True)
def _dc_correction(x, y, lam_x, lam_y):
    if x == 0 and y == 0:
        return 1.0 + 0.1 * lam_x * lam_y
    if x == 0 and y == 1:
        v = 1.0 - 0.1 * lam_x
        return v if v > 1e-6 else 1e-6
    if x == 1 and y == 0:
        v = 1.0 - 0.1 * lam_y
        return v if v > 1e-6 else 1e-6
    if x == 1 and y == 1:
        return 1.1
    return 1.0


@njit(cache=True, fastmath=True)
def _compute_lambdas(a_h, d_h, a_a, d_a, c_x, c_y, gamma):
    delta = (a_h + d_h - a_a - d_a) * 0.5
    log_lam_x = c_x + a_h - d_a - gamma * delta
    log_lam_y = c_y + a_a - d_h + gamma * delta
    return np.exp(log_lam_x), np.exp(log_lam_y)


@njit(cache=True, fastmath=True)
def _gamma_loglik_lam(g, lam, phi):
    """λ-abhängiger Teil der Gamma-Beobachtungs-Loglik.

    Modell: g ~ Gamma(shape=φ, rate=φ/λ)  ⇒  E[g]=λ, Var[g]=λ²/φ.
    Volle log-Dichte = φ·log(φ/λ) − logΓ(φ) + (φ−1)·log g − (φ/λ)·g.
    Die in g und φ konstanten Terme (−logΓ(φ)+(φ−1)log g) sind in jedem
    Metropolis-Quotienten identisch (g, φ fix) und werden weggelassen.
    """
    return -phi * np.log(lam) - phi * g / lam


@njit(cache=True, fastmath=True)
def _match_loglik(
    m,
    attack_flat, defense_flat,
    team_start, home_idx, away_idx,
    match_home_local, match_away_local,
    obs_x, obs_y,
    c_x, c_y, gamma,
    delta_m,
    lam_x_avg, lam_y_avg,
    max_k,
    continuous, phi,
):
    h = home_idx[m]
    a = away_idx[m]
    lh = match_home_local[m]
    la = match_away_local[m]
    a_h = attack_flat[team_start[h] + lh]
    d_h = defense_flat[team_start[h] + lh]
    a_a = attack_flat[team_start[a] + la]
    d_a = defense_flat[team_start[a] + la]

    if delta_m == 1:
        lam_x = lam_x_avg
        lam_y = lam_y_avg
    else:
        lam_x, lam_y = _compute_lambdas(a_h, d_h, a_a, d_a, c_x, c_y, gamma)

    if continuous:
        # Kontinuierliches xG: Gamma-Messmodell, kein Trunc/Dixon-Coles.
        return (_gamma_loglik_lam(obs_x[m], lam_x, phi) +
                _gamma_loglik_lam(obs_y[m], lam_y, phi))

    # Diskret: int-Cast hält den Numba-Branch typkorrekt (No-Op bei int64-obs).
    x = int(obs_x[m])
    y = int(obs_y[m])
    p = (_trunc_poisson_pmf(x, lam_x, max_k) *
         _trunc_poisson_pmf(y, lam_y, max_k) *
         _dc_correction(x, y, lam_x, lam_y))
    if p < 1e-300:
        p = 1e-300
    return np.log(p)


@njit(cache=True, fastmath=True)
def _time_prior_logpdf(val, prev_val, dt, sigma2, tau):
    var = dt / tau * sigma2
    if var < 1e-9:
        var = 1e-9
    diff = val - prev_val
    return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * diff * diff / var


@njit(cache=True, fastmath=True)
def _initial_prior_logpdf(val, mu, sigma2):
    # N(mu, sigma2): mu=0 reproduziert exakt den klassischen 0-Prior.
    d = val - mu
    return -0.5 * np.log(2.0 * np.pi * sigma2) - 0.5 * d * d / sigma2


# ─────────────────────────────────────────────────────────────────────
# Numba-kompilierter MCMC-Chunk
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _run_chunk(
    # State
    attack_flat, defense_flat, delta,
    # League-Struktur (flach)
    team_start, team_matches_flat, strength_days,
    home_idx, away_idx, match_home_local, match_away_local,
    obs_x, obs_y,
    # Hyperparameter
    c_x, c_y, gamma, epsilon, tau, sigma2,
    init_prior_mean,         # shape (n_teams,): Prior-Mittel der Initial-Stärke
    max_k,
    continuous, phi,
    proposal_sd,
    # Lauf-Steuerung
    it_start, it_end,        # absolute iter range [it_start, it_end)
    burnin, thin,
    # Output-Buffer
    sample_attack, sample_defense, sample_delta,
    sample_offset,           # erster freier Sample-Slot
):
    """
    Führt iterations [it_start, it_end) aus.

    Returns:
        accept_count, propose_count, new_sample_offset
    """
    n_teams = team_start.shape[0] - 1
    n_matches = home_idx.shape[0]
    lam_x_avg = np.exp(c_x)
    lam_y_avg = np.exp(c_y)
    log_eps = np.log(epsilon)
    log_one_minus_eps = np.log(1.0 - epsilon)

    accept_count = 0
    propose_count = 0
    n_saved = sample_offset

    for it in range(it_start, it_end):

        # === 1) Alle Stärken updaten (zufällige Reihenfolge) ===========
        team_order = np.random.permutation(n_teams)
        for ti in range(n_teams):
            team = team_order[ti]
            s_start = team_start[team]
            s_end = team_start[team + 1]
            n_local = s_end - s_start
            local_order = np.random.permutation(n_local)

            for li in range(n_local):
                local = local_order[li]
                gidx = s_start + local
                m = team_matches_flat[gidx]
                days_this = strength_days[gidx]

                # Für beide Komponenten (attack, defense)
                for which in range(2):
                    if which == 0:
                        current = attack_flat[gidx]
                    else:
                        current = defense_flat[gidx]
                    proposed = current + np.random.normal(0.0, proposal_sd)

                    # Prior aus vorherigem Spiel
                    log_alpha = 0.0
                    if local == 0:
                        # Initial-Stärke: informativer Prior um init_prior_mean[team]
                        # (Marktwert-Baseline; 0 ⇒ klassischer 0-Prior).
                        mu0 = init_prior_mean[team]
                        log_alpha += _initial_prior_logpdf(proposed, mu0, sigma2)
                        log_alpha -= _initial_prior_logpdf(current, mu0, sigma2)
                    else:
                        prev_idx = gidx - 1
                        dt = days_this - strength_days[prev_idx]
                        if which == 0:
                            prev_val = attack_flat[prev_idx]
                        else:
                            prev_val = defense_flat[prev_idx]
                        log_alpha += _time_prior_logpdf(proposed, prev_val,
                                                       dt, sigma2, tau)
                        log_alpha -= _time_prior_logpdf(current, prev_val,
                                                       dt, sigma2, tau)

                    # Prior zum nächsten Spiel
                    if local < n_local - 1:
                        next_idx = gidx + 1
                        dt2 = strength_days[next_idx] - days_this
                        if which == 0:
                            next_val = attack_flat[next_idx]
                        else:
                            next_val = defense_flat[next_idx]
                        log_alpha += _time_prior_logpdf(next_val, proposed,
                                                       dt2, sigma2, tau)
                        log_alpha -= _time_prior_logpdf(next_val, current,
                                                       dt2, sigma2, tau)

                    # Likelihood (nur das betroffene Spiel)
                    old_ll = _match_loglik(
                        m, attack_flat, defense_flat,
                        team_start, home_idx, away_idx,
                        match_home_local, match_away_local,
                        obs_x, obs_y, c_x, c_y, gamma,
                        delta[m], lam_x_avg, lam_y_avg, max_k,
                        continuous, phi,
                    )
                    if which == 0:
                        attack_flat[gidx] = proposed
                    else:
                        defense_flat[gidx] = proposed
                    new_ll = _match_loglik(
                        m, attack_flat, defense_flat,
                        team_start, home_idx, away_idx,
                        match_home_local, match_away_local,
                        obs_x, obs_y, c_x, c_y, gamma,
                        delta[m], lam_x_avg, lam_y_avg, max_k,
                        continuous, phi,
                    )
                    log_alpha += new_ll - old_ll

                    if np.log(np.random.random()) < log_alpha:
                        accept_count += 1
                    else:
                        # Zurücksetzen
                        if which == 0:
                            attack_flat[gidx] = current
                        else:
                            defense_flat[gidx] = current
                    propose_count += 1

        # === 2) Bernoulli-Indikatoren updaten ==========================
        m_order = np.random.permutation(n_matches)
        for mi in range(n_matches):
            m = m_order[mi]
            current = delta[m]
            proposed = 1 - current

            old_ll = _match_loglik(
                m, attack_flat, defense_flat,
                team_start, home_idx, away_idx,
                match_home_local, match_away_local,
                obs_x, obs_y, c_x, c_y, gamma,
                current, lam_x_avg, lam_y_avg, max_k,
                continuous, phi,
            )
            new_ll = _match_loglik(
                m, attack_flat, defense_flat,
                team_start, home_idx, away_idx,
                match_home_local, match_away_local,
                obs_x, obs_y, c_x, c_y, gamma,
                proposed, lam_x_avg, lam_y_avg, max_k,
                continuous, phi,
            )
            log_prior_curr = log_eps if current == 1 else log_one_minus_eps
            log_prior_prop = log_eps if proposed == 1 else log_one_minus_eps
            log_alpha = (log_prior_prop - log_prior_curr) + (new_ll - old_ll)

            if np.log(np.random.random()) < log_alpha:
                delta[m] = proposed

        # === 3) Globalen Level-Drift entfernen =========================
        # Nicht-identifizierbare Richtung a+=k, d+=k
        s = 0.0
        n_total = attack_flat.shape[0]
        for i in range(n_total):
            s += attack_flat[i] + defense_flat[i]
        global_mean = s / (2.0 * n_total)
        for i in range(n_total):
            attack_flat[i] -= global_mean
            defense_flat[i] -= global_mean

        # === 4) Snapshot speichern =====================================
        if it >= burnin and (it - burnin) % thin == 0:
            for i in range(n_total):
                sample_attack[n_saved, i] = attack_flat[i]
                sample_defense[n_saved, i] = defense_flat[i]
            for m in range(n_matches):
                sample_delta[n_saved, m] = delta[m]
            n_saved += 1

    return accept_count, propose_count, n_saved


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def _flat_samples_to_dicts(L, sample_attack, sample_defense, sample_delta,
                           n_saved):
    """Konvertiert flache Sample-Arrays in das v1-Dict-Format."""
    n_teams = L.n_teams
    out_a, out_d, out_delta = [], [], []
    for s in range(n_saved):
        ad: dict = {}
        dd: dict = {}
        for t in range(n_teams):
            start = int(L.team_start[t])
            end = int(L.team_start[t + 1])
            ad[t] = sample_attack[s, start:end].copy()
            dd[t] = sample_defense[s, start:end].copy()
        out_a.append(ad)
        out_d.append(dd)
        out_delta.append(sample_delta[s].copy())
    return out_a, out_d, out_delta


def run_mcmc(L: League, n_iter: int = 5000, burnin: int = 1000,
             thin: int = 5, proposal_sd: float = 0.05,
             seed: int = 42, verbose: bool = True,
             progress_queue=None, chunk: int = 200,
             init_attack: np.ndarray | None = None,
             init_defense: np.ndarray | None = None,
             init_delta: np.ndarray | None = None):
    """
    Numba-MCMC. Identisches Schnittstellen-Format zu v1.

    Args:
        L: League-Struktur (mit flachen Arrays + Hyperparametern).
        n_iter, burnin, thin: Standard-MCMC-Parameter.
        proposal_sd: SD der Metropolis-Proposal-Verteilung.
        seed: Zufallszahl-Seed (für Reproduzierbarkeit).
        verbose: tqdm-Progressbar.
        progress_queue: optional multiprocessing.Queue für globale Anzeige.
        chunk: Größe der iter-Pakete (Default 200) — kleiner Wert ⇒
            häufigere Progressbar-Updates, marginal höhere Python-Latenz.
        init_attack, init_defense: optionale flache Start-Arrays
            (shape ``(team_start[-1],)``) zum *Warm-Start*. Bei
            Walk-Forward-Refits (``backtest.py``) initialisiert man damit
            jeden Knoten eines Teams auf dessen zuletzt bekannte Stärke,
            sodass der Burn-in drastisch kürzer ausfällt.
        init_delta: optionaler flacher Start-Vektor für die Mischungs-
            Indikatoren (shape ``(n_matches,)``).
    """
    np.random.seed(seed)

    n_total = int(L.team_start[-1])
    attack_flat = np.zeros(n_total, dtype=np.float64)
    defense_flat = np.zeros(n_total, dtype=np.float64)
    delta = np.zeros(L.n_matches, dtype=np.int64)
    if init_attack is not None:
        attack_flat[:] = np.asarray(init_attack, dtype=np.float64)
    if init_defense is not None:
        defense_flat[:] = np.asarray(init_defense, dtype=np.float64)
    if init_delta is not None:
        delta[:] = np.asarray(init_delta, dtype=np.int64)

    # Informativer Prior (Marktwert-Baseline); None ⇒ Nullvektor ⇒ 0-Prior.
    if getattr(L, "init_prior_mean", None) is not None:
        init_prior_mean = np.ascontiguousarray(L.init_prior_mean, dtype=np.float64)
    else:
        init_prior_mean = np.zeros(L.n_teams, dtype=np.float64)

    # Sample-Buffer-Größe vorab bestimmen
    n_post = max(0, n_iter - burnin)
    n_samples_target = (n_post + thin - 1) // thin
    sample_attack = np.zeros((n_samples_target, n_total))
    sample_defense = np.zeros((n_samples_target, n_total))
    sample_delta = np.zeros((n_samples_target, L.n_matches), dtype=np.int64)

    accept_count = 0
    propose_count = 0
    sample_offset = 0

    iterator = range(0, n_iter, chunk)
    if verbose:
        iterator = tqdm(iterator, desc="MCMC",
                        total=(n_iter + chunk - 1) // chunk, unit="chunk")

    for it_start in iterator:
        it_end = min(it_start + chunk, n_iter)
        ac, pc, sample_offset = _run_chunk(
            attack_flat, defense_flat, delta,
            L.team_start, L.team_matches_flat, L.strength_days,
            L.home_idx, L.away_idx,
            L.match_home_local, L.match_away_local,
            L.obs_x, L.obs_y,
            L.c_x, L.c_y, L.gamma, L.epsilon, L.tau, PRIOR_VAR,
            init_prior_mean,
            MAX_GOALS,
            int(L.continuous_obs), L.phi,
            proposal_sd,
            it_start, it_end,
            burnin, thin,
            sample_attack, sample_defense, sample_delta,
            sample_offset,
        )
        accept_count += ac
        propose_count += pc

        if progress_queue is not None:
            progress_queue.put(it_end - it_start)

    a_dicts, d_dicts, delta_list = _flat_samples_to_dicts(
        L, sample_attack, sample_defense, sample_delta, sample_offset,
    )

    return {
        "attack": a_dicts,
        "defense": d_dicts,
        "delta": delta_list,
        "acceptance": float(accept_count) / max(1, propose_count),
    }


def posterior_means(samples) -> tuple[dict, dict, np.ndarray]:
    """Posteriori-Mittelwerte (identisch zu v1)."""
    team_ids = list(samples["attack"][0].keys())

    mean_attack: dict = {}
    mean_defense: dict = {}
    for team in team_ids:
        a_stack = np.stack([s[team] for s in samples["attack"]])
        d_stack = np.stack([s[team] for s in samples["defense"]])
        mean_attack[team] = a_stack.mean(axis=0)
        mean_defense[team] = d_stack.mean(axis=0)

    delta_stack = np.stack(samples["delta"])
    p_outlier = delta_stack.mean(axis=0)
    return mean_attack, mean_defense, p_outlier


def warmup_jit():
    """
    Triggert die JIT-Kompilation aller @njit-Funktionen mit Dummy-Eingaben.

    Sinnvoll vor dem Start paralleler Worker, damit jeder Sub-Prozess
    den AOT-Cache (cache=True) statt einer Neukompilation nutzt.
    """
    # Minimaler Dummy-State: 2 Teams, 1 Spiel
    attack_flat = np.zeros(2, dtype=np.float64)
    defense_flat = np.zeros(2, dtype=np.float64)
    delta = np.zeros(1, dtype=np.int64)
    team_start = np.array([0, 1, 2], dtype=np.int64)
    team_matches_flat = np.array([0, 0], dtype=np.int64)
    strength_days = np.array([0, 0], dtype=np.int64)
    home_idx = np.array([0], dtype=np.int64)
    away_idx = np.array([1], dtype=np.int64)
    match_home_local = np.array([0], dtype=np.int64)
    match_away_local = np.array([0], dtype=np.int64)
    obs_x = np.array([1], dtype=np.int64)
    obs_y = np.array([0], dtype=np.int64)
    obs_x_f = np.array([1.3], dtype=np.float64)
    obs_y_f = np.array([0.4], dtype=np.float64)
    sa = np.zeros((1, 2))
    sd = np.zeros((1, 2))
    sdt = np.zeros((1, 1), dtype=np.int64)
    ipm = np.zeros(2, dtype=np.float64)   # init_prior_mean: 2 Dummy-Teams
    # Beide Branch-Spezialisierungen kompilieren: diskret (int64-obs,
    # continuous=0) und kontinuierlich (float64-obs, continuous=1).
    _run_chunk(
        attack_flat, defense_flat, delta,
        team_start, team_matches_flat, strength_days,
        home_idx, away_idx, match_home_local, match_away_local,
        obs_x, obs_y,
        0.5, 0.1, 0.1, 0.2, 100.0, 1.0 / 37.0,
        ipm,
        5, 0, 5.0, 0.05, 0, 1, 0, 1, sa, sd, sdt, 0,
    )
    _run_chunk(
        attack_flat, defense_flat, delta,
        team_start, team_matches_flat, strength_days,
        home_idx, away_idx, match_home_local, match_away_local,
        obs_x_f, obs_y_f,
        0.5, 0.1, 0.1, 0.2, 100.0, 1.0 / 37.0,
        ipm,
        5, 1, 5.0, 0.05, 0, 1, 0, 1, sa, sd, sdt, 0,
    )
