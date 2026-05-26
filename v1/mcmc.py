"""
mcmc.py
=======
MCMC-Sampler (Metropolis-Hastings) für das Rue-Salvesen-Modell.

Zustand:
  - attack[team]:  Array Länge n_matches[team] mit Angriffsstärken
  - defense[team]: Array Länge n_matches[team] mit Abwehrstärken
  - delta[match]:  Bernoulli-Indikator (Mixture: 1 = "Durchschnittsspiel")

Wir machen Single-Site-Metropolis: in jeder Iteration werden alle Variablen
einzeln in zufälliger Reihenfolge aktualisiert. Für jede Stärke werden:
  - Time-Prior-Beiträge (links + rechts in der Brownschen Kette)
  - Likelihood des betroffenen Spiels
neu bewertet.

Beschleunigung
--------------
Der innere Schleifenkern (``_run_chunk`` + Hilfsfunktionen) ist mit Numba
``@njit`` JIT-kompiliert (typischer Speedup 50–100×). Der **Algorithmus ist
unverändert** gegenüber der reinen Python-Variante: identische Proposals
(Gauß-Random-Walk mit ``proposal_sd``), identische Time-Priors, identische
Mixture-Indikator-Updates, identische Akzeptanzregel und identisches
Level-Drift-Entfernen. Es werden dieselben Parameter geschätzt.

Für den Kern werden die Zustands-Variablen (attack, defense) als *flache*
numpy-Arrays geführt, segmentiert durch ``team_start``. Das Mapping zurück
auf die dict-API von League erfolgt erst beim Speichern der Samples — die
öffentliche Schnittstelle (``run_mcmc``, ``posterior_means``) ist unverändert.

Hinweis zum Seed: Numbas interner RNG wird durch ein Python-seitiges
``np.random.seed`` *nicht* gesetzt. Daher seedet ``run_mcmc`` den Kern über
die kompilierte Hilfsfunktion ``_seed`` — so bleibt die ursprüngliche
Reproduzierbarkeit pro ``seed`` (und die Unabhängigkeit paralleler Ketten)
erhalten.
"""

from __future__ import annotations

from math import lgamma

import numpy as np
from numba import njit

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # Fallback ohne Fortschrittsbalken
        return iterable

from model import (
    League, TAU, GAMMA, EPSILON, PRIOR_VAR, MAX_GOALS,
)


# log(k!) für k=0..MAX_GOALS, als Modul-Konstante (von den @njit-Funktionen
# eingefroren). Identische Werte wie in model.py.
_LOG_FACT = np.array([lgamma(k + 1) for k in range(MAX_GOALS + 1)],
                     dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────
# Numba-kompilierte Hilfsfunktionen (1:1-Port der Python-Logik)
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _poisson_pmf(k, lam):
    if lam <= 0.0:
        return 0.0
    return np.exp(k * np.log(lam) - lam - _LOG_FACT[k])


@njit(cache=True, fastmath=True)
def _poisson_tail(max_k, lam):
    """P(X >= max_k) — direkt aus PMFs aufsummiert (wie model._poisson_tail)."""
    if lam <= 0.0:
        return 1.0 if max_k == 0 else 0.0
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
    """Dixon-Coles-Korrekturfaktor κ."""
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
def _match_loglik(
    m,
    attack_flat, defense_flat,
    team_start, home_idx, away_idx,
    match_home_local, match_away_local,
    home_goals, away_goals,
    c_x, c_y, gamma,
    delta_m,
    lam_x_avg, lam_y_avg,
    max_k,
):
    """Log-Likelihood eines einzelnen Spiels gegeben den Zustand."""
    h = home_idx[m]
    a = away_idx[m]
    lh = match_home_local[m]
    la = match_away_local[m]
    a_h = attack_flat[team_start[h] + lh]
    d_h = defense_flat[team_start[h] + lh]
    a_a = attack_flat[team_start[a] + la]
    d_a = defense_flat[team_start[a] + la]

    # Bei delta=1 ist das Spiel ein Durchschnittsspiel → verwende c_x/c_y
    if delta_m == 1:
        lam_x = lam_x_avg
        lam_y = lam_y_avg
    else:
        lam_x, lam_y = _compute_lambdas(a_h, d_h, a_a, d_a, c_x, c_y, gamma)

    x = home_goals[m]
    y = away_goals[m]
    p = (_trunc_poisson_pmf(x, lam_x, max_k) *
         _trunc_poisson_pmf(y, lam_y, max_k) *
         _dc_correction(x, y, lam_x, lam_y))
    if p < 1e-300:
        p = 1e-300
    return np.log(p)


@njit(cache=True, fastmath=True)
def _time_prior_logpdf(val, prev_val, dt, sigma2, tau):
    """log N(val | prev_val, dt/tau * sigma2)"""
    var = dt / tau * sigma2
    if var < 1e-9:
        var = 1e-9
    diff = val - prev_val
    return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * diff * diff / var


@njit(cache=True, fastmath=True)
def _initial_prior_logpdf(val, sigma2):
    return -0.5 * np.log(2.0 * np.pi * sigma2) - 0.5 * val * val / sigma2


@njit(cache=True)
def _seed(s):
    """Seedet Numbas internen RNG (muss aus jit-Code heraus geschehen)."""
    np.random.seed(s)


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
    home_goals, away_goals,
    # Hyperparameter
    c_x, c_y, gamma, epsilon, tau, sigma2,
    max_k,
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

                    # log-acceptance: priors + likelihood
                    log_alpha = 0.0

                    # Prior aus dem vorherigen Spiel (oder Anfangs-Prior)
                    if local == 0:
                        log_alpha += _initial_prior_logpdf(proposed, sigma2)
                        log_alpha -= _initial_prior_logpdf(current, sigma2)
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

                    # Prior für das nächste Spiel
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

                    # Likelihood-Anteil (nur das betroffene Spiel)
                    old_ll = _match_loglik(
                        m, attack_flat, defense_flat,
                        team_start, home_idx, away_idx,
                        match_home_local, match_away_local,
                        home_goals, away_goals, c_x, c_y, gamma,
                        delta[m], lam_x_avg, lam_y_avg, max_k,
                    )
                    if which == 0:
                        attack_flat[gidx] = proposed
                    else:
                        defense_flat[gidx] = proposed
                    new_ll = _match_loglik(
                        m, attack_flat, defense_flat,
                        team_start, home_idx, away_idx,
                        match_home_local, match_away_local,
                        home_goals, away_goals, c_x, c_y, gamma,
                        delta[m], lam_x_avg, lam_y_avg, max_k,
                    )
                    log_alpha += new_ll - old_ll

                    # Akzeptanz
                    if np.log(np.random.random()) < log_alpha:
                        accept_count += 1
                    else:
                        # ablehnen → zurücksetzen
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
                home_goals, away_goals, c_x, c_y, gamma,
                current, lam_x_avg, lam_y_avg, max_k,
            )
            new_ll = _match_loglik(
                m, attack_flat, defense_flat,
                team_start, home_idx, away_idx,
                match_home_local, match_away_local,
                home_goals, away_goals, c_x, c_y, gamma,
                proposed, lam_x_avg, lam_y_avg, max_k,
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

        # === 4) Snapshot speichern (nach Burn-in) ======================
        if it >= burnin and (it - burnin) % thin == 0:
            for i in range(n_total):
                sample_attack[n_saved, i] = attack_flat[i]
                sample_defense[n_saved, i] = defense_flat[i]
            for m in range(n_matches):
                sample_delta[n_saved, m] = delta[m]
            n_saved += 1

    return accept_count, propose_count, n_saved


# ─────────────────────────────────────────────────────────────────────
# Flache Arrays aus der (v1-)League ableiten
# ─────────────────────────────────────────────────────────────────────

def _build_flat(L: League):
    """Leitet die Numba-freundlichen flachen Arrays aus der League ab.

    Rein strukturell — verändert weder Modell noch Daten; entspricht der
    chronologischen Spiele-Buchhaltung in ``L.team_matches`` / ``L.local_idx``.
    """
    n_teams = len(L.teams)
    n_matches = len(L.home_goals)

    n_local = np.array([len(L.team_matches[t]) for t in range(n_teams)],
                       dtype=np.int64)
    team_start = np.zeros(n_teams + 1, dtype=np.int64)
    team_start[1:] = np.cumsum(n_local)
    total = int(team_start[-1])

    team_matches_flat = np.zeros(total, dtype=np.int64)
    strength_days = np.zeros(total, dtype=np.int64)
    for t in range(n_teams):
        s = team_start[t]
        for k, gm in enumerate(L.team_matches[t]):
            team_matches_flat[s + k] = gm
            strength_days[s + k] = L.match_days[gm]

    match_home_local = np.zeros(n_matches, dtype=np.int64)
    match_away_local = np.zeros(n_matches, dtype=np.int64)
    for m in range(n_matches):
        match_home_local[m] = L.local_idx[(int(L.home_idx[m]), m)]
        match_away_local[m] = L.local_idx[(int(L.away_idx[m]), m)]

    return (team_start, team_matches_flat, strength_days,
            match_home_local, match_away_local)


def _flat_samples_to_dicts(L, sample_attack, sample_defense, sample_delta,
                           team_start, n_saved):
    """Konvertiert flache Sample-Arrays in das ursprüngliche Dict-Format."""
    n_teams = len(L.teams)
    out_a, out_d, out_delta = [], [], []
    for s in range(n_saved):
        ad: dict = {}
        dd: dict = {}
        for t in range(n_teams):
            start = int(team_start[t])
            end = int(team_start[t + 1])
            ad[t] = sample_attack[s, start:end].copy()
            dd[t] = sample_defense[s, start:end].copy()
        out_a.append(ad)
        out_d.append(dd)
        out_delta.append(sample_delta[s].copy())
    return out_a, out_d, out_delta


# ─────────────────────────────────────────────────────────────────────
# Public API (Signatur und Rückgabeformat identisch zur Python-Variante)
# ─────────────────────────────────────────────────────────────────────

def run_mcmc(L: League, n_iter: int = 5000, burnin: int = 1000,
             thin: int = 5, proposal_sd: float = 0.05,
             seed: int = 42, verbose: bool = True, progress_queue=None,
             chunk: int = 200):
    """
    Führt den (Numba-beschleunigten) MCMC-Sampler aus.

    ``chunk`` steuert nur die Paketgröße, in der der kompilierte Kern
    aufgerufen wird (für Fortschritts-Updates); das Ergebnis ist davon
    unabhängig, da Numbas RNG-Zustand prozessweit zwischen Aufrufen erhalten
    bleibt.

    Returns:
        samples: dict mit
            'attack':  Liste von Snapshots, jeder ein dict[team] -> array
            'defense': Liste von Snapshots
            'delta':   Liste von delta-Arrays
            'acceptance': Akzeptanzrate
    """
    # Numbas interner RNG wird von Python-seitigem np.random.seed nicht
    # erfasst → aus jit-Code heraus seeden (erhält Reproduzierbarkeit pro
    # seed und die Unabhängigkeit paralleler Ketten in parallel.py).
    _seed(seed)

    (team_start, team_matches_flat, strength_days,
     match_home_local, match_away_local) = _build_flat(L)

    # Numba-stabile int64-Sichten der Stammdaten
    home_idx = np.ascontiguousarray(L.home_idx, dtype=np.int64)
    away_idx = np.ascontiguousarray(L.away_idx, dtype=np.int64)
    home_goals = np.ascontiguousarray(L.home_goals, dtype=np.int64)
    away_goals = np.ascontiguousarray(L.away_goals, dtype=np.int64)

    n_total = int(team_start[-1])
    n_matches = len(L.home_goals)
    attack_flat = np.zeros(n_total, dtype=np.float64)
    defense_flat = np.zeros(n_total, dtype=np.float64)
    delta = np.zeros(n_matches, dtype=np.int64)

    # Sample-Buffer-Größe vorab bestimmen
    n_post = max(0, n_iter - burnin)
    n_samples_target = (n_post + thin - 1) // thin
    sample_attack = np.zeros((n_samples_target, n_total))
    sample_defense = np.zeros((n_samples_target, n_total))
    sample_delta = np.zeros((n_samples_target, n_matches), dtype=np.int64)

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
            team_start, team_matches_flat, strength_days,
            home_idx, away_idx, match_home_local, match_away_local,
            home_goals, away_goals,
            L.c_x, L.c_y, GAMMA, EPSILON, TAU, PRIOR_VAR,
            MAX_GOALS,
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
        L, sample_attack, sample_defense, sample_delta,
        team_start, sample_offset,
    )

    return {
        "attack": a_dicts,
        "defense": d_dicts,
        "delta": delta_list,
        "acceptance": float(accept_count) / max(1, propose_count),
    }


def posterior_means(samples) -> tuple[dict, dict, np.ndarray]:
    """Berechnet posteriori Mittelwerte aus den MCMC-Samples."""
    n_samples = len(samples["attack"])
    team_ids = list(samples["attack"][0].keys())

    mean_attack = {}
    mean_defense = {}
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
    Triggert die JIT-Kompilation des Kerns mit Dummy-Eingaben.

    Sinnvoll vor dem Start paralleler Worker, damit jeder Sub-Prozess den
    AOT-Cache (cache=True) statt einer Neukompilation nutzt.
    """
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
    home_goals = np.array([1], dtype=np.int64)
    away_goals = np.array([0], dtype=np.int64)
    sa = np.zeros((1, 2))
    sd = np.zeros((1, 2))
    sdt = np.zeros((1, 1), dtype=np.int64)
    _seed(0)
    _run_chunk(
        attack_flat, defense_flat, delta,
        team_start, team_matches_flat, strength_days,
        home_idx, away_idx, match_home_local, match_away_local,
        home_goals, away_goals,
        0.5, 0.1, GAMMA, EPSILON, TAU, PRIOR_VAR,
        MAX_GOALS, 0.05, 0, 1, 0, 1, sa, sd, sdt, 0,
    )
