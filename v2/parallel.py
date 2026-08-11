"""
parallel.py
===========
Mehrere MCMC-Ketten parallel laufen lassen.

v2-Änderung: Vor dem Spawn der Worker wird der Numba-JIT im Hauptprozess
einmal "warmgelaufen" (cache=True), damit jeder Sub-Prozess den
kompilierten Code aus dem Cache lädt statt selbst zu kompilieren.
"""

from __future__ import annotations

import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Empty

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        class _Dummy:
            def update(self, n=1): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return _Dummy()

from mcmc import run_mcmc, warmup_jit


def _run_one_chain(args):
    L, n_iter, burnin, thin, proposal_sd, seed, progress_queue = args
    return run_mcmc(L, n_iter=n_iter, burnin=burnin, thin=thin,
                    proposal_sd=proposal_sd, seed=seed, verbose=False,
                    progress_queue=progress_queue)


def gelman_rubin(chains_values: list[np.ndarray]) -> float:
    """R-hat nach Gelman-Rubin für eine skalare Größe über mehrere Ketten."""
    n = min(len(c) for c in chains_values)
    chains = np.stack([c[:n] for c in chains_values])

    chain_means = chains.mean(axis=1)
    W = float(np.mean([np.var(c, ddof=1) for c in chains]))
    if W < 1e-10:
        return 1.0
    B = n * float(np.var(chain_means, ddof=1))
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def run_mcmc_parallel(L, n_chains: int = 4, n_iter_per_chain: int = 5000,
                      burnin: int = 1000, thin: int = 5,
                      proposal_sd: float = 0.05, base_seed: int = 42) -> dict:
    """
    Lässt n_chains MCMC-Ketten parallel laufen und kombiniert die Samples.
    """
    # JIT pre-compile im Hauptprozess → cache=True wird genutzt
    print("  Numba-JIT vorkompilieren ...", end="", flush=True)
    warmup_jit()
    print(" OK")

    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()

    args_list = [
        (L, n_iter_per_chain, burnin, thin, proposal_sd, base_seed + i, progress_queue)
        for i in range(n_chains)
    ]

    total_iters = n_chains * n_iter_per_chain
    chains = [None] * n_chains
    stop_reader = threading.Event()

    def _reader(pbar):
        while not stop_reader.is_set():
            try:
                n = progress_queue.get(timeout=0.2)
                pbar.update(n)
            except Empty:
                pass

    print(f"  Starte {n_chains} parallele Ketten à {n_iter_per_chain} Iterationen ...")
    with tqdm(total=total_iters, desc="MCMC", unit="iter",
              unit_scale=True, smoothing=0.1) as pbar:
        reader_thread = threading.Thread(target=_reader, args=(pbar,), daemon=True)
        reader_thread.start()

        with ProcessPoolExecutor(max_workers=n_chains) as executor:
            futures = {executor.submit(_run_one_chain, args): i
                       for i, args in enumerate(args_list)}
            for future in as_completed(futures):
                chains[futures[future]] = future.result()

        stop_reader.set()
        reader_thread.join()

    combined: dict = {
        "attack": [],
        "defense": [],
        "home_advantage": [],
        "delta": [],
        "chains": chains,
    }
    for r in chains:
        combined["attack"].extend(r["attack"])
        combined["defense"].extend(r["defense"])
        combined["home_advantage"].extend(r["home_advantage"])
        combined["delta"].extend(r["delta"])
    combined["acceptance"] = float(np.mean([r["acceptance"] for r in chains]))

    rhat_values = []
    for team in range(len(L.teams)):
        for key in ("attack", "defense"):
            series = [
                np.array([s[team].mean() for s in r[key]])
                for r in chains
            ]
            rhat_values.append(gelman_rubin(series))
        if L.use_team_home_advantage:
            series = [
                np.array([sample[team] for sample in r["home_advantage"]])
                for r in chains
            ]
            rhat_values.append(gelman_rubin(series))

    combined["rhat_max"] = float(np.max(rhat_values))
    combined["rhat_mean"] = float(np.mean(rhat_values))

    status = "OK" if combined["rhat_max"] < 1.05 else "nicht konvergiert!"
    print(f"  {len(combined['attack'])} Samples, "
          f"Akzeptanz {combined['acceptance']:.1%}, "
          f"R-hat max {combined['rhat_max']:.3f} ({status})")
    return combined
