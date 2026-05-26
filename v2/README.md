# Rue-Salvesen-Modell auf Bundesligadaten 2000–2026

Vollständige Implementierung des Bayes'schen dynamischen Modells von
**Rue & Salvesen (2000)** mit historischen Bundesligaspielen.

## Was du bekommst

Eine modulare Python-Pipeline, die

1. die Bundesligaspiele 2000–2026 von [football-data.co.uk](https://www.football-data.co.uk/) lädt,
2. das Modell auf eine ausgewählte Saison anwendet (MCMC-Inferenz),
3. zehn statische Plots und drei Animationen erzeugt — alles für deine Präsentation.

## Schnellstart

```bash
# 1. Abhängigkeiten installieren (am besten in venv)
python -m venv venv
source venv/bin/activate         # auf Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Pipeline starten
python run.py
```

Beim ersten Lauf werden die Bundesligadaten heruntergeladen (~5 MB, cached in `data_cache/`).
Der MCMC-Lauf braucht je nach Rechner 5–20 Minuten. Das Ergebnis wird in `mcmc_cache.pkl` zwischengespeichert, sodass Re-Runs der Visualisierungen sofort gehen.

Alle Plots landen im Ordner `output/`.

## Konfiguration

In `run.py` ganz oben:

```python
START_YEAR = 2000
END_YEAR   = 2026
ANALYSIS_SEASON = "2023/24"   # welche Saison MCMC-analysiert wird

N_ITER       = 3000           # MCMC-Iterationen
BURNIN       = 500
THIN         = 5
PROPOSAL_SD  = 0.05
```

Für eine schnelle Test-Pipeline: `N_ITER = 500`, `BURNIN = 100`. Für Präsentations-Qualität: `N_ITER = 10000` oder mehr.

## Was die Pipeline produziert

### Phase 1–2: Daten
- `01_tor_verteilung.png` — Histogramm der Heim-/Auswärtstore (wie Fig. 1 im Paper)
- `02_heimvorteil_2000_2026.png` — Heimvorteil-Trends über 26 Saisons

### Phase 3: Modell-Bausteine
- `03_brownsche_bewegung.png` — Beispielpfade des Zeitmodells
- `04_tormodell_aufbau.png` — Naives Poisson → Dixon-Coles → Mischung, Schritt für Schritt
- `05_dag_struktur.png` — Modellstruktur als DAG (wie Fig. 2)

### Phase 4–5: MCMC-Ergebnisse
- `06_mcmc_trace.png` — Konvergenz-Diagnostik
- `07_staerken_evolution.png` — Angriffs- und Abwehrstärken der Top-Teams über die Saison (wie Fig. 6)
- `08_team_profile.png` — Scatterplot Angriff vs. Abwehr aller Teams
- `09_rangverteilung.png` — Posterior-Erwartung für Endrang vs. tatsächliche Rangliste (wie Fig. 5)
- `10_ueberraschungs_spiele.png` — Top 10 unerklärbarste Spiele der Saison (wie Tab. 2)

### Phase 6: Animationen
- `anim_01_staerken.gif` — Stärken wachsen über die Saison
- `anim_02_tabelle.gif` — Tabellenbewegung Spieltag für Spieltag
- `anim_03_mcmc.gif` — MCMC-Wanderung der Stärken eines Teams

## Dateistruktur

```
bundesliga_pipeline/
├── run.py             # Hauptpipeline
├── data.py            # Datenlader (mit Cache + Fallback)
├── model.py           # Tormodell, Lambdas, Likelihood
├── mcmc.py            # Metropolis-Hastings-Sampler
├── viz.py             # Statische Plots
├── animations.py      # GIF-Animationen
├── requirements.txt
└── README.md
```

## Was tut welche Datei?

**`data.py`** — Lädt jede Saison einzeln, normalisiert Spaltennamen, cached lokal. Falls eine Saison nicht ladbar ist, generiert synthetische Daten als Fallback.

**`model.py`** — Implementiert das Tormodell aus dem Paper:
- `compute_lambdas(...)`: Berechnet (λ_x, λ_y) aus Teamstärken inkl. psychologischem Effekt γ
- `trunc_poisson_pmf(...)`: Trunkierte Poisson bei max. 5 Toren
- `dc_correction(...)`: Dixon-Coles-Korrektur κ für 0:0, 1:1, 1:0, 0:1
- `match_likelihood(...)`: Vollständige Mischungs-Likelihood π_g

**`mcmc.py`** — Single-Site-Metropolis-Sampler. Aktualisiert pro Iteration:
- Alle Angriffs- und Abwehrstärken einzeln
- Alle Bernoulli-Indikatoren δ für die Mischung
- Akzeptanzrate sollte um 30–60% liegen

**`viz.py`** und **`animations.py`** — Plotting-Code, hauptsächlich Matplotlib.

## Tipps für die Präsentation

1. **Foliendrehbuch passend zur Pipeline**: Phase 1 → Folie "Daten", Phase 3 → Folie "Modellzutaten", Phase 4 → Folie "MCMC", Phase 5 → Ergebnisfolien.
2. **Animationen direkt einbetten** (PowerPoint kann GIFs).
3. **Bei Live-Demo**: Vorher einmal laufen lassen und Cache-Datei mitbringen, dann sind die Plots in Sekunden da.
4. **Falls Zeitdruck**: `04_tormodell_aufbau.png` und `09_rangverteilung.png` sind die "Wow"-Plots — die zeigen am besten, was das Modell wirklich macht.

## Bekannte Einschränkungen

- MCMC ist in reinem Python implementiert; auf großen Saisons ist es langsam. Für Produktionscode würde man Numba / Cython / JAX nutzen.
- Pre-Promotion-Teams (aufsteigende Mannschaften) bekommen den Standard-Prior; mit Vor-Saison-Daten könnte man hier besser starten.
- `EPSILON = 0.2`, `TAU = 100`, `GAMMA = 0.1` sind die Paper-Werte. Für eine andere Liga wären eigene Schätzungen sinnvoll (Grid Search über die Pseudolikelihood, siehe Paper §3.2).

## Quelle

Rue, H., & Salvesen, Ø. (2000). *Prediction and retrospective analysis of soccer matches in a league.* The Statistician, 49(3), 399–418.
