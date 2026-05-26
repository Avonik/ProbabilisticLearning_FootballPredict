# Modell-Historie — Rue & Salvesen Bundesliga-Wettmodell (v2)

**Stand: 2026-05-23.** Dieses Dokument fasst chronologisch alle Schritte
zusammen — vom ursprünglichen, *geleakten* Ergebnis bis zum aktuellen,
leckfreien Walk-Forward gegen die Buchmacher-Schlusslinie. Ziel: jederzeit
einen präzisen Überblick, *warum* die Pipeline so aussieht, wie sie aussieht.

---

## 0. Ausgangslage — das verdächtige „+4 %"

Das Modell (Rue & Salvesen 2000: Team-Stärken Angriff/Abwehr als
Brownian-Motion-Trajektorien, Poisson-Tormodell mit Dixon-Coles-Korrektur und
ε-Mischung) schien die Buchmacher (Pinnacle-Schlussquoten) **um ~4 % auf RPS zu
schlagen**. Da Closing-Odds nahezu effizient sind, war das ein Warnsignal für
einen Leak — und der Anlass für diese ganze Prüfung.

---

## 1. Leak-Diagnose

Gefundene Lecks im alten `compare_to_bookmaker` (in `run.py`):

1. **Look-ahead / Smoothing (Hauptleck).** `run.py` fittet das MCMC auf die
   **ganze** Saison → eine *Glättungs*-Posteriori `p(α_k | y_{1:T})`. Die Stärke
   eines Teams zu Spieltag *k* ist damit durch **zukünftige** Spiele
   mitbestimmt (über den engen Brownian-Zeit-Prior). Wer damit Holdout-Spiele
   „vorhersagt", nutzt Zukunftswissen. Ehrlich nötig wäre die *Filter*-Posteriori
   `p(α_k | y_{1:k-1})`.
2. **xG-β auf Vollsaison** (inkl. Holdout) gefittet.
3. **c_x/c_y-Normalisierung** auf Vollsaison.

**Beweis:** Ein ehrlicher Walk-Forward kehrte das Vorzeichen um — der
Buchmacher war ~9–13 % **besser**, nicht schlechter. Das „+4 %" war reiner
Look-ahead.

---

## 2. Pipeline-Umbau: retrospektiv ≠ Vorhersage

**Aufteilung der Pipeline:**

- **`run.py`** = nur noch RETROSPEKTIV (Voll-Saison-Fit, alle Plots/Animationen).
  Der Buchmacher-Vergleich wurde hier entfernt.
- **`backtest.py`** (neu) = EHRLICHER Walk-Forward / Expanding Window:
  - Für jeden Holdout-Spieltag *d*: Liga nur aus Spielen mit `Datum < d` bauen,
    MCMC fitten, Spiele von *d* aus der **letzten** Stärke je Team (Filter-
    Schätzung am Trainings-Ende) vorhersagen.
  - **Warm-Start:** jeder Refit startet aus den Stärken des Vortags → kurzer
    Burn-in (`mcmc.run_mcmc` bekam `init_attack/init_defense/init_delta`).
  - **xG-β nur auf Trainings-Saisons** (`fit_xg_weights(..., cache=False)`).
  - **c_x/c_y** automatisch aus dem Präfix.

**Ehrliches Ergebnis (gerundetes Proxy-xG, Trunc-Poisson):** Modell verliert
klar — 2025/26 RPS **0.2177** vs. Buchmacher 0.1983 (**−9.8 %**). Modell schlägt
weiter empirische (0.2395) / uniforme (0.2352) Baselines → echtes Können, aber
kein Schlagen der Closing-Odds. Wissenschaftlich der erwartete Befund.

**Konvergenz (Mehrketten + R-hat):** `backtest.py` bekam optionale parallele
Ketten (`USE_MULTI_CHAIN`) mit R-hat-Diagnostik. Befund: Single-Chain ≈
Multi-Chain ≈ Heavy-Budget bei der **Vorhersage** (RPS stabil), aber R-hat der
latenten Team-*Level* bleibt erhöht (~1.2–1.46) — der Single-Site-Sampler
mischt die Level langsam. Punktschätzung vertrauenswürdig; für eine formale
Konvergenzaussage bräuchte es mehr Iterationen / besseren Sampler.

---

## 3. Kontinuierliches xG-Beobachtungsmodell (Gamma statt gerundetem Poisson)

**Problem:** xG wurde auf ganze Zahlen **gerundet** (`model.py`), weil die
Likelihood eine Trunc-Poisson·Dixon-Coles-PMF ist (ganzzahlig). Das injiziert
das Effizienz-Rauschen zurück, das xG entfernen sollte; Poisson erzwingt
zudem Var = Mittelwert.

**Lösung:** xG als *kontinuierliche Messung* der Torrate λ modellieren:
`xG ~ Gamma(shape=φ, rate=φ/λ)`, E[xG]=λ, Var[xG]=λ²/φ.

- **Geändert: nur die Fit-Likelihood.** λ-abhängiger Teil im Kern:
  `ℓ(λ) = −φ·log λ − φ·g/λ` (`mcmc._gamma_loglik_lam`); Dixon-Coles und
  Trunkierung fliegen aus dem **Fit** (Diskret-Artefakte).
- **Unverändert:** latente Stärken, Zeit-Prior, γ-Term, ε-Mischung; **und die
  Vorhersage** bleibt ein **diskretes Poisson(λ)-Gitter** → W/U/N (echte Tore
  sind ganzzahlig). φ taucht in der Vorhersage NICHT auf.
- **Neuer Hebel φ (Präzision):** großes φ ⇒ xG wird stärker vertraut ⇒
  schärfere Stärken-Updates ⇒ weniger Hedging.

**Ergebnis (Proxy-xG, 2025/26):** φ=5 → RPS **0.2146** (4 Ketten), Lücke −8.2 %.
**φ-Sweep** zeigt klares Optimum bei ~5 (φ=2≈0.2177, φ=10≈0.2213, φ=30≈0.2346):
zu hohes φ überfittet das **grobe** Proxy-xG. → Der Gewinn ist durch die
xG-Qualität gedeckelt.

---

## 4. Echtes xG (Understat via `soccerdata`)

**Motivation:** Der φ-Sweep zeigte, dass die Proxy-Qualität (`xg.py`:
β≈0.30·Schüsse-aufs-Tor, r≈0.85 zu echtem xG) der Flaschenhals ist.

**Umsetzung (`real_xg.py`):**
- Echte Match-xG von Understat über `soccerdata` (cached unter `~/soccerdata`).
  Direkt-Scraping ist tot (Understat = JS-SPA, FBref = HTTP 403); `soccerdata`
  ist der funktionierende Weg.
- Gemappt über **(Saison-Startjahr aus dem Spieldatum, Heim, Auswärts)** mit
  fester football-data→Understat-Namenstabelle (`_FD_TO_US`).
  **100 % Abdeckung**: alle 3672 Bundesliga-Spiele ab 2014/15.
- In `backtest.py` per Flag `USE_REAL_XG`: echtes xG überschreibt das Proxy-xG;
  Spiele < 2014/15 behalten den Proxy. Der 2025/26-Walk-Forward fittet nur auf
  dieser Saison → 100 % echtes xG.
- **Kein neues Leck:** xG wird *gemessen* (kein gefitteter Koeffizient), und im
  Walk-Forward fließt das xG eines Spiels nie in seine eigene Vorhersage.

**Ergebnis:**
- **2025/26 (4 Ketten, φ=5):** Modell RPS **0.1973** vs. Buchmacher 0.1983 →
  **+0.5 %** (gleichauf). Sprung von −8.2 % auf 0 allein durch xG-Qualität.
- **Multi-Season (5 Saisons, manuell):** in 4/5 Saisons 0.5–2 % besser, in 1
  Saison 0.5 % schlechter.

---

## 5. Leak-Audit des Echt-xG-Ergebnisses

Weil „Modell ≈ Buchmacher" maximale Skepsis verdient:

- **Strukturargument (stärkste Evidenz):** ein Pipeline-Leak (Look-ahead,
  Merge-Fehler) würde **alle** Varianten gleich aufblasen. Stattdessen wächst
  der Vorsprung mit der xG-Qualität (Tore −9.8 % → Proxy-Gamma −8.2 % → echtes
  xG ~0 %). Das ist die Signatur eines echten Features, nicht eines Lecks.
- **Quoten-Vintage:** 2014/15–2024/25 zu ~100 % **Pinnacle Closing**; 2025/26
  Mix aus Markt-Avg-Closing + Pinnacle-Closing. → scharfe Schlusslinie, keine
  weichen/Eröffnungsquoten.
- **Selektionsbias:** 0 Spiele ohne Quote → nichts herausgefallen.
- **xG-Zuordnung:** corr(xG_home, FTHG)=+0.64 vs. corr(xG_home, FTAG)=−0.16,
  Mittel xG≈Mittel Tore → kein Heim/Auswärts-Swap, richtige Skala.

**Verbleibende, ehrlich offene Punkte:**
1. **xG-Anachronismus:** modernes Understat-xG rückwirkend auf alte Saisons →
   leichter unfairer Vorteil ggü. dem *alten* Markt (kein Look-ahead). Jüngste
   Saisons (2024/25, 2025/26) sind davon frei.
2. **Hyperparameter-Herkunft:** getunte τ/γ/ε dürfen nur aus Saisons VOR der
   Analyse-Saison stammen. → Flag `USE_TUNED_HYPERPARAMS=False` nutzt feste
   Paper-Defaults (τ=100, γ=0.10, ε=0.20), garantiert leckfrei.

**Statistischer Realitätscheck:** „4/5 Saisons besser um 0.5–2 %" ist **kein**
Edge-Beweis. Binomial: P(≥4/5 | echte Gleichheit) = 0.19. Margen 0.5–2 % rel.
≈ 0.001–0.004 absolut, also < 1 gepaarter SE pro Saison. → Die Daten sind
**vereinbar mit exakter Gleichheit**. Ehrliche aktuelle Aussage: **„auf
Augenhöhe mit Closing-Odds — noch kein nachgewiesener Vorsprung."**

---

## 6. Aktuelle Konfiguration (`backtest.py`)

| Flag | Bedeutung |
|------|-----------|
| `USE_XG` | xG (statt roher Tore) füttert die Likelihood |
| `USE_CONTINUOUS_XG` | Gamma-Messmodell statt gerundetem Poisson |
| `PHI` | xG-Präzision (Var=λ²/φ); auf Proxy ≈5 optimal, für echtes xG noch zu sweepen |
| `USE_REAL_XG` | echtes Understat-xG statt Schuss-Proxy |
| `USE_TUNED_HYPERPARAMS` | getunte τ/γ/ε (Cache) **oder** feste Paper-Defaults (leckfrei) |
| `USE_MULTI_CHAIN`, `N_CHAINS` | parallele Ketten + R-hat-Diagnostik |
| `WF_BASE_*`, `WF_WARM_*`, `WF_THIN` | Walk-Forward-MCMC-Budget (kalt/warm) |

**Beteiligte Dateien:** `run.py` (retrospektiv), `backtest.py` (Walk-Forward),
`model.py` (League + Gamma/Poisson-Likelihood + Vorhersage), `mcmc.py`
(Numba-MCMC + Gamma-Kern), `xg.py` (Proxy-xG), `real_xg.py` (echtes xG),
`tune.py` (τ/γ/ε-Tuning), `parallel.py` (Mehrketten + R-hat).

---

## 7. Offene nächste Schritte

1. **Multi-Season-Signifikanzstudie** (der Entscheider): Saisons ≥2014/15
   schleifen, **Per-Spiel-RPS-Differenzen poolen**, gepaarter Wilcoxon-Test +
   Bootstrap-CI; `bm_source`/`xg_source` mitprotokollieren; leckfreie
   Hyperparameter (`USE_TUNED_HYPERPARAMS=False`).
2. **Placebo / Negativ-Kontrolle:** Lauf mit verwürfeltem xG — verschwindet der
   „Vorsprung", sitzt das Signal echt im xG; bleibt er, gibt es einen Leak.
3. **φ-Sweep auf echtem xG** (φ∈{5,15,40}) → Default rechtfertigen/anpassen.
4. **Konvergenz** (optional): R-hat < 1.05 via mehr Iterationen oder einem
   Team-Level-Block-Update im Sampler.
