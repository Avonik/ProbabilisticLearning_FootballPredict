# Betting-Strategien — Methodik und Zwischenstand

**Stand: 2026-08-11.** Analysiert wurde das beste v2-Modell **ohne**
teamspezifischen Heimvorteil (`multiseason_20260811_115310`): echtes
Understat-xG, kontinuierliche Gamma-Likelihood, Marktwert-Prior und ehrliche
Walk-Forward-Prognosen. Das Paper-Modell wurde nicht verwendet.

## Bestimmung der Strategien

- 925 Holdout-Spiele aus 2016/17–2025/26 wurden über `MatchID` mit den
  tatsächlich publizierten Dezimalquoten verbunden.
- Strategiesuche nur auf 2016/17–2021/22 (554 Spiele); zeitlich späterer,
  unangetasteter Test auf 2022/23–2025/26 (371 Spiele).
- ROI verwendet die **rohe Quote inklusive Buchmachermarge** und einen festen
  Einsatz von einer Unit. Normalisierte inverse Quoten werden nur für die
  faire Marktwahrscheinlichkeit verwendet.
- Pro Spiel ist höchstens eine Wette erlaubt: die qualifizierte Auswahl mit
  dem größten Signal.
- Getestete Gruppen: H/D/A, Markt- und Modellfavorit, Übereinstimmung bzw.
  Widerspruch der Favoriten, Quotenbereiche `<2`, `2–3.5`, `>=3.5`.
- Signale:
  - Wahrscheinlichkeitsabstand
    `gap = p_model - p_market_fair`, Schwellen 1–15 Prozentpunkte;
  - modellierter Erwartungswert
    `edge = p_model * decimal_odds - 1`, Schwellen 0–30 %.
- Unsicherheit: 95%-Bootstrap-Intervalle resampeln ganze Matchdays. Zusätzlich
  wurden Saisonverteilung, Quotenabschläge und das Entfernen der größten
  Gewinner geprüft.

Das Skript `analyze_betting_strategies.py` reproduziert Grid Search,
Favoritensegmente und vorab definierte Kontrollregeln.

Als Match-Level-Eingabe dient
[`multiseason_per_match_rps.csv`](output/multiseason_20260811_115310/multiseason_per_match_rps.csv).
Die im Text berichteten festen Regeln und Preisvarianten stehen kompakt in
[`fixed_strategy_summary.csv`](output/betting_strategy_20260811_134057/fixed_strategy_summary.csv).
Das vollständige untersuchte Strategieraster ist in
[`strategy_grid.csv`](output/betting_strategy_20260811_134057/strategy_grid.csv)
abgelegt.

## Vollständiger getesteter Strategieraum

Die Grid Search bildet das Kreuzprodukt der folgenden Dimensionen. Damit sind
alle getesteten Strategien auch ohne Ausführung des Skripts definiert:

| Dimension | Getestete Werte |
|-----------|-----------------|
| Preisstand | `closing`, `opening`, `opening_mean`, `open_close_mid` |
| Auswahlfamilie | alle 11 Familien der nächsten Tabelle |
| Signal | `edge`, `gap` |
| `edge`-Schwellen | 0; 2.5; 5; 7.5; 10; 15; 20; 30 % |
| `gap`-Schwellen | 1; 2; 3; 4; 5; 7.5; 10; 15 Prozentpunkte |
| Einsatz | konstant eine Unit; höchstens eine Auswahl je Spiel |

Das ergibt 4 × 11 × (8 + 8) = **704 mögliche Konfigurationen**. Eine
Konfiguration wurde nur in die Ergebnisauswertung aufgenommen, wenn sie im
Development-Zeitraum mindestens 30 Wetten erzeugte. Dadurch verbleiben
**573 tatsächlich ausgewertete Konfigurationen**. Ihre vollständigen
Kennzahlen stehen zeilenweise in
[`strategy_grid.csv`](output/betting_strategy_20260811_134057/strategy_grid.csv).

| Familie im CSV | Auswahl vor Anwendung der Signal-Schwelle | Ausgewertete Konfigurationen |
|----------------|-------------------------------------------|-----------------------------:|
| `any` | H, D oder A; gewählt wird das stärkste qualifizierte Signal | 60 |
| `H` | nur Heimsieg | 60 |
| `D` | nur Remis | 52 |
| `A` | nur Auswärtssieg | 58 |
| `market_favorite` | Ausgang mit höchster fairer Marktwahrscheinlichkeit | 48 |
| `model_favorite` | Ausgang mit höchster Modellwahrscheinlichkeit | 58 |
| `agree_favorite` | Modell und Markt favorisieren denselben Ausgang | 48 |
| `disagree_model_pick` | Modellfavorit, wenn dieser nicht Marktfavorit ist | 52 |
| `fav_odds<2` | Dezimalquote kleiner als 2.0 | 23 |
| `mid_2-3.5` | Dezimalquote von 2.0 bis kleiner als 3.5 | 54 |
| `dog>=3.5` | Dezimalquote mindestens 3.5 | 60 |

Zusätzlich wurden sieben interpretierbare Regeln unabhängig vom Ranking für
alle vier Preisstände protokolliert. Die 56 Zeilen aus Regel × Preisstand ×
Datensplit stehen in
[`fixed_strategy_summary.csv`](output/betting_strategy_20260811_134057/fixed_strategy_summary.csv).

| Regel im CSV | Familie | Signal | Schwelle |
|--------------|----------|--------|---------:|
| `favorite_gap_04` | `market_favorite` | `gap` | 4 Prozentpunkte |
| `model_favorite_gap_05` | `model_favorite` | `gap` | 5 Prozentpunkte |
| `any_gap_10` | `any` | `gap` | 10 Prozentpunkte |
| `away_gap_10` | `A` | `gap` | 10 Prozentpunkte |
| `underdog_gap_10` | `dog>=3.5` | `gap` | 10 Prozentpunkte |
| `draw_edge_0` | `D` | `edge` | 0 % |
| `draw_edge_025` | `D` | `edge` | 2.5 % |

### Ergebnisüberblick über alle Strategiefamilien

Die folgende Tabelle zeigt für jeden Preisstand und jede der 11 Familien die
ausschließlich anhand des Development-Zeitraums gewählte Regel. Ausgewählt
wurde nicht der höchste rohe ROI, sondern der im Skript definierte,
stichprobengedämpfte `dev_selection_score`, der zusätzlich profitable Saisons
honoriert. Dadurch sind alle getesteten Familien im Bericht vertreten; dies
ist keine Behauptung, dass die jeweils gewählte Regel profitabel ist. Die
ungerundeten Werte und Zusatzmetriken stehen in
[`best_per_family.csv`](output/betting_strategy_20260811_134057/best_per_family.csv).

| Preis | Familie | Signal | Schwelle | Dev n | Dev ROI | Test n | Test ROI |
|-------|----------|--------|---------:|------:|--------:|-------:|---------:|
| Closing | `A` | `gap` | 0.10 | 36 | +5.2 % | 18 | +110.4 % |
| Closing | `D` | `edge` | 0.05 | 229 | +11.2 % | 79 | +2.6 % |
| Closing | `H` | `gap` | 0.075 | 76 | +10.0 % | 46 | −26.5 % |
| Closing | `agree_favorite` | `gap` | 0.04 | 87 | +9.3 % | 84 | −17.3 % |
| Closing | `any` | `gap` | 0.10 | 71 | +9.3 % | 44 | +36.2 % |
| Closing | `disagree_model_pick` | `gap` | 0.05 | 50 | −7.4 % | 27 | −31.7 % |
| Closing | `dog>=3.5` | `gap` | 0.10 | 39 | +43.2 % | 8 | +211.8 % |
| Closing | `fav_odds<2` | `gap` | 0.03 | 33 | −20.6 % | 44 | −8.1 % |
| Closing | `market_favorite` | `gap` | 0.04 | 87 | +9.3 % | 84 | −17.3 % |
| Closing | `mid_2-3.5` | `edge` | 0 | 206 | +6.4 % | 137 | −29.1 % |
| Closing | `model_favorite` | `gap` | 0.05 | 117 | −0.7 % | 97 | −21.6 % |
| Opening | `A` | `edge` | 0.30 | 83 | +24.8 % | 21 | +40.8 % |
| Opening | `D` | `gap` | 0.01 | 293 | +0.1 % | 130 | +29.1 % |
| Opening | `H` | `gap` | 0.10 | 36 | +27.6 % | 25 | +5.0 % |
| Opening | `agree_favorite` | `gap` | 0.075 | 45 | +14.6 % | 46 | −9.4 % |
| Opening | `any` | `gap` | 0.10 | 67 | +17.3 % | 34 | +10.7 % |
| Opening | `disagree_model_pick` | `edge` | 0.20 | 35 | +4.0 % | 10 | +58.0 % |
| Opening | `dog>=3.5` | `gap` | 0.075 | 74 | +17.9 % | 16 | −72.9 % |
| Opening | `fav_odds<2` | `gap` | 0.01 | 64 | −4.7 % | 65 | +3.4 % |
| Opening | `market_favorite` | `gap` | 0.075 | 45 | +14.6 % | 46 | −9.4 % |
| Opening | `mid_2-3.5` | `gap` | 0.10 | 32 | +17.5 % | 20 | +13.9 % |
| Opening | `model_favorite` | `gap` | 0.10 | 45 | +16.7 % | 30 | +11.0 % |
| Opening-Mittel | `A` | `edge` | 0.30 | 74 | +33.5 % | 19 | +19.8 % |
| Opening-Mittel | `D` | `edge` | 0 | 286 | −3.0 % | 127 | +33.7 % |
| Opening-Mittel | `H` | `gap` | 0.10 | 34 | +34.1 % | 25 | +4.5 % |
| Opening-Mittel | `agree_favorite` | `edge` | 0.15 | 34 | +14.3 % | 27 | +10.4 % |
| Opening-Mittel | `any` | `gap` | 0.10 | 67 | +16.4 % | 34 | +10.4 % |
| Opening-Mittel | `disagree_model_pick` | `gap` | 0.01 | 58 | −3.0 % | 29 | +1.4 % |
| Opening-Mittel | `dog>=3.5` | `gap` | 0.10 | 33 | +12.9 % | 4 | +8.3 % |
| Opening-Mittel | `fav_odds<2` | `gap` | 0.01 | 66 | −6.1 % | 67 | +2.2 % |
| Opening-Mittel | `market_favorite` | `edge` | 0.15 | 34 | +14.3 % | 27 | +10.4 % |
| Opening-Mittel | `mid_2-3.5` | `gap` | 0.10 | 32 | +16.7 % | 20 | +13.8 % |
| Opening-Mittel | `model_favorite` | `gap` | 0.10 | 45 | +16.0 % | 30 | +10.7 % |
| Open/Close-Mittel | `A` | `edge` | 0.30 | 82 | +17.9 % | 24 | +40.1 % |
| Open/Close-Mittel | `D` | `edge` | 0 | 304 | +5.6 % | 119 | +27.0 % |
| Open/Close-Mittel | `H` | `gap` | 0.10 | 35 | +9.7 % | 21 | +5.6 % |
| Open/Close-Mittel | `agree_favorite` | `gap` | 0.075 | 37 | +22.4 % | 42 | −17.4 % |
| Open/Close-Mittel | `any` | `gap` | 0.10 | 66 | +20.3 % | 35 | +20.3 % |
| Open/Close-Mittel | `disagree_model_pick` | `gap` | 0.04 | 54 | −10.9 % | 30 | −21.0 % |
| Open/Close-Mittel | `dog>=3.5` | `gap` | 0.10 | 33 | +49.9 % | 4 | +8.7 % |
| Open/Close-Mittel | `fav_odds<2` | `edge` | 0.025 | 37 | −6.9 % | 38 | −9.3 % |
| Open/Close-Mittel | `market_favorite` | `gap` | 0.075 | 37 | +22.4 % | 42 | −17.4 % |
| Open/Close-Mittel | `mid_2-3.5` | `edge` | 0 | 195 | +5.7 % | 134 | −28.9 % |
| Open/Close-Mittel | `model_favorite` | `edge` | 0.10 | 104 | +1.5 % | 72 | −16.8 % |

## Untersuchte Preisstände

| Variante | Definition | Abdeckung |
|----------|------------|----------:|
| `closing` | Pinnacle Closing, dann Markt-Avg Closing, dann weitere Fallbacks | 925/925 |
| `opening` | Pinnacle Opening, sonst Bet365 Opening | 925/925 |
| `opening_mean` | Mittel verfügbarer Pinnacle-/Bet365-Opening-Quoten | 925/925 |
| `open_close_mid` | geometrisches Mittel aus Opening und Closing | 925/925 |

`open_close_mid` ist nur ein Sensitivitätstest. Dieser Mittelwert ist keine
beobachtete, garantiert handelbare Quote zu einem bestimmten Zeitpunkt.

## Wichtigste Resultate

### Favoriten

Favoritenregeln sind nicht stabil. Bei Closing-Quoten erzielte „Marktfavorit,
wenn das Modell mindestens vier Prozentpunkte optimistischer ist" in der
Entwicklung +9.3 % ROI, verlor im späteren Test aber **−17.3 %**. Bei Opening
waren es −1.6 % bzw. **−12.8 %**. Auch Modellfavoriten verloren im Closing-Test
−21.6 %. Damit gibt es keine belastbare Favoritenstrategie.

### Großer Modell–Markt-Abstand

Regel: stärkste Auswahl spielen, wenn `gap >= 0.10`.

| Preis | Entwicklung: n / ROI | späterer Test: n / ROI | Test-95%-KI |
|-------|---------------------:|------------------------:|------------:|
| Closing | 71 / +9.3 % | 44 / +36.2 % | −20.5 bis +90.2 % |
| Opening | 67 / +17.3 % | 34 / +10.7 % | −33.4 bis +58.3 % |
| Opening-Mittel | 67 / +16.4 % | 34 / +10.4 % | −34.2 bis +58.5 % |
| Open/Close-Mittel | 66 / +20.3 % | 35 / +20.3 % | −22.0 bis +60.5 % |

Das Vorzeichen ist über alle Preisvarianten positiv. Die Intervalle sind aber
breit und umfassen null. Beim Closing-Test fällt der ROI nach Entfernen der
drei größten Gewinner von +36.2 % auf −5.9 %.

### Remiswetten

Regel: Remis spielen, wenn `edge >= 0` beziehungsweise mit 2.5%-Puffer.

| Preis | Regel | Entwicklung: n / ROI | späterer Test: n / ROI | Test-95%-KI |
|-------|-------|---------------------:|------------------------:|------------:|
| Closing | edge ≥0 | 324 / +7.7 % | 116 / +32.7 % | −6.5 bis +75.3 % |
| Closing | edge ≥2.5 % | 278 / +10.0 % | 95 / +23.1 % | −19.0 bis +68.7 % |
| Opening | edge ≥0 | 318 / −0.2 % | 138 / +22.8 % | −13.7 bis +62.3 % |
| Opening | edge ≥2.5 % | 269 / −0.7 % | 101 / +50.9 % | +4.0 bis +100.5 % |
| Opening-Mittel | edge ≥0 | 286 / −3.0 % | 127 / +33.7 % | −5.0 bis +73.3 % |
| Open/Close-Mittel | edge ≥0 | 304 / +5.6 % | 119 / +27.0 % | −11.7 bis +67.6 % |

Der Opening-Test mit 2.5%-Puffer ist formal positiv, wurde aber in der
Entwicklung nicht bestätigt. Im späteren Zeitraum traten außerdem mehr Remis
ein, als selbst das Modell erwartete. Das spricht für günstige Ergebnisvarianz
oder einen Zeitregime-Effekt statt eines stabil nachgewiesenen Edges.

### Opening-Auswärtskandidat

Die development-seitig beste Opening-Regel innerhalb der Auswärtsfamilie war
`p_model * opening_odds - 1 >= 0.30`:

- Entwicklung: 83 Wetten, +24.8 % ROI, fünf von sechs Saisons profitabel;
- späterer Test: 21 Wetten, +40.8 % ROI, drei von vier Saisons profitabel;
- Test-95%-KI: −48.0 bis +130.2 %;
- nach Entfernen des größten Testgewinners: +7.3 %, nach den zwei größten:
  −22.7 %.

Die Regel ist wegen mittlerer Quoten um 9–10 und nur 21 Testwetten stark von
wenigen Treffern abhängig. Sie ist eine Forschungshypothese, keine belastbare
Einsatzstrategie.

## Liefert das Modell Zusatzinformation?

Auf den Entwicklungssaisons wurde je Ergebnis eine binäre Kalibrierung
`outcome ~ logit(p_market) + [logit(p_model)-logit(p_market)]` gefittet. Der
Koeffizient der Modellabweichung war für keinen Ausgang signifikant:

| Ausgang | Koeffizient Modellabweichung | p-Wert |
|---------|-----------------------------:|-------:|
| Heim | −0.291 | 0.314 |
| Remis | −0.585 | 0.325 |
| Auswärts | +0.023 | 0.941 |

Eine reine Rekalibrierung der Marktprognosen erzeugte ähnlich gute oder teils
bessere historische Wettresultate als Markt plus v2-Modell. Daher kann der
beobachtete Profit noch nicht überzeugend dem Modell zugeschrieben werden.

## Fazit

Es gibt drei Kandidaten für einen **vorher festgelegten neuen** Test:

1. Remis, wenn `edge >= 2.5 %`;
2. stärkste Auswahl, wenn `gap >= 10 Prozentpunkte`;
3. Opening-Auswärtssieg, wenn `edge >= 30 %` (explorativ, hohe Varianz).

Keine Regel ist bereits als profitabel nachgewiesen. Nach der jetzigen Suche
sind alle zehn Bundesliga-Saisons „gesehen"; eine weitere Optimierung auf
denselben Spielen wäre Data Snooping. Glaubwürdige Bestätigung benötigt neue
Spiele oder einen externen, vorab festgelegten Testdatensatz. Opening-Quoten
sind zudem oft nur mit kleinen Limits verfügbar; tatsächliche Ausführbarkeit,
Quotenbewegung und Einsatzlimits sind im Backtest nicht enthalten.

## Reproduktion

```bash
python analyze_betting_strategies.py \
  output/multiseason_20260811_115310/multiseason_per_match_rps.csv
```

Ergebnisse des dokumentierten Laufs:
`output/betting_strategy_20260811_134057/`:

- [`fixed_strategy_summary.csv`](output/betting_strategy_20260811_134057/fixed_strategy_summary.csv):
  alle im Bericht hervorgehobenen, fest definierten Regeln mit Development-
  und Testergebnis sowie Bootstrap-Intervall;
- [`strategy_grid.csv`](output/betting_strategy_20260811_134057/strategy_grid.csv):
  vollständiges Ergebnisraster aller getesteten Regel-, Preis- und
  Schwellenkombinationen;
- [`best_per_family.csv`](output/betting_strategy_20260811_134057/best_per_family.csv):
  je Strategiefamilie die ausschließlich auf dem Development-Zeitraum
  ausgewählte beste Regel und ihr späteres Testergebnis;
- [`favorite_accuracy_segments.csv`](output/betting_strategy_20260811_134057/favorite_accuracy_segments.csv):
  RPS-Vergleich von Modell und Markt nach Stärke des Marktfavoriten.


----------------------
Kommentar

Die beste und am ehesten belastbare Strategie ist:
Pro Spiel die Auswahl mit dem größten Signal wetten, sobald
p_model − p_market_fair ≥ 10 Prozentpunkte.

Warum diese?
Quote	Development-ROI	Test-ROI	Testwetten
Closing	+9,3 %	+36,2 %	44
Opening	+17,3 %	+10,7 %	34
Opening-Mittel	+16,4 %	+10,4 %	34
Open/Close-Mittel	+20,3 %	+20,3 %	35


Sie ist die einzige überzeugende Kandidatin, die in Development und Test sowie bei allen vier Preisvarianten dasselbe positive Vorzeichen zeigt.
Aber: Noch kein statistisch gesicherter Profit. Die Konfidenzintervalle enthalten null und das Closing-Ergebnis hängt stark von wenigen hohen Gewinnen ab. Deshalb würde ich sie im Bericht als beste Forschungshypothese, nicht als nachgewiesen profitable Strategie bezeichnen.
Die Opening-Auswärtssieg-Regel mit edge ≥ 30 % liefert zwar höhere Einzelwerte, beruht aber nur auf 21 Testwetten und kippt nach Entfernung der zwei größten Gewinne ins Negative. Die Remisstrategie ist ebenfalls schwächer begründet, weil sie im Development bei Opening-Quoten negativ war.