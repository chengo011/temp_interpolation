# Projekt: Rekonstruktion fehlender Sensordaten mittels Deep Learning und klassischer Interpolation

## 1. Ziel des Projekts

Entwickle ein vollständiges Machine-Learning-Projekt, das fehlende Abschnitte einer meteorologischen Zeitreihe rekonstruiert.

Das zentrale Ziel ist nicht die Vorhersage zukünftiger Werte, sondern die **Interpolation innerhalb einer Datenlücke**.

Beispiel:

Original:

T₁, T₂, T₃, T₄, T₅, T₆, T₇

Beobachtete Daten:

T₁, T₂, ?, ?, ?, T₆, T₇

Das Modell soll die fehlenden Werte T₃, T₄ und T₅ rekonstruieren.

Dabei dürfen sowohl Informationen **vor** als auch **nach** der Lücke verwendet werden. Genau dadurch unterscheidet sich die Aufgabe von normalem Forecasting.

Die wissenschaftliche Hauptfrage lautet:

**Kann ein Deep-Learning-Modell fehlende Sensordaten genauer rekonstruieren als klassische Interpolationsverfahren, insbesondere bei längeren Datenlücken?**

---

# 2. Datensatz

Verwende den Datensatz:

**Jena Climate Dataset 2009–2016**

Der Datensatz basiert auf Wetterdaten des Max-Planck-Instituts für Biogeochemie in Jena. Die verbreitete TensorFlow-Version enthält Messungen im Abstand von 10 Minuten und 14 meteorologische Merkmale.

Primäre Zielvariable:

**Temperatur T (degC)**

Diese Variable soll künstlich entfernt und anschließend rekonstruiert werden.

Zusätzliche Sensorinformationen können als Eingabe verwendet werden, beispielsweise:

- Luftdruck
- relative Luftfeuchtigkeit
- Luftdichte
- Taupunkt
- Windgeschwindigkeit
- Windrichtung
- weitere numerische Wettergrößen des Datensatzes

Das Projekt soll zunächst die Temperatur rekonstruieren. Eine spätere Erweiterung auf mehrere gleichzeitig fehlende Sensorgrößen soll möglich sein.

---

# 3. Grundidee des Experiments

Der Datensatz enthält überwiegend vorhandene Messwerte.

Deshalb werden künstlich Datenlücken erzeugt.

Der entscheidende Vorteil davon ist, dass die tatsächlichen Werte weiterhin bekannt sind.

Aus

T₁, T₂, T₃, T₄, T₅, T₆, T₇

wird beispielsweise

T₁, T₂, ?, ?, ?, T₆, T₇.

Das Modell sieht nur die beschädigte Version.

Nach der Rekonstruktion können die vorhergesagten Werte

T̂₃, T̂₄, T̂₅

direkt mit den ursprünglichen

T₃, T₄, T₅

verglichen werden.

Damit lässt sich objektiv messen, wie gut die Interpolation funktioniert.

---

# 4. Zwei Versuchsszenarien

Implementiere zwei getrennte Experimente.

## Szenario A: Ausfall nur des Temperatursensors

Innerhalb der Datenlücke fehlt ausschließlich die Temperatur.

Andere Messgrößen wie Luftdruck oder Luftfeuchtigkeit bleiben vorhanden.

Beispiel:

Zeitpunkt:

12:00  
12:10  
12:20  
12:30  
12:40

Temperatur:

18.1  
?  
?  
?  
19.0

Luftdruck:

1012  
1012  
1011  
1011  
1011

Luftfeuchtigkeit:

65  
66  
68  
69  
70

Das Deep-Learning-Modell kann dadurch Beziehungen zwischen Temperatur und anderen Sensoren lernen.

Dieses Szenario entspricht beispielsweise dem temporären Ausfall eines einzelnen Temperatursensors.

## Szenario B: Kompletter Sensorausfall

Innerhalb der Datenlücke fehlen alle meteorologischen Messungen.

Das Modell kennt nur:

- Messwerte vor der Lücke
- Messwerte nach der Lücke
- Zeitpunkt der Messung

Dieses Szenario ist schwieriger und entspricht beispielsweise einem vollständigen Ausfall einer Wetterstation.

Szenario A soll das Hauptprojekt darstellen.

Szenario B soll als zusätzliches Experiment implementiert werden.

---

# 5. Aufteilung des Datensatzes

Die Daten dürfen **nicht zufällig aufgeteilt** werden, weil es sich um eine Zeitreihe handelt.

Verwende eine chronologische Aufteilung.

Empfohlene Aufteilung:

**Training:** 2009–2014

**Validation:** 2015

**Test:** 2016

Damit wird verhindert, dass zeitlich benachbarte Messungen gleichzeitig im Trainings- und Testdatensatz vorkommen.

Der Testdatensatz darf während der Entwicklung des Modells nicht zur Auswahl von Hyperparametern verwendet werden.

---

# 6. Datenbereinigung

Vor dem Training muss der Datensatz überprüft und bereinigt werden.

Folgende Schritte sind erforderlich:

1. Zeitstempel korrekt einlesen und chronologisch sortieren.

2. Prüfen, ob Messungen tatsächlich ungefähr alle zehn Minuten vorliegen.

3. Doppelte Zeitstempel entfernen oder eindeutig behandeln.

4. Bereits vorhandene ungültige oder fehlende Werte identifizieren.

5. Physikalisch offensichtlich ungültige Werte behandeln.

Beim Jena-Datensatz gibt es beispielsweise ungültige Windgeschwindigkeitswerte von -9999; auch das offizielle TensorFlow-Tutorial behandelt diese Werte speziell.

Für das Hauptprojekt sollen Zeitabschnitte verwendet werden, für die der ursprüngliche Temperaturwert bekannt ist, weil dieser für die spätere Evaluation benötigt wird.

---

# 7. Feature Engineering

## Temperatur

Die Temperatur ist die primäre Zielgröße.

Während einer künstlich erzeugten Lücke wird ihr Wert aus dem Modelleingang entfernt.

## Zeitinformation

Der reine Zeitstempel sollte nicht als einfache Zahl verwendet werden.

Stattdessen sollen zyklische Features erzeugt werden.

Für Tageszeit:

- Day Sin
- Day Cos

Für Jahreszeit:

- Year Sin
- Year Cos

Damit kann das Modell beispielsweise verstehen, dass 23:50 Uhr zeitlich nahe an 00:10 Uhr liegt.

Die gleiche Vorgehensweise wird auch im TensorFlow-Tutorial für den Jena-Datensatz verwendet.

## Windrichtung

Falls Windrichtung verwendet wird, sollte sie ebenfalls nicht direkt als Winkel zwischen 0° und 360° verwendet werden.

360° und 0° repräsentieren fast dieselbe Richtung.

Windrichtung und Windgeschwindigkeit können deshalb in horizontale und vertikale Komponenten transformiert werden.

## Normalisierung

Alle numerischen Features müssen normalisiert werden.

Für jedes Feature werden Mittelwert und Standardabweichung ausschließlich auf dem Trainingsdatensatz berechnet.

Validation und Test verwenden exakt dieselben Werte.

Informationen aus Validation oder Test dürfen niemals zur Berechnung der Normalisierung verwendet werden.

---

# 8. Erzeugung künstlicher Datenlücken

Dies ist einer der wichtigsten Bestandteile des Projekts.

Aus vollständigen Abschnitten der Zeitreihe sollen Trainingsbeispiele erzeugt werden.

Verwende verschiedene Lückenlängen.

Da ein Zeitschritt zehn Minuten entspricht, sollen mindestens folgende Längen untersucht werden:

| Schritte | Dauer |
|---:|---:|
| 1 | 10 Minuten |
| 3 | 30 Minuten |
| 6 | 1 Stunde |
| 12 | 2 Stunden |
| 24 | 4 Stunden |
| 36 | 6 Stunden |

Optional kann zusätzlich eine Lücke von zwölf oder 24 Stunden getestet werden.

Während des Trainings soll die Lückenlänge zufällig aus mehreren möglichen Längen gewählt werden.

Damit lernt das Modell nicht nur eine einzige feste Lückengröße.

---

# 9. Eingabefenster

Jedes Trainingsbeispiel soll einen ausreichend großen Zeitraum um die Lücke herum enthalten.

Empfehlung:

Mindestens **12 Stunden Kontext vor der Lücke** und **12 Stunden Kontext nach der Lücke**.

Bei Messungen alle zehn Minuten entspricht dies jeweils 72 Messpunkten.

Das Modell bekommt damit:

Vergangenheit → Datenlücke → Zukunft

und rekonstruiert den mittleren Bereich.

Für sehr lange Lücken kann das Kontextfenster später auf jeweils 24 Stunden erweitert werden.

---

# 10. Missing-Mask

Das Modell muss explizit erkennen können, welche Werte tatsächlich fehlen.

Deshalb wird zusätzlich zu den normalen Features eine binäre Maskierungsvariable verwendet.

Beispielsweise:

1 = Wert ist vorhanden

0 = Wert fehlt

Ein fehlender Temperaturwert darf nicht einfach durch 0 °C dargestellt werden, weil das Modell sonst nicht unterscheiden könnte zwischen

„Temperatur beträgt tatsächlich 0 °C“

und

„Temperatur fehlt“.

Nach der Normalisierung kann der fehlende Temperaturwert beispielsweise auf den normalisierten Mittelwert gesetzt werden. Die zusätzliche Maske informiert das Modell darüber, dass dieser Wert nicht beobachtet wurde.

Die Maskierung ist zwingend notwendig.

---

# 11. Klassische Vergleichsverfahren

Das Deep-Learning-Modell darf nicht isoliert bewertet werden.

Implementiere mindestens folgende klassische Methoden:

### Lineare Interpolation

Verbindet die Messwerte vor und nach der Lücke linear.

Dies ist die wichtigste Baseline.

### Kubische Spline-Interpolation

Verwendet eine glatte Kurve durch mehrere umliegende Messpunkte.

### PCHIP oder vergleichbare formtreue kubische Interpolation

Optional, aber empfehlenswert.

Sie reduziert bestimmte Überschwinger, die bei normalen kubischen Splines auftreten können.

### Forward Fill

Der letzte bekannte Wert wird weitergeführt.

Dies ist keine echte Interpolation, aber eine nützliche einfache Kontrollbaseline.

Die zentrale Gegenüberstellung lautet anschließend:

Linear  
vs. Spline  
vs. Deep Learning.

---

# 12. Deep-Learning-Hauptmodell

Das Hauptmodell soll ein **bidirektionales LSTM-Rekonstruktionsmodell** sein.

Ein bidirektionales Modell ist hier sinnvoll, weil die Aufgabe ausdrücklich Interpolation ist.

Das Modell darf sowohl Informationen vor als auch nach der Datenlücke verwenden.

## Vorgeschlagene Architektur

Eingabe:

Sequenz aus allen Features inklusive Missing-Mask.

Danach:

**Bidirectional LSTM, etwa 64 Einheiten**

→

Dropout, ungefähr 0,2

→

**zweites Bidirectional LSTM, etwa 32 Einheiten**

→

kleine vollständig verbundene Schicht

→

eine Ausgabeeinheit pro Zeitpunkt

→

rekonstruierte Temperatur

Das Modell soll für jeden Zeitpunkt des Fensters eine Temperatur ausgeben.

Relevant für den Fehler sind aber ausschließlich die künstlich maskierten Positionen.

---

# 13. Warum bidirektionales LSTM?

Angenommen:

18.1 → 18.3 → ? → ? → ? → 19.4 → 19.6

Ein normales LSTM betrachtet hauptsächlich:

18.1 → 18.3

und versucht daraus die Zukunft vorherzusagen.

Für Interpolation ist das unnötig eingeschränkt.

Das bidirektionale Modell kann gleichzeitig erkennen:

Vor der Lücke steigt die Temperatur.

Nach der Lücke liegt sie bereits bei ungefähr 19.4 °C.

Daraus kann es einen plausiblen Verlauf innerhalb der Lücke rekonstruieren.

---

# 14. Loss-Funktion

Ein extrem wichtiger Punkt:

Der Trainingsfehler darf hauptsächlich beziehungsweise ausschließlich für die **künstlich entfernten Werte** berechnet werden.

Ansonsten könnte das Modell einen niedrigen Fehler erreichen, indem es vorhandene Werte einfach kopiert.

Für jeden maskierten Punkt wird verglichen:

wahre Temperatur

gegen

rekonstruierte Temperatur.

Als Haupt-Loss kann Mean Squared Error verwendet werden.

Alternativ kann Huber Loss getestet werden, weil dieser weniger empfindlich gegenüber einzelnen starken Ausreißern ist.

Die Evaluation erfolgt trotzdem mit leicht interpretierbaren Metriken.

---

# 15. Training

Empfohlene Grundeinstellungen:

- Optimierer: Adam
- Batchgröße: ungefähr 64 oder 128
- maximal etwa 50–100 Epochen
- Early Stopping
- Modell mit bestem Validation-Fehler speichern
- Lernrate reduzieren, wenn sich der Validation-Fehler längere Zeit nicht verbessert
- reproduzierbare Random Seeds verwenden

Das Modell soll nicht einfach die letzte Epoche speichern, sondern den Zustand mit der besten Validation-Performance.

---

# 16. Evaluation

Die wichtigste Metrik ist:

## MAE – Mean Absolute Error

Der Fehler wird in Grad Celsius angegeben.

Beispiel:

MAE = 0,31 °C

ist unmittelbar verständlich.

Zusätzlich:

## RMSE – Root Mean Squared Error

RMSE bestraft größere Fehler stärker.

Beide Werte müssen nach Rücktransformation in die ursprüngliche Celsius-Skala berechnet werden.

---

# 17. Evaluation nach Lückenlänge

Die Ergebnisse dürfen nicht nur als eine einzige Zahl präsentiert werden.

Für jede Lückenlänge müssen separate Ergebnisse berechnet werden.

Beispieltabelle:

| Methode | 10 min | 30 min | 1 h | 2 h | 4 h | 6 h |
|---|---:|---:|---:|---:|---:|---:|
| Linear | MAE | MAE | MAE | MAE | MAE | MAE |
| Spline | MAE | MAE | MAE | MAE | MAE | MAE |
| BiLSTM | MAE | MAE | MAE | MAE | MAE | MAE |

Die interessanteste Frage lautet:

**Wie verändert sich der relative Vorteil von Deep Learning mit wachsender Lückenlänge?**

Bei sehr kleinen Lücken könnte lineare Interpolation bereits extrem gut sein.

Der eigentliche Vorteil des neuronalen Netzes könnte erst bei längeren oder komplexeren Abschnitten sichtbar werden.

---

# 18. Faire Testbedingungen

Für alle Methoden müssen exakt dieselben Testlücken verwendet werden.

Erzeuge daher einmal eine feste Testmenge aus künstlichen Lücken und speichere deren Positionen.

Danach werden:

- lineare Interpolation
- Spline
- BiLSTM
- weitere Modelle

auf exakt denselben Beispielen getestet.

Sonst wäre der Vergleich wissenschaftlich nicht sauber.

---

# 19. Wichtige Zusatzexperimente

Nach dem Hauptmodell sollen mehrere kontrollierte Experimente durchgeführt werden.

## Experiment 1: Einfluss der Lückenlänge

Untersuche, wie stark der Fehler mit steigender Lückenlänge wächst.

## Experiment 2: Nur Temperatur vs. mehrere Sensoren

Trainiere Variante A nur mit:

- Temperatur
- Zeit
- Missing-Mask

Trainiere Variante B zusätzlich mit:

- Luftdruck
- Luftfeuchtigkeit
- weiteren Wettervariablen

Damit lässt sich untersuchen, ob zusätzliche Sensoren die Rekonstruktion verbessern.

## Experiment 3: Einzelner Sensorausfall vs. komplette Stationslücke

Vergleiche Szenario A und Szenario B.

## Experiment 4: Kontextlänge

Vergleiche beispielsweise:

- 3 Stunden Kontext pro Seite
- 6 Stunden
- 12 Stunden
- 24 Stunden

Damit lässt sich herausfinden, wie viel Umgebung das Modell tatsächlich benötigt.

---

# 20. Optionales zweites Deep-Learning-Modell

Wenn das Hauptprojekt erfolgreich funktioniert, implementiere zusätzlich einen kleinen **Transformer Encoder**.

Der Transformer erhält dieselben Eingaben und dieselbe Missing-Mask wie das BiLSTM.

Das erzeugt einen interessanten Vergleich:

Linear  
vs. Spline  
vs. BiLSTM  
vs. Transformer.

Der Transformer ist aber eine Erweiterung.

Das Projekt muss bereits mit dem BiLSTM vollständig funktionieren.

---

# 21. Visualisierungen

Das fertige Projekt muss automatisch mehrere Diagramme erzeugen.

## Rekonstruktionsbeispiel

Zeige einen Zeitabschnitt mit:

- tatsächlicher Temperatur
- sichtbaren Messpunkten
- künstlicher Datenlücke
- linearer Interpolation
- Spline
- Deep-Learning-Rekonstruktion

Die Datenlücke sollte im Diagramm optisch markiert werden.

## Fehler gegen Lückenlänge

x-Achse:

Lückenlänge

y-Achse:

MAE in °C

Eine Linie pro Methode.

Dieses Diagramm ist wahrscheinlich die wichtigste Darstellung des gesamten Projekts.

## Vergleich der Modelle

Balkendiagramm des durchschnittlichen Test-MAE.

## Fehlerverteilung

Optional:

Histogramm oder Boxplot der absoluten Fehler.

---

# 22. Beispiel für das gewünschte Ergebnis

Das Projekt soll am Ende Aussagen dieser Form ermöglichen:

„Bei Lücken von zehn Minuten unterscheiden sich lineare Interpolation und das neuronale Netz kaum.“

„Bei zwei Stunden großen Lücken erzielt das BiLSTM einen um X % niedrigeren MAE.“

„Zusätzliche Wettervariablen verbessern die Rekonstruktion insbesondere bei längeren Datenlücken.“

Ob diese Aussagen tatsächlich zutreffen, muss selbstverständlich aus den Experimenten hervorgehen und darf nicht vorher angenommen werden.

---

# 23. Keine Datenlecks

Besonders auf folgende Fehler achten:

Der Testdatensatz darf nicht während des Trainings verwendet werden.

Normalisierungsparameter dürfen ausschließlich aus Trainingsdaten berechnet werden.

Überlappende Zeitfenster aus demselben Zeitraum dürfen nicht auf Training und Test verteilt werden.

Künstlich entfernte Zielwerte dürfen nicht unbeabsichtigt in einem Eingabefeature enthalten sein.

Bei der Generierung einer Lücke muss kontrolliert werden, dass die Originalwerte separat als Ground Truth gespeichert bleiben.

---

# 24. Projektstruktur

Erstelle ein übersichtlich strukturiertes Projekt mit getrennten Bereichen für:

- Datendownload
- Datenbereinigung
- Feature Engineering
- Gap Generation
- klassische Interpolation
- Deep-Learning-Modell
- Training
- Evaluation
- Visualisierung
- gespeicherte Modelle
- Ergebnisse
- Konfiguration
- Tests

Trainingslogik, Modelldefinition und Datenauswertung sollen nicht in einer einzigen großen Datei vermischt werden.

---

# 25. Konfigurierbarkeit

Folgende Parameter sollen zentral konfigurierbar sein:

- verwendete Features
- Zielvariable
- Trainingszeitraum
- Validation-Zeitraum
- Testzeitraum
- Lückenlängen
- Kontextlänge
- Batchgröße
- Anzahl Epochen
- Learning Rate
- LSTM-Größe
- Dropout
- Random Seed
- Szenario A oder B

Dadurch sollen Experimente ohne Änderungen der eigentlichen Programmlogik durchgeführt werden können.

---

# 26. Reproduzierbarkeit

Das Projekt muss bei gleichem Random Seed möglichst dieselben Datenlücken und Ergebnisse erzeugen.

Speichere außerdem:

- Konfiguration jedes Experiments
- Modellparameter
- Trainingshistorie
- Validation-Metriken
- Testmetriken
- Datum des Experiments

So können verschiedene Versuche später miteinander verglichen werden.

---

# 27. Automatische Tests

Implementiere Tests für besonders fehleranfällige Bestandteile.

Mindestens prüfen:

1. Zeitliche Reihenfolge der Daten.
2. Keine Überschneidung zwischen Training, Validation und Test.
3. Masken entsprechen tatsächlich den entfernten Positionen.
4. Ground-Truth-Werte werden nicht überschrieben.
5. Modellinput enthält keine versteckte Zielinformation.
6. Ausgabedimension entspricht der Eingabesequenz.
7. Loss wird nur auf relevanten maskierten Positionen berechnet.
8. Normalisierung verwendet nur Trainingsstatistiken.
9. Lineare Interpolation liefert für einfache Testfälle mathematisch korrekte Ergebnisse.

---

# 28. Endprodukte des Projekts

Nach vollständiger Ausführung sollen mindestens folgende Artefakte vorhanden sein:

### Trainiertes Modell

Das beste BiLSTM-Modell.

### Preprocessing-Informationen

Normalisierungsparameter und verwendete Features.

### Ergebnistabelle

MAE und RMSE für jede Methode und jede Lückenlänge.

### Diagramme

Mindestens:

- Rekonstruktionsbeispiele
- MAE gegen Lückenlänge
- Modellvergleich
- Trainings- und Validation-Loss

### README

Das README soll erklären:

- Ziel des Projekts
- verwendeten Datensatz
- Installation
- Datenvorbereitung
- Training
- Evaluation
- Ergebnisse
- Projektstruktur
- wie neue Daten interpoliert werden können

---

# 29. Beispiel für die spätere Benutzung

Das fertige System soll einen Abschnitt einer Zeitreihe erhalten können.

Beispielsweise:

08:00 — 12.3 °C  
08:10 — 12.5 °C  
08:20 — fehlend  
08:30 — fehlend  
08:40 — fehlend  
08:50 — 13.4 °C  
09:00 — 13.6 °C

Das System erkennt anhand der Missing-Mask die Lücke und liefert beispielsweise:

08:20 — 12.72 °C  
08:30 — 12.94 °C  
08:40 — 13.17 °C

Diese Werte sind Rekonstruktionen des Modells und müssen entsprechend gekennzeichnet werden.

---

# 30. Minimalversion des Projekts

Falls zunächst eine kleine funktionsfähige Version erstellt werden soll, beschränke sie auf:

**Datensatz:** Jena Climate

**Target:** Temperatur

**Auflösung:** 10 Minuten

**Szenario:** nur Temperatursensor fällt aus

**Gap-Größen:** 10 min, 30 min, 1 h, 2 h, 4 h, 6 h

**Baselines:** lineare Interpolation und kubischer Spline

**Deep Learning:** bidirektionales LSTM

**Metriken:** MAE und RMSE

**Split:** chronologisch

**Outputs:** Ergebnistabelle + Rekonstruktionsplots + Fehler-vs.-Lückenlänge

Diese Minimalversion muss vollständig funktionieren, bevor Transformer oder komplexere Varianten hinzugefügt werden.

---

# 31. Empfohlene Reihenfolge der Entwicklung

Phase 1:

Datensatz herunterladen und analysieren.

Phase 2:

Daten reinigen und chronologisch in Train, Validation und Test aufteilen.

Phase 3:

Künstliche Datenlücken generieren und kontrollieren.

Phase 4:

Lineare und Spline-Interpolation implementieren.

Phase 5:

Evaluation der klassischen Verfahren implementieren.

Phase 6:

Missing-Mask und Deep-Learning-Datenpipeline entwickeln.

Phase 7:

Bidirektionales LSTM implementieren.

Phase 8:

Modell trainieren und auf Validation optimieren.

Phase 9:

Finale Evaluation auf dem unangetasteten Testdatensatz.

Phase 10:

Experimente für unterschiedliche Lückenlängen durchführen.

Phase 11:

Diagramme und Ergebnistabellen generieren.

Phase 12:

Optional multivariate Features, kompletten Sensorausfall und Transformer hinzufügen.

---

# 32. Erfolgskriterien

Das Projekt gilt als technisch erfolgreich, wenn:

1. künstliche Datenlücken reproduzierbar erzeugt werden,
2. lineare und Spline-Interpolation funktionieren,
3. das BiLSTM fehlende Werte rekonstruieren kann,
4. die Evaluation ausschließlich auf zuvor versteckten Ground-Truth-Werten erfolgt,
5. MAE und RMSE nach Lückenlänge ausgegeben werden,
6. alle Modelle auf identischen Testfällen verglichen werden,
7. mindestens ein Diagramm echte Werte, Datenlücke und Rekonstruktionen gemeinsam zeigt,
8. das Projekt vollständig reproduzierbar ausgeführt werden kann.

Es ist **kein Erfolgskriterium**, dass das neuronale Netz zwangsläufig besser als klassische Interpolation sein muss.

Ein wissenschaftlich interessantes Ergebnis wäre auch, wenn gezeigt wird, dass lineare oder Spline-Interpolation bei kurzen Lücken besser beziehungsweise ähnlich gut ist und Deep Learning erst unter bestimmten Bedingungen Vorteile bietet.

---

# 33. Zentrale Forschungsfrage

Die finale Auswertung soll insbesondere diese Frage beantworten:

**Wie hängt die Qualität klassischer und Deep-Learning-basierter Interpolation von der Länge fehlender Abschnitte und der Verfügbarkeit zusätzlicher Sensordaten ab?**

Die Ergebnisse sollen nicht nur zeigen, welches Modell den kleinsten Gesamtfehler besitzt, sondern erklären, unter welchen Bedingungen welches Verfahren sinnvoll ist.