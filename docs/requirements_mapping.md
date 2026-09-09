# Zuordnung der Anleitung zur Umsetzung

| Vorgabe | Umsetzung / Nachweis |
| --- | --- |
| Jena Climate 2009–2016, zehn Minuten | `download.py`, Rohdaten mit SHA-256; kein Resampling |
| Hauptziel Temperatur | zentrale Zielvariable `T (degC)` |
| Szenario A und B | `gaps.py`, je Szenario vier Kontextvarianten |
| Chronologische Aufteilung | `cleaning.split_frame`, `split_summary.json` |
| Ungültige Werte, Zeitlücken, Duplikate | `cleaning.py`, `cleaning_audit.json` |
| Zyklische Zeit- und Windmerkmale | `features.engineer_features` |
| Normalisierung nur aus Training | `Standardizer`, gespeicherte Fit-Zeitspanne |
| Lücken 1/3/6/12/24/36 Schritte | Konfigurationen, feste CSV-Manifeste |
| Zufällige Trainingslücken | `RandomGapBatches`, Seed je Epoche |
| Beidseitiger Kontext | exakte Fensterlänge `2 × context + gap` |
| Missing-Mask | eine Beobachtungsmaske je Sensor; Zeitmerkmale bleiben sichtbar |
| Linear, Spline, PCHIP, Forward Fill | `baselines.py` |
| BiLSTM 64/32, Dropout 0,2 | `model.TemperatureBiLSTM` |
| Fehler nur für entfernte Werte | `model.masked_mse`, Gradienten-Test |
| Adam, Early Stopping, Lernratenreduktion | `training.train_model` |
| Bestes Validation-Modell | `best_model.pt`, `training_metadata.json` |
| MAE/RMSE in Celsius je Länge | `evaluation.py`, Tabellen je Modell |
| Identische Testlücken | gemeinsame Manifeste, Abgleich der Punkt-IDs |
| Temperatur-only vs. mehrere Sensoren | eigener 12-Stunden-Temperatur-only-Lauf |
| Kontext 3/6/12/24 Stunden | beide Szenarien mit je vier Modellen |
| Keine Testoptimierung | sämtliche Trainingsläufe vor erster Testvorhersage |
| Kein verstecktes Temperatursignal | Feature-Zulassungsliste, Gegenfaktentest |
| Diagramme | PNG/PDF je Modell sowie Gesamtvergleiche |
| Modularität und Konfiguration | getrennte Module, neun JSON-Konfigurationen |
| Reproduzierbarkeit | Seeds, Umgebungsdatei, Bibliotheks-Lockdatei, Protokoll-Prüfsummen |
| Automatische Tests | `tests/test_project.py`, XML-Testbericht |
| Anwendung auf neue Zeitreihen | `inference.py`, CLI `interpolate`, Beispiel-CSV-Dateien |
| Technische Abschlussprüfung | `scripts/verify_artifacts.py`, unabhängige Metrikprüfung |

Der Transformer ist laut Anleitung optional und wurde nach Abstimmung nicht in
den Versuchsplan aufgenommen. Die spätere Erweiterbarkeit wird durch getrennte
Sensormasken, austauschbare Modelldefinition und modulare Datenverarbeitung
vorbereitet; das aktuelle Ausgabeformat rekonstruiert ausdrücklich nur Temperatur.

Die Empfehlung von 12 Stunden Kontext gilt für den Hauptlauf; kürzere und längere
Fenster sind die ausdrücklich geforderten kontrollierten Zusatzexperimente.
