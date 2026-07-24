![[overview.png]]

### Erste Schritte

Laden Sie ein Metallografie-Bild (JPG, PNG oder BMP) über den Uploader links hoch. Das war's schon — die Analyse startet automatisch, sobald es hochgeladen ist.

### Reiter

- **Original / Zeichnen** — Ihr Originalbild. Zeichnen Sie ein Rechteck darauf und klicken Sie auf „Porosität des ausgewählten Bereichs berechnen“, um nur diesen Bereich zu prüfen.
- **Flächenmaske** — die rote Überlagerung zeigt den ausgewählten Bereich. Bitte passen Sie die Maske so an, dass sie innerhalb des Objekts liegt und alle Poren abdeckt.

        Die Berechnung und Visualisierung ist nur korrekt, wenn die Flächenmaske richtig angepasst ist (Löcher sind abgedeckt und schwarze Bereiche ausgeschlossen).

- **Poren** — Poren blau hervorgehoben. Aktivieren Sie „3 größte Poren hervorheben“, um die drei größten zu markieren und zu vermessen.
- **Korngitter** — erscheint nur bei geätzten Proben. Stellen Sie zuerst sicher, dass der Zoomfaktor korrekt eingestellt ist, denn er wandelt Pixel in echte µm um.

### Bedienfelder (links)

Bitte klicken Sie auf „Neu berechnen“, um Änderungen anzuwenden.

- **Probentyp**: Ungeätzte Bilder: nur Porosität, Geätzte Bilder: zusätzlich Korngröße messen.
- **Grundeinstellungen**
  - OCR-Maßstabserkennung: aktivieren, wenn Tesseract OCR installiert ist, um den Zoomfaktor automatisch zu erkennen
  - Schnelle Maskenverarbeitung: verkleinert die Bildauflösung für schnellere Berechnung, mit geringerer Genauigkeit. Je kleiner die Verarbeitungsgröße, desto schneller. Die Größe bezieht sich auf die längste Seite. Standardmäßig aktiviert, 512 Pixel.
  - Zoomfaktor: manuell eingegeben oder per OCR-Erkennung. Standard 5000 µm.

- **Maskeneinstellung** — Zum Anpassen der Flächenmaske.
  - Porengröße anpassen: Standardmäßig werden kleine Poren angenommen, bei größeren Poren bitte anpassen.
  - Maskenschrumpfung anpassen: Schnelle Maskenverarbeitung führt oft zu Ungenauigkeiten am Rand — schrumpfen Sie die Maske für eine bessere Rand- und Gesamtporosität.
  - Legende zuschneiden: Sie können entweder in Pixel (Standard 100) oder als Prozentsatz der Höhe (Standard 10 %) wählen. Der vom unteren Rand gezählte Bereich wird von der Maske abgezogen. Für mehr Kontrolle und Präzision nutzen Sie stattdessen den Reiter Flächenmaske und zeichnen ihn selbst.

- **Erweiterte Einstellung** — Nur für erfahrene Benutzer. Morphologie-, Rand- und Schwellenwert-Regler im Algorithmus. Die erweiterte Einstellung überschreibt die Anpassungen von Porengröße und Maskenschrumpfung!
- **Analyseaktionen** — alles auf Standard zurücksetzen, oder nach einer Änderung neu berechnen.
