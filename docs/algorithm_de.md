# Algorithmus für Porositäts- und Korngrößenanalyse

Diese Anwendung analysiert Mikroskopiebilder, um die Materialporosität und Korngröße zu bestimmen. Der Algorithmus stützt sich auf automatische Schwellenwertbildung, morphologische Operationen und das statistische Linienschnittverfahren zur Verarbeitung der Proben. Im Folgenden finden Sie eine detaillierte Aufschlüsselung der Algorithmusschritte, mit besonderem Fokus auf die **erweiterten Einstellungen**, die sein Verhalten steuern.

---

## 1. Bildvorbereitung und Entfernen der Legende

Um zu verhindern, dass Text oder Maßstabsmarkierungen fälschlicherweise als Poren oder Probenmaterial klassifiziert werden, entfernt der Algorithmus zunächst die Legende am unteren Bildrand.

- **Zuschneide-Logik:** Das Bild wird zugeschnitten, falls ein Zuschneideverhältnis angegeben ist, um die Legende zu schwärzen.

![alt text]({7EE7A3A7-6C82-4E89-B748-A70A77076C2A}.png)

## 2. Die Probe finden (Flächenmaske)

![alt text]({055C127B-990B-43AC-ABE1-EFB373700913}.png)

Der Algorithmus trennt die Probe vom Hintergrund, indem er eine Interessensbereich-Maske erstellt. Hier kommen die zentralen erweiterten Parameter **Schwellenwert** und **Morphologie** ins Spiel.

### Erweiterte Schwellenwert-Einstellung

- **Masken-Schwellenwert (0 = Automatisch):** Steuert die anfängliche Binarisierung zur Erkennung der äußeren Probengrenze. Bei **0** wird Otsus Methode verwendet, um automatisch den optimalen Grenzwert zu finden. Ein manueller Wert (1-255) erzwingt einen bestimmten Helligkeits-Grenzwert.

### Erweiterte Morphologie-Einstellung

![alt text]({7E1D9BAB-8874-49F7-8537-E7A435D080F6}.png)

Nach der Binarisierung bereinigen morphologische Operationen die Maske. Diese Einstellungen überschreiben direkt die einfachen „Porengröße“-Voreinstellungen.

- **Open-Kernel-Verhältnis (0-5):** Steuert die Entfernung kleiner Rauschflecken von der Probengrenze. Höhere Werte entfernen größere Flecken, können aber echte Details wegätzen (Bereich von 1/400 bis 1/10 der Verarbeitungshöhe).

- **Open-Iterationen:** Legt fest, wie oft der Rauschentfernungsschritt (Open) wiederholt wird. Ein Wert von 0 deaktiviert ihn; höhere Werte ergeben eine aggressivere Bereinigung.

- **Close-Kernel-Verhältnis (0-5):** Bestimmt, wie groß ein dunkler Bereich sein darf, bevor er aufgefüllt und als Teil der festen Probenmaske gezählt wird. Höhere Werte tolerieren größere dunkle Bereiche.

- **Close-Iterationen:** Legt fest, wie oft der Lochfüllschritt (Close) wiederholt wird. Ein Wert von 0 deaktiviert ihn; höhere Werte ergeben ein aggressiveres Füllen.

- **Maske schrumpfen (0-5):** Eine optionale Einstellung, die ganz am Ende angewendet wird, um die endgültige Grenze nach innen zu ziehen. Dies verhindert, dass die Maske an rauen äußeren Rändern oder Polierartefakten klebt.

```text
function select_area_of_interest(image, mask_threshold, morphology, shrink_ratio):
    binary_img = apply_threshold(image, mask_threshold)

    // Erweiterte Morphologie-Bereinigung
    mask = morphology_open(binary_img, morphology.open_kernel, morphology.open_iters)
    mask = morphology_close(mask, morphology.close_kernel, morphology.close_iters)

    // Äußere Randartefakte ausschließen
    if shrink_ratio > 0:
        mask = erode(mask, shrink_ratio)

    return mask

```

---

## 3. Porositätsberechnung

![alt text]({9A459C17-6C00-4032-8A71-86FC620FA27E}.png)

Porosität ist der Prozentsatz der gültigen Porenfläche im Verhältnis zur gesamten Materialfläche. Der Algorithmus invertiert das Binärbild, um mögliche Poren hervorzuheben, und filtert sie strikt anhand der Flächenmaske.

### Erweiterte Schwellenwert-Einstellung

- **Schwellenwert (0 = Automatisch):** Bestimmt den Helligkeits-Grenzwert speziell zur Trennung von Poren und festem Material innerhalb der Probe. Ein Wert von **0** verwendet die automatische Otsu-Schwellenwertbildung; andernfalls wird ein manueller Wert (1-255) angewendet.
  Verwechseln Sie dies nicht mit dem Masken-Schwellenwert — es handelt sich um zwei unterschiedliche Werte, die in unterschiedlichen Phasen verwendet werden. Der Helligkeitskontrast zwischen Probe und Hintergrund unterscheidet sich oft völlig vom Kontrast zwischen festem Material und einer mikroskopischen Pore. Ist der Hintergrund beispielsweise sehr dunkel, kann ein einzelner globaler Schwellenwert die Probe perfekt umreißen, aber versehentlich hellgraue Schatten innerhalb der Probe als Poren klassifizieren. Durch die Trennung können Sie den Masken-Schwellenwert so einstellen, dass er den Umriss der Probe perfekt erfasst, und unabhängig davon den Schwellenwert so abstimmen, dass er nur die tatsächlichen Poren im Material erfasst.

### Erweiterte Randeinstellung

Die Anwendung berechnet sowohl die Gesamtbild-Porosität als auch die Randporosität. Die Randporosität ist auf einen dünnen Ring nahe dem Rand der Maske beschränkt, gesteuert durch diese Einstellungen:

- **Randbreite (px):** Eine feste Dicke in Pixeln für den Randring. Bei einem Wert größer als 0 überschreibt dies die Verhältnis-Einstellung.

- **Randverhältnis-Stufe (0-3):** Wird nur verwendet, wenn die Randbreite 0 ist. Legt die Randdicke als Prozentsatz (5 %, 10 %, 15 % oder 20 %) des geschätzten Radius der Probe fest.

```text
function calculate_porosity(image, mask, threshold, border_settings):
    binary_img = apply_threshold(image, threshold)
    whole_area = count_white_pixels(mask)

    // Poren strikt innerhalb der Probenfläche finden
    inverse_binary = bitwise_not(binary_img)
    actual_pores = bitwise_and(inverse_binary, mask)

    // Gesamtporosität
    pore_area = count_white_pixels(actual_pores)
    full_porosity = (pore_area * 100) / whole_area

    // Randporosität
    border_mask = generate_border(mask, border_settings.width, border_settings.ratio)
    border_pores = bitwise_and(inverse_binary, border_mask)
    border_porosity = (count_white_pixels(border_pores) * 100) / count_white_pixels(border_mask)

    return full_porosity, border_porosity

```

---

## 4. Korngrößenberechnung (nur geätzte Proben)

[Bild hier einfügen: Beispiel der Korngrenzen-Überlagerung mit Gitterlinien]

Bei geätzten Proben wird die Korngröße mittels des statistischen Linienschnittverfahrens berechnet.

- **Kantenerkennung:** Die Canny-Kantenerkennung identifiziert Korngrenzen, strikt beschränkt auf die Flächenmaske.

- **Gitterlinien:** Horizontale und vertikale Testlinien werden über das Bild gezogen. Der Benutzer legt die Anzahl der Linien fest; mehr Linien ergeben eine bessere Abtastung, dauern aber länger.

- **Berechnung:** Der Algorithmus zählt die Grenzüberschneidungen entlang der Gitterlinien. Die mittlere Schnittlänge (Pixel) wird durch die Anzahl der Schnittpunkte geteilt und mithilfe des erkannten **Zoomfaktors** in Mikrometer (µm) umgerechnet.

```text
function calculate_grain_size(image, mask, num_lines, zoom_factor):
    edges = canny_edge_detection(image)
    valid_edges = bitwise_and(edges, mask)

    total_line_length = 0
    total_intersections = 0

    // Testlinien verarbeiten
    for line in generate_grid_lines(num_lines):
        total_line_length += count_pixels(line, mask)
        total_intersections += count_edge_crossings(line, valid_edges)

    mean_intercept_px = total_line_length / total_intersections
    grain_size_um = mean_intercept_px * (zoom_factor / image_width)

    return grain_size_um

```

---

## 5. Leistungsoptionen: Schnell vs. Genau

Um die Verarbeitungszeiten bei hochauflösenden Mikroskopiebildern zu handhaben, bietet die Anwendung Regler für die Ausführungsgeschwindigkeit.

- **Schnelle Maskenverarbeitung:** Bei Aktivierung wird das Bild verkleinert, bevor der Algorithmus versucht, die Probengrenze zu erkennen. Dies beschleunigt den morphologischen Bereinigungsschritt drastisch.

- **Verarbeitungsgröße (px):** Legt die maximale Pixelabmessung der längsten Seite während der schnellen Maskenverarbeitung fest. Höhere Werte (z. B. 1024) liefern präzisere Grenzen, rechnen aber langsamer als niedrigere Werte (z. B. 512). Wird die schnelle Maskenverarbeitung deaktiviert, laufen die Operationen in der ursprünglichen, vollen Auflösung.
