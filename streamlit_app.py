import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pytesseract
import re
import shutil
import os
import base64
import mimetypes

if shutil.which("tesseract") is None:
    _candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for _path in _candidate_paths:
        if os.path.isfile(_path):
            pytesseract.pytesseract.tesseract_cmd = _path
            break

# Import your existing backend logic
from segment import analyze_porosity, calculate_grain_size, select_border_of_interest

st.set_page_config(page_title="Porosity & Grain Analysis", layout="wide")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(APP_DIR, "videos")
DOCS_DIR = os.path.join(APP_DIR, "docs")

def _image_to_data_uri(filename):
    path = os.path.join(DOCS_DIR, filename)
    if not os.path.isfile(path):
        return None
    mime, _ = mimetypes.guess_type(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime or 'application/octet-stream'};base64,{b64}"

def render_doc_markdown(content):
    """Render a docs/*.md file, inlining local images as base64 data URIs.

    st.markdown() turns markdown image syntax into a plain <img src="..."> tag, and
    a relative local filename doesn't resolve to anything Streamlit actually serves --
    only an absolute URL or a data URI works. This also supports Obsidian's ![[filename]]
    wiki-embed syntax (translated to the same data URI form), since that's how these
    docs tend to get written.
    """
    def _replace_wiki(m):
        filename = m.group(1).split("|")[0].strip()
        uri = _image_to_data_uri(filename)
        return f"![{filename}]({uri})" if uri else m.group(0)

    def _replace_standard(m):
        alt, path = m.group(1), m.group(2)
        if path.startswith(("http://", "https://", "data:")):
            return m.group(0)
        uri = _image_to_data_uri(path)
        return f"![{alt}]({uri})" if uri else m.group(0)

    content = re.sub(r'!\[\[([^\]]+)\]\]', _replace_wiki, content)
    content = re.sub(r'!\[([^\]]*)\]\(([^)\s]+)\)', _replace_standard, content)
    st.markdown(content)

# ==========================
# TRANSLATIONS
# ==========================
# Every user-facing string lives here, keyed by a short id. t(key) looks up the
# current language. Widgets whose stored VALUE is compared elsewhere in the code
# (Sample Type, Pore Size, Select View) use language-independent canonical keys
# ("etched", "small", "original_draw", ...) as their actual option values, with
# format_func used only to show the translated label -- so switching languages
# never breaks a value already stored in session_state.
T = {
    "app_title": {"en": "Porosity & Grain Analysis Tool", "de": "Porositäts- & Korngrößenanalyse-Tool"},
    "controls_header": {"en": "Controls", "de": "Steuerung"},
    "upload_image": {"en": "Upload Image", "de": "Bild hochladen"},

    "sample_type_label": {"en": "Sample Type", "de": "Probentyp"},
    "sample_type_help": {
        "en": "Unetched: porosity analysis only. Etched: grain boundaries are visible, so grain size is measured too.",
        "de": "Ungeätzt: nur Porositätsanalyse. Geätzt: Korngrenzen sind sichtbar, daher wird auch die Korngröße gemessen."
    },
    "unetched": {"en": "Unetched", "de": "Ungeätzt"},
    "etched": {"en": "Etched", "de": "Geätzt"},

    "basic_settings": {"en": "Basic Settings", "de": "Grundeinstellungen"},
    "ocr_label": {"en": "Auto-detect Scale (OCR)", "de": "Maßstab automatisch erkennen (OCR)"},
    "ocr_help": {
        "en": "Uses Tesseract OCR to read the µm scale from the bottom of newly uploaded images. Requires Tesseract to be installed on this machine.",
        "de": "Verwendet Tesseract OCR, um den µm-Maßstab am unteren Rand neu hochgeladener Bilder zu lesen. Erfordert eine installierte Tesseract-Instanz auf diesem Rechner."
    },
    "fast_mask_label": {"en": "Fast Mask Processing", "de": "Schnelle Maskenverarbeitung"},
    "fast_mask_help": {
        "en": "Downsizes the image before detecting the sample boundary, for speed. Turn off to classify the boundary at full resolution instead.",
        "de": "Verkleinert das Bild vor der Erkennung der Probengrenze, um Geschwindigkeit zu gewinnen. Deaktivieren, um die Grenze stattdessen in voller Auflösung zu erkennen."
    },
    "processing_size_label": {"en": "Processing Size (px)", "de": "Verarbeitungsgröße (px)"},
    "processing_size_help": {
        "en": "The image is downsized so its largest side is at most this many pixels before detecting the sample boundary. Higher = more precise but slower.",
        "de": "Das Bild wird so verkleinert, dass seine längste Seite höchstens diese Anzahl an Pixeln hat, bevor die Probengrenze erkannt wird. Höher = präziser, aber langsamer."
    },
    "zoom_factor_label": {"en": "Zoom Factor (µm)", "de": "Zoomfaktor (µm)"},
    "zoom_factor_help": {
        "en": "The real-world width of the image in micrometers, used to convert pixel measurements into grain size (µm). Auto-filled by OCR if enabled above.",
        "de": "Die reale Breite des Bildes in Mikrometern, verwendet um Pixelmessungen in Korngröße (µm) umzurechnen. Wird bei aktiviertem OCR oben automatisch ausgefüllt."
    },

    "mask_setting": {"en": "Mask Setting", "de": "Maskeneinstellung"},
    "adjust_pore_size_label": {"en": "Adjust Pore Size", "de": "Porengröße anpassen"},
    "adjust_pore_size_help": {
        "en": "Lets you tune how large a dark region can be before it's still folded into the sample area, instead of using the default.",
        "de": "Ermöglicht es einzustellen, wie groß ein dunkler Bereich sein darf, um noch zur Probenfläche gezählt zu werden, statt den Standardwert zu verwenden."
    },
    "pore_size_demo_heading": {"en": "How Pore Size Works", "de": "Wie die Porengröße funktioniert"},
    "pore_size_demo_caption": {
        "en": "Demonstration of how the Pore Size setting affects the mask on an example image.",
        "de": "Demonstration, wie die Einstellung der Porengröße die Maske an einem Beispielbild beeinflusst."
    },
    "pore_size_demo_tooltip": {
        "en": "Watch a short video showing how Pore Size affects the mask, on an example image",
        "de": "Kurzes Video ansehen, das zeigt, wie die Porengröße die Maske an einem Beispielbild beeinflusst"
    },
    "pore_size_label": {"en": "Pore Size", "de": "Porengröße"},
    "pore_size_help": {
        "en": "Small Pores only folds tiny specks into the sample area; Big Pores tolerates larger dark regions as part of the sample boundary. Adjusting Close/Open Kernel Ratio directly in Advanced Options below overrides this preset.",
        "de": "Kleine Poren zählt nur winzige Flecken zur Probenfläche; Große Poren toleriert größere dunkle Bereiche als Teil der Probengrenze. Eine direkte Anpassung des Close-/Open-Kernel-Verhältnisses in den erweiterten Einstellungen überschreibt diese Voreinstellung."
    },
    "hole_small": {"en": "Small Pores", "de": "Kleine Poren"},
    "hole_medium": {"en": "Medium", "de": "Mittel"},
    "hole_large": {"en": "Large", "de": "Groß"},
    "hole_big": {"en": "Big Pores", "de": "Große Poren"},

    "adjust_mask_shrink_label": {"en": "Adjust Mask Shrink", "de": "Maskenschrumpfung anpassen"},
    "adjust_mask_shrink_help": {
        "en": "Lets you shrink the final mask inward so it doesn't hug the sample's outer edge, instead of using the default (no shrink).",
        "de": "Ermöglicht es, die endgültige Maske nach innen zu schrumpfen, damit sie nicht am äußeren Rand der Probe klebt, statt den Standard (keine Schrumpfung) zu verwenden."
    },
    "mask_shrink_demo_heading": {"en": "How Mask Shrink Works", "de": "Wie die Maskenschrumpfung funktioniert"},
    "mask_shrink_demo_caption": {
        "en": "Demonstration of how the Shrink Mask setting pulls the boundary inward on an example image.",
        "de": "Demonstration, wie die Maskenschrumpfung die Grenze an einem Beispielbild nach innen zieht."
    },
    "mask_shrink_demo_tooltip": {
        "en": "Watch a short video showing how Mask Shrink affects the boundary, on an example image",
        "de": "Kurzes Video ansehen, das zeigt, wie die Maskenschrumpfung die Grenze an einem Beispielbild beeinflusst"
    },
    "shrink_mask_label": {"en": "Shrink Mask (0-5)", "de": "Maske schrumpfen (0-5)"},
    "shrink_mask_help": {
        "en": "Pulls the final area-of-interest mask inward so it doesn't hug the sample's outer edge -- useful for excluding polishing/mounting artifacts right at the border. Applied last, after the Pore Size / Close Kernel step, so it isn't undone by hole-filling. Fixed values: 0 = no shrink, 1 = 1%, 2 = 2%, 3 = 4%, 4 = 7%, 5 = 10% of the mask's estimated radius.",
        "de": "Zieht die endgültige Interessensbereich-Maske nach innen, damit sie nicht am äußeren Rand der Probe klebt -- nützlich, um Polier-/Einbettungsartefakte direkt am Rand auszuschließen. Wird zuletzt angewendet, nach dem Schritt Porengröße/Close-Kernel, damit es nicht durch das Lochfüllen rückgängig gemacht wird. Feste Werte: 0 = keine Schrumpfung, 1 = 1 %, 2 = 2 %, 3 = 4 %, 4 = 7 %, 5 = 10 % des geschätzten Radius der Maske."
    },
    "crop_legend_label": {"en": "Crop Legend", "de": "Legende zuschneiden"},
    "crop_legend_help": {
        "en": "Blacks out a strip at the bottom of the image (e.g. a scale bar or legend) before analysis, so it isn't mistaken for sample or pore area. This area is deducted from the mask. For finer control, use the Area Mask tab and draw it yourself instead.",
        "de": "Schwärzt einen Streifen am unteren Bildrand (z. B. Maßstabsbalken oder Legende) vor der Analyse, damit er nicht als Proben- oder Porenfläche fehlinterpretiert wird. Dieser Bereich wird von der Maske abgezogen. Für mehr Kontrolle nutzen Sie stattdessen den Reiter Flächenmaske und zeichnen ihn selbst."
    },
    "crop_height_unit_label": {"en": "Crop Height Unit", "de": "Einheit der Zuschneidehöhe"},
    "crop_height_unit_help": {
        "en": "Whether the crop height below is given in pixels or as a percentage of the image height.",
        "de": "Ob die Zuschneidehöhe unten in Pixeln oder als Prozentsatz der Bildhöhe angegeben wird."
    },
    "crop_unit_pixels": {"en": "Pixels", "de": "Pixel"},
    "crop_unit_percentage": {"en": "Percentage of Height", "de": "Prozent der Höhe"},
    "crop_height_label": {"en": "Crop Height (px from bottom)", "de": "Zuschneidehöhe (px vom unteren Rand)"},
    "crop_height_help": {
        "en": "How many pixels tall the cropped-out strip is, measured up from the bottom of the image. This area is deducted from the mask. For finer control, use the Area Mask tab and draw it yourself instead.",
        "de": "Wie viele Pixel hoch der zugeschnittene Streifen ist, gemessen vom unteren Bildrand nach oben. Dieser Bereich wird von der Maske abgezogen. Für mehr Kontrolle nutzen Sie stattdessen den Reiter Flächenmaske und zeichnen ihn selbst."
    },
    "crop_height_percent_label": {"en": "Crop Height (% of image height)", "de": "Zuschneidehöhe (% der Bildhöhe)"},
    "crop_height_percent_help": {
        "en": "How tall the cropped-out strip is, as a percentage of the image's height, measured up from the bottom. This area is deducted from the mask. For finer control, use the Area Mask tab and draw it yourself instead.",
        "de": "Wie hoch der zugeschnittene Streifen ist, als Prozentsatz der Bildhöhe, gemessen vom unteren Bildrand nach oben. Dieser Bereich wird von der Maske abgezogen. Für mehr Kontrolle nutzen Sie stattdessen den Reiter Flächenmaske und zeichnen ihn selbst."
    },

    "advanced_setting": {"en": "Advanced Setting", "de": "Erweiterte Einstellung"},
    "advanced_warning": {
        "en": "These controls are meant for advanced users who understand the underlying algorithm.",
        "de": "Diese Einstellungen sind für erfahrene Benutzer gedacht, die den zugrunde liegenden Algorithmus verstehen."
    },
    "algorithm_link_advanced": {"en": "Click here to understand the algorithm", "de": "Hier klicken, um den Algorithmus zu verstehen"},
    "algorithm_link_tutorial": {"en": "Click here to see how the algorithm works", "de": "Hier klicken, um zu sehen, wie der Algorithmus funktioniert"},
    "algorithm_dialog_title": {"en": "How the Algorithm Works", "de": "Wie der Algorithmus funktioniert"},

    "morphology_setting_header": {
        "en": "**Morphology Setting** (overrides the Mask Setting panel's Pore Size preset)",
        "de": "**Morphologie-Einstellung** (überschreibt die Porengrößen-Voreinstellung im Maskeneinstellungs-Panel)"
    },
    "open_kernel_label": {"en": "Open Kernel Ratio (0-5)", "de": "Open-Kernel-Verhältnis (0-5)"},
    "open_kernel_help": {
        "en": "Removes small noise specks from the sample boundary mask. Higher = removes larger specks, but can erode real detail. Fixed values (as a fraction of the mask's processing height): 0 = 1/400, 1 = 1/200, 2 = 1/100, 3 = 1/50, 4 = 1/25, 5 = 1/10.",
        "de": "Entfernt kleine Rauschflecken aus der Probengrenzmaske. Höher = entfernt größere Flecken, kann aber echte Details wegätzen. Feste Werte (als Bruchteil der Verarbeitungshöhe der Maske): 0 = 1/400, 1 = 1/200, 2 = 1/100, 3 = 1/50, 4 = 1/25, 5 = 1/10."
    },
    "close_kernel_label": {"en": "Close Kernel Ratio (0-5)", "de": "Close-Kernel-Verhältnis (0-5)"},
    "close_kernel_help": {
        "en": "How large a dark region can be before it's still folded into the sample area mask. Higher = tolerates bigger dark regions as part of the sample. Fixed values (as a fraction of the mask's processing height): 0 = 1/400, 1 = 1/200, 2 = 1/100, 3 = 1/50, 4 = 1/25, 5 = 1/10.",
        "de": "Wie groß ein dunkler Bereich sein darf, um noch zur Probenflächenmaske gezählt zu werden. Höher = toleriert größere dunkle Bereiche als Teil der Probe. Feste Werte (als Bruchteil der Verarbeitungshöhe der Maske): 0 = 1/400, 1 = 1/200, 2 = 1/100, 3 = 1/50, 4 = 1/25, 5 = 1/10."
    },
    "open_iters_label": {"en": "Open Iterations", "de": "Open-Iterationen"},
    "open_iters_help": {
        "en": "How many times the noise-removal (open) step repeats. This is a direct repeat count, not a mapped value -- 0 disables it, 5 repeats it 5 times. Higher = more aggressive cleanup.",
        "de": "Wie oft der Rauschentfernungsschritt (Open) wiederholt wird. Dies ist eine direkte Wiederholungsanzahl, kein zugeordneter Wert -- 0 deaktiviert ihn, 5 wiederholt ihn 5-mal. Höher = aggressivere Bereinigung."
    },
    "close_iters_label": {"en": "Close Iterations", "de": "Close-Iterationen"},
    "close_iters_help": {
        "en": "How many times the hole-filling (close) step repeats. This is a direct repeat count, not a mapped value -- 0 disables it, 5 repeats it 5 times. Higher = more aggressive filling.",
        "de": "Wie oft der Lochfüllschritt (Close) wiederholt wird. Dies ist eine direkte Wiederholungsanzahl, kein zugeordneter Wert -- 0 deaktiviert ihn, 5 wiederholt ihn 5-mal. Höher = aggressiveres Füllen."
    },

    "border_setting_header": {"en": "**Border Setting**", "de": "**Randeinstellung**"},
    "border_width_label": {"en": "Border Width (px)", "de": "Randbreite (px)"},
    "border_width_help": {
        "en": "Fixed border thickness, in pixels, used for the Border Porosity measurement. This is used directly as entered (not a mapped value). Leave at 0 to use Border Ratio Level instead.",
        "de": "Feste Randdicke in Pixeln für die Randporositätsmessung. Wird direkt wie eingegeben verwendet (kein zugeordneter Wert). Auf 0 lassen, um stattdessen die Randverhältnis-Stufe zu verwenden."
    },
    "border_ratio_label": {"en": "Border Ratio Level (0-3)", "de": "Randverhältnis-Stufe (0-3)"},
    "border_ratio_help": {
        "en": "Only used when Border Width is 0. Sets the border ring's thickness as a percentage of the sample's estimated radius. Fixed values: 0 = 5%, 1 = 10%, 2 = 15%, 3 = 20% of the estimated radius.",
        "de": "Wird nur verwendet, wenn die Randbreite 0 ist. Legt die Dicke des Randrings als Prozentsatz des geschätzten Radius der Probe fest. Feste Werte: 0 = 5 %, 1 = 10 %, 2 = 15 %, 3 = 20 % des geschätzten Radius."
    },

    "threshold_setting_header": {"en": "**Threshold Setting**", "de": "**Schwellenwert-Einstellung**"},
    "threshold_label": {"en": "Threshold (0 = Auto)", "de": "Schwellenwert (0 = Automatisch)"},
    "threshold_help": {
        "en": "Manual brightness cutoff (0-255) used to separate pores from solid material, used directly as entered (not a mapped value). 0 lets the app choose automatically (Otsu's method).",
        "de": "Manueller Helligkeits-Grenzwert (0-255) zur Trennung von Poren und festem Material, wird direkt wie eingegeben verwendet (kein zugeordneter Wert). 0 lässt die App automatisch wählen (Otsu-Methode)."
    },
    "mask_threshold_label": {"en": "Mask Threshold (0 = Auto)", "de": "Masken-Schwellenwert (0 = Automatisch)"},
    "mask_threshold_help": {
        "en": "Manual brightness cutoff (0-255) used to detect the sample's outer boundary (area of interest), used directly as entered (not a mapped value). 0 lets the app choose automatically (Otsu's method).",
        "de": "Manueller Helligkeits-Grenzwert (0-255) zur Erkennung der äußeren Probengrenze (Interessensbereich), wird direkt wie eingegeben verwendet (kein zugeordneter Wert). 0 lässt die App automatisch wählen (Otsu-Methode)."
    },

    "grain_size_calc_header": {"en": "Grain Size Calculation", "de": "Korngrößenberechnung"},
    "grain_lines_label": {"en": "Grain Size Grid Lines (H & V)", "de": "Korngrößen-Gitterlinien (H & V)"},
    "grain_lines_help": {
        "en": "Number of horizontal and vertical test lines used to measure grain size via the line-intercept method. More lines sample more of the image but take longer.",
        "de": "Anzahl der horizontalen und vertikalen Testlinien zur Messung der Korngröße mittels Linienschnittverfahren. Mehr Linien erfassen mehr vom Bild, dauern aber länger."
    },

    "analysis_actions": {"en": "Analysis Actions", "de": "Analyseaktionen"},
    "reset_button": {"en": "Reset Default Parameters", "de": "Standardparameter zurücksetzen"},
    "recalculate_button": {"en": "Recalculate", "de": "Neu berechnen"},

    "analyzing_spinner": {"en": "Analyzing image...", "de": "Bild wird analysiert..."},
    "metric_full_porosity": {"en": "Full Image Porosity", "de": "Gesamtbild-Porosität"},
    "metric_border_porosity": {"en": "Border Porosity", "de": "Randporosität"},
    "metric_grain_size": {"en": "Mean Grain Size", "de": "Mittlere Korngröße"},

    "select_view_label": {"en": "Select View", "de": "Ansicht auswählen"},
    "view_original": {"en": "Original", "de": "Original"},
    "view_area_mask": {"en": "Area Mask", "de": "Flächenmaske"},
    "view_pores": {"en": "Pores", "de": "Poren"},
    "view_border": {"en": "Border", "de": "Rand"},
    "view_grain_grid": {"en": "Grain Grid", "de": "Korngitter"},

    "draw_info": {
        "en": "Draw rectangles on the image below to outline specific regions. Then use the buttons below the image to process your selection.",
        "de": "Zeichnen Sie Rechtecke auf das Bild unten, um bestimmte Bereiche zu markieren. Verwenden Sie dann die Schaltflächen unter dem Bild, um Ihre Auswahl zu verarbeiten."
    },
    "calc_region_demo_heading": {
        "en": "How Calculate Selected Region Porosity Works",
        "de": "Wie die Berechnung der Porosität eines ausgewählten Bereichs funktioniert"
    },
    "calc_region_demo_caption": {
        "en": "Demonstration of drawing a region and calculating its porosity on an example image.",
        "de": "Demonstration, wie ein Bereich gezeichnet und dessen Porosität an einem Beispielbild berechnet wird."
    },
    "calc_region_demo_tooltip": {
        "en": "Watch a short video showing how to draw a region and calculate its porosity",
        "de": "Kurzes Video ansehen, das zeigt, wie man einen Bereich zeichnet und dessen Porosität berechnet"
    },
    "calc_region_button": {"en": "Calculate Selected Region Porosity", "de": "Porosität des ausgewählten Bereichs berechnen"},
    "region_result": {"en": "Selected Region Porosity: {value:.2f}%", "de": "Porosität des ausgewählten Bereichs: {value:.2f}%"},
    "draw_rect_warning": {"en": "Please draw a rectangle first.", "de": "Bitte zuerst ein Rechteck zeichnen."},

    "erase_info": {
        "en": "Draw rectangles on the mask to erase regions. Click 'Erase Selection' to apply. Erased regions remain across calculations until reset.",
        "de": "Zeichnen Sie Rechtecke auf die Maske, um Bereiche zu löschen. Klicken Sie auf „Auswahl löschen“, um dies anzuwenden. Gelöschte Bereiche bleiben bis zum Zurücksetzen über Berechnungen hinweg bestehen."
    },
    "erase_demo_heading": {"en": "How Erase Selection Works", "de": "Wie das Löschen der Auswahl funktioniert"},
    "erase_demo_caption": {
        "en": "Demonstration of drawing and erasing regions from the mask on an example image.",
        "de": "Demonstration, wie Bereiche aus der Maske gezeichnet und gelöscht werden, an einem Beispielbild."
    },
    "erase_demo_tooltip": {
        "en": "Watch a short video showing how to erase regions from the mask",
        "de": "Kurzes Video ansehen, das zeigt, wie man Bereiche aus der Maske löscht"
    },
    "erase_selection_button": {"en": "Erase Selection", "de": "Auswahl löschen"},
    "draw_rect_warning2": {"en": "Please draw at least one rectangle first.", "de": "Bitte zuerst mindestens ein Rechteck zeichnen."},
    "reset_erasures_button": {"en": "Reset Erasures", "de": "Löschungen zurücksetzen"},
    "erasures_cleared_warning": {
        "en": "Erasures cleared. Click 'Recalculate' in the sidebar to restore the default mask.",
        "de": "Löschungen entfernt. Klicken Sie in der Seitenleiste auf „Neu berechnen“, um die Standardmaske wiederherzustellen."
    },
    "upload_to_process_mask": {"en": "Upload an image to process the mask.", "de": "Laden Sie ein Bild hoch, um die Maske zu verarbeiten."},

    "highlight_biggest_label": {"en": "Highlight 3 Biggest Pores", "de": "3 größte Poren hervorheben"},
    "highlight_biggest_help": {
        "en": "Draws a bounding box and diameter label on the 3 largest pores in the area of interest.",
        "de": "Zeichnet einen Begrenzungsrahmen und eine Durchmesserbeschriftung für die 3 größten Poren im Interessensbereich."
    },
    "no_pores_info": {"en": "No pores detected to highlight.", "de": "Keine Poren zum Hervorheben erkannt."},

    "ocr_success": {
        "en": "✅ **Scale Detected:** Tesseract OCR successfully scanned the bottom 20% of the image and found a zoom factor of **{value} µm**. The scale has been automatically updated in the sidebar.",
        "de": "✅ **Maßstab erkannt:** Tesseract OCR hat die unteren 20 % des Bildes erfolgreich gescannt und einen Zoomfaktor von **{value} µm** gefunden. Der Maßstab wurde automatisch in der Seitenleiste aktualisiert."
    },
    "ocr_warning": {
        "en": "⚠️ **Scale Not Found:** Tesseract OCR scanned the bottom of the image but could not detect any numbers. The zoom factor has been set to the default **5000 µm**.",
        "de": "⚠️ **Maßstab nicht gefunden:** Tesseract OCR hat den unteren Bildbereich gescannt, konnte aber keine Zahlen erkennen. Der Zoomfaktor wurde auf den Standardwert **5000 µm** gesetzt."
    },
    "ocr_error": {
        "en": "❌ **OCR Error:** Tesseract OCR failed to run or is not properly configured on your system. Defaulting to **5000 µm**.",
        "de": "❌ **OCR-Fehler:** Tesseract OCR konnte nicht ausgeführt werden oder ist auf diesem System nicht richtig konfiguriert. Standardwert **5000 µm** wird verwendet."
    },
}

def t(key, **kwargs):
    text = T[key][st.session_state.language]
    return text.format(**kwargs) if kwargs else text

def _localized_format_func(label_key_of, lang):
    """Build a format_func that's fully self-contained: `lang` is baked in via a
    default argument at creation time, so the closure never needs to look anything
    up (session_state, thread-locals, etc.) when it's actually called. This matters
    because Streamlit -- and its AppTest harness in particular -- can invoke a
    widget's format_func outside the normal script execution context (e.g. when
    reconciling widget state between reruns), where st.session_state isn't reliably
    accessible and a mutable "current language" lookup would go stale or raise.
    `label_key_of` maps a widget's raw option value to a key in T; it can be a dict
    (Pore Size, Select View) or the identity, since Sample Type's raw values ("unetched"/
    "etched") are themselves already valid T keys.
    """
    def _fmt(value, _label_key_of=label_key_of, _lang=lang):
        t_key = _label_key_of[value] if isinstance(_label_key_of, dict) else value
        return T[t_key][_lang]
    return _fmt

HOLE_SIZE_KEYS = ["small", "medium", "large", "big"]
HOLE_SIZE_LABEL_KEYS = {"small": "hole_small", "medium": "hole_medium", "large": "hole_large", "big": "hole_big"}
SAMPLE_TYPE_KEYS = ["unetched", "etched"]
VIEW_LABEL_KEYS = {
    "original_draw": "view_original", "area_mask": "view_area_mask", "pores": "view_pores",
    "border": "view_border", "grain_grid": "view_grain_grid",
}
CROP_UNIT_KEYS = ["pixels", "percentage"]
CROP_UNIT_LABEL_KEYS = {"pixels": "crop_unit_pixels", "percentage": "crop_unit_percentage"}

@st.dialog("Demo", width="large")
def show_demo_video(video_filename, heading, caption):
    st.subheader(heading)
    st.video(os.path.join(VIDEOS_DIR, video_filename))
    st.caption(caption)

def demo_video_button(key, video_filename, heading, caption, tooltip):
    """Small, bordered 'Demo' button that opens a large video dialog when clicked."""
    if st.button("Demo", icon="▶", key=key, help=tooltip):
        show_demo_video(video_filename, heading, caption)

@st.dialog("How the Algorithm Works", width="large")
def show_algorithm_info():
    doc_name = "algorithm_de.md" if st.session_state.language == "de" else "algorithm.md"
    with open(os.path.join(DOCS_DIR, doc_name), encoding="utf-8") as f:
        content = f.read()
    with st.container(height=1000):
        render_doc_markdown(content)

def algorithm_info_button(key, label):
    if st.button(label, key=key, type="tertiary"):
        show_algorithm_info()

def run_ocr_zoom_detection(original_img):
    """Scan the bottom 20% of the image for a µm scale value via Tesseract OCR.
    Updates detected_zoom, zoom_notification, and the zoom_factor_input widget value.
    """
    height, width = original_img.shape[:2]
    bottom_20 = original_img[int(height * 0.8):height, 0:width]

    try:
        gray_bottom = cv2.cvtColor(bottom_20, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_bottom, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh)
        numbers = re.findall(r'\b\d+\b', text)

        if numbers:
            st.session_state.detected_zoom = int(numbers[-1])
            st.session_state.zoom_notification = {"type": "success", "key": "ocr_success", "value": st.session_state.detected_zoom}
        else:
            st.session_state.detected_zoom = 5000
            st.session_state.zoom_notification = {"type": "warning", "key": "ocr_warning"}
    except Exception:
        st.session_state.detected_zoom = 5000
        st.session_state.zoom_notification = {"type": "error", "key": "ocr_error"}

    st.session_state.zoom_factor_input = st.session_state.detected_zoom

# Simplified stand-in for the Advanced Options open/close kernel ratio + iteration sliders:
# how large a dark region can be before it's still folded into the sample's
# area-of-interest mask. Picking a preset here writes directly into the same
# open_idx/close_idx/open_iters/close_iters session_state keys the Advanced Options
# sliders use, so adjusting those sliders afterward naturally overrides the preset --
# there's a single source of truth, whichever was set most recently.
HOLE_SIZE_PRESETS = {
    "small":  {"open_idx": 1, "close_idx": 0, "open_iters": 1, "close_iters": 1},
    "medium": {"open_idx": 1, "close_idx": 1, "open_iters": 1, "close_iters": 1},
    "large":  {"open_idx": 1, "close_idx": 2, "open_iters": 1, "close_iters": 1},
    "big":    {"open_idx": 1, "close_idx": 3, "open_iters": 1, "close_iters": 1},
}

def apply_pore_size_preset():
    """on_change callback for the Mask Setting panel's Pore Size slider."""
    preset = HOLE_SIZE_PRESETS[st.session_state.hole_size]
    st.session_state.open_idx = preset['open_idx']
    st.session_state.close_idx = preset['close_idx']
    st.session_state.open_iters = preset['open_iters']
    st.session_state.close_iters = preset['close_iters']

# Default values for every user-adjustable setting. Used both to seed
# session state and to back the "Reset Default Parameters" button.
PARAM_DEFAULTS = {
    'sample_type': 'unetched',
    'enable_ocr': False,
    'fast_mask': True,
    'processing_size': 512,
    'zoom_factor_input': 5000,
    'adjust_pore_size': False,
    'hole_size': 'small',
    'adjust_mask_shrink': False,
    'mask_shrink_idx': 0,
    'crop_legend': False,
    'crop_height_unit': 'pixels',
    'crop_height': 100,
    'crop_height_percent': 10,
    'grain_lines': 5,
    'open_idx': 1,
    'close_idx': 0,
    'open_iters': 1,
    'close_iters': 1,
    'border_pixels': 0,
    'border_ratio_idx': 1,
    'threshold': 0,
    'mask_threshold': 0,
    'highlight_biggest_pores': False,
}

def init_session_state():
    """Initialize session state variables to store images and results in memory."""
    defaults = {
        'original_img': None,
        'binary_img': None,
        'mask': None,
        'pores_mask': None,
        'border_mask': None,
        'grain_overlay': None,
        'full_porosity': None,
        'border_porosity': None,
        'grain_size_um': None,
        'grain_intersections': None,
        'erased_boxes': [],
        'last_uploaded_name': None,
        'needs_calc': False,
        'detected_zoom': 5000,
        'zoom_notification': None,
        'ocr_done_for_current_image': False,
        'pending_reset': False,
        'current_canvas_bg': None,
        'current_mask_bg': None,
        'language': 'en',
        'view_selection': 'original_draw',
        **PARAM_DEFAULTS,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def main():
    init_session_state()

    # Apply a pending "Reset Default Parameters" click here, before any of the
    # affected widgets render below -- writing to their session_state keys
    # after they've already rendered this run raises a StreamlitAPIException.
    if st.session_state.pending_reset:
        for key, val in PARAM_DEFAULTS.items():
            st.session_state[key] = val
        st.session_state.detected_zoom = PARAM_DEFAULTS['zoom_factor_input']
        st.session_state.zoom_notification = None
        st.session_state.ocr_done_for_current_image = False
        st.session_state.pending_reset = False
        st.session_state.needs_calc = True

    with st.sidebar:
        st.radio(
            "Language", options=["en", "de"], key="language", horizontal=True,
            format_func=lambda k: "English" if k == "en" else "Deutsch",
            label_visibility="collapsed"
        )

    st.title(t("app_title"))
    st.markdown("---")

    # ==========================
    # 1. FILE UPLOAD & PRE-PROCESSING
    # ==========================
    # We must process the file *before* rendering the sidebar widgets
    # so they can naturally adopt the updated zoom values without a st.rerun()
    with st.sidebar:
        st.header(t("controls_header"))
        uploaded_file = st.file_uploader(t("upload_image"), type=['jpg', 'png', 'bmp'])

    if uploaded_file is not None:
        if st.session_state.last_uploaded_name != uploaded_file.name:
            # Read and store the image firmly in session state first
            image = Image.open(uploaded_file)
            original_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            st.session_state.original_img = original_img
            st.session_state.last_uploaded_name = uploaded_file.name
            st.session_state.erased_boxes = []
            st.session_state.needs_calc = True
            st.session_state.ocr_done_for_current_image = False
            st.session_state.zoom_notification = None
            # zoom_factor_input is deliberately NOT reset here -- it should carry over
            # from the previous image (most batches share the same scale/zoom) unless
            # OCR is enabled and detects a different value for this specific image.

        # Run OCR scale detection once per image, whenever it's enabled. This must
        # happen here, before the sidebar widgets render below, so that a freshly
        # uploaded image (OCR already on) and OCR being ticked on for the current
        # image both pick up the result in the same run -- updating
        # zoom_factor_input after its widget has rendered would raise a
        # StreamlitAPIException.
        if st.session_state.enable_ocr and not st.session_state.ocr_done_for_current_image:
            run_ocr_zoom_detection(st.session_state.original_img)
            st.session_state.ocr_done_for_current_image = True

    # ==========================
    # 2. SIDEBAR SETTINGS
    # ==========================
    # Rendered unconditionally, even before any image is uploaded. When these ~15+
    # widgets used to only appear once original_img was set, they all mounted for the
    # first time in the very same render pass as the upload event -- which reliably
    # desynced some of them on the frontend (verified: the widget would visually show
    # its blank fallback value even though the backend's session_state already held the
    # correct one, and no later rerun ever corrected the display). Mounting them upfront,
    # before any upload occurs, means the file upload only ever has to update already-
    # hydrated widgets instead of mounting a large batch of new ones simultaneously.
    with st.sidebar:
        st.radio(
            t("sample_type_label"),
            options=SAMPLE_TYPE_KEYS,
            format_func=_localized_format_func(None, st.session_state.language),
            key="sample_type",
            help=t("sample_type_help")
        )

        st.subheader(t("basic_settings"))
        st.checkbox(
            t("ocr_label"),
            key="enable_ocr",
            help=t("ocr_help")
        )
        fast_mask = st.checkbox(
            t("fast_mask_label"),
            key="fast_mask",
            help=t("fast_mask_help")
        )
        if fast_mask:
            processing_size = st.number_input(
                t("processing_size_label"), min_value=128, max_value=2048, step=64, key="processing_size",
                help=t("processing_size_help")
            )
        else:
            processing_size = st.session_state.processing_size
        zoom_factor = st.number_input(
            t("zoom_factor_label"), step=10, key="zoom_factor_input",
            help=t("zoom_factor_help")
        )

        st.subheader(t("mask_setting"))
        col_pore_cb, col_pore_demo = st.columns([5, 2])
        with col_pore_cb:
            st.checkbox(
                t("adjust_pore_size_label"),
                key="adjust_pore_size",
                help=t("adjust_pore_size_help")
            )
        with col_pore_demo:
            demo_video_button(
                "pore_size_demo_btn", "pore_size.mp4",
                t("pore_size_demo_heading"), t("pore_size_demo_caption"), t("pore_size_demo_tooltip")
            )
        if st.session_state.adjust_pore_size:
            st.select_slider(
                t("pore_size_label"), options=HOLE_SIZE_KEYS, key="hole_size", on_change=apply_pore_size_preset,
                format_func=_localized_format_func(HOLE_SIZE_LABEL_KEYS, st.session_state.language),
                help=t("pore_size_help")
            )
        col_shrink_cb, col_shrink_demo = st.columns([5, 2])
        with col_shrink_cb:
            st.checkbox(
                t("adjust_mask_shrink_label"),
                key="adjust_mask_shrink",
                help=t("adjust_mask_shrink_help")
            )
        with col_shrink_demo:
            demo_video_button(
                "mask_shrink_demo_btn", "shrink.mp4",
                t("mask_shrink_demo_heading"), t("mask_shrink_demo_caption"), t("mask_shrink_demo_tooltip")
            )
        shrink_ratios = {0: 0.0, 1: 0.01, 2: 0.02, 3: 0.04, 4: 0.07, 5: 0.10}
        if st.session_state.adjust_mask_shrink:
            mask_shrink_idx = st.slider(
                t("shrink_mask_label"), 0, 5, key="mask_shrink_idx",
                help=t("shrink_mask_help")
            )
            effective_shrink_ratio = shrink_ratios[mask_shrink_idx]
        else:
            effective_shrink_ratio = 0.0
        crop_legend = st.checkbox(
            t("crop_legend_label"),
            key="crop_legend",
            help=t("crop_legend_help")
        )
        crop_height = None
        crop_height_percent = None
        if crop_legend:
            st.radio(
                t("crop_height_unit_label"), options=CROP_UNIT_KEYS, key="crop_height_unit",
                format_func=_localized_format_func(CROP_UNIT_LABEL_KEYS, st.session_state.language),
                help=t("crop_height_unit_help"), horizontal=True
            )
            if st.session_state.crop_height_unit == "pixels":
                crop_height = st.number_input(
                    t("crop_height_label"), min_value=1, key="crop_height",
                    help=t("crop_height_help")
                )
            else:
                crop_height_percent = st.number_input(
                    t("crop_height_percent_label"), min_value=1, max_value=100, key="crop_height_percent",
                    help=t("crop_height_percent_help")
                )

        with st.expander(t("advanced_setting")):
            st.warning(t("advanced_warning"))
            algorithm_info_button("advanced_algorithm_info_btn", t("algorithm_link_advanced"))

            with st.container(border=True):
                st.markdown(t("morphology_setting_header"))
                kernel_options = {0: 1/400, 1: 1/200, 2: 1/100, 3: 1/50, 4: 1/25, 5: 1/10}
                open_idx = st.slider(
                    t("open_kernel_label"), 0, 5, key="open_idx",
                    help=t("open_kernel_help")
                )
                close_idx = st.slider(
                    t("close_kernel_label"), 0, 5, key="close_idx",
                    help=t("close_kernel_help")
                )
                open_iters = st.slider(
                    t("open_iters_label"), 0, 5, key="open_iters",
                    help=t("open_iters_help")
                )
                close_iters = st.slider(
                    t("close_iters_label"), 0, 5, key="close_iters",
                    help=t("close_iters_help")
                )

            with st.container(border=True):
                st.markdown(t("border_setting_header"))
                border_pixels = st.number_input(
                    t("border_width_label"), min_value=0, key="border_pixels",
                    help=t("border_width_help")
                )
                border_pixels = border_pixels if border_pixels > 0 else None

                border_ratios = {0: 0.05, 1: 0.10, 2: 0.15, 3: 0.20}
                border_ratio_idx = st.slider(
                    t("border_ratio_label"), 0, 3, key="border_ratio_idx",
                    help=t("border_ratio_help")
                )

            with st.container(border=True):
                st.markdown(t("threshold_setting_header"))
                threshold = st.slider(
                    t("threshold_label"), 0, 255, key="threshold",
                    help=t("threshold_help")
                )
                mask_threshold = st.slider(
                    t("mask_threshold_label"), 0, 255, key="mask_threshold",
                    help=t("mask_threshold_help")
                )

        if st.session_state.sample_type == "etched":
            with st.expander(t("grain_size_calc_header")):
                grain_lines = st.slider(
                    t("grain_lines_label"), min_value=1, max_value=20, key="grain_lines",
                    help=t("grain_lines_help")
                )
        else:
            grain_lines = st.session_state.grain_lines

        st.subheader(t("analysis_actions"))
        has_param_changes = any(
            st.session_state[key] != default for key, default in PARAM_DEFAULTS.items()
        )
        if st.button(
            t("reset_button"),
            disabled=not has_param_changes,
            use_container_width=True
        ):
            st.session_state.pending_reset = True
            st.rerun()
        if st.button(t("recalculate_button"), type="primary", use_container_width=True):
            st.session_state.needs_calc = True

    # ==========================
    # 3. MAIN CONTENT AREA (Calculations & Results)
    # ==========================
    if st.session_state.original_img is None:
        tutorial_name = "tutorial_de.md" if st.session_state.language == "de" else "tutorial.md"
        with open(os.path.join(DOCS_DIR, tutorial_name), encoding="utf-8") as f:
            render_doc_markdown(f.read())
        algorithm_info_button("tutorial_algorithm_info_btn", t("algorithm_link_tutorial"))

    if st.session_state.original_img is not None:
        img_cv = st.session_state.original_img

        # Percentage-based crop height can only be resolved to actual pixels once the
        # image (and thus its height) is known, which isn't until this point.
        if crop_legend and crop_height_percent is not None:
            effective_crop_height = int(img_cv.shape[0] * crop_height_percent / 100)
        else:
            effective_crop_height = crop_height

        # --- Centralized Calculation Block ---
        if st.session_state.needs_calc:
            with st.spinner(t("analyzing_spinner")):
                porosity, binary, mask, combined, b_porosity, b_combined, b_mask = analyze_porosity(
                    img_cv,
                    crop_legend_enabled=crop_legend,
                    crop_height=effective_crop_height,
                    open_kernel_ratio=kernel_options[open_idx],
                    close_kernel_ratio=kernel_options[close_idx],
                    manual_threshold=threshold,
                    mask_threshold=mask_threshold,
                    use_area_of_interest=True,
                    open_iterations=open_iters,
                    close_iterations=close_iters,
                    border_pixels=border_pixels,
                    border_ratio=border_ratios[border_ratio_idx],
                    fast_mask_enabled=fast_mask,
                    processing_size=processing_size,
                    shrink_ratio=effective_shrink_ratio
                )

                if st.session_state.erased_boxes:
                    for box in st.session_state.erased_boxes:
                        x0, y0, w, h = box['x'], box['y'], box['w'], box['h']
                        mask[y0:y0+h, x0:x0+w] = 0

                    inverse_binary = cv2.bitwise_not(binary)
                    combined = cv2.bitwise_and(inverse_binary, mask)
                    whole_area = np.count_nonzero(mask == 255)
                    pore_area = np.count_nonzero(combined == 255)
                    porosity = (pore_area * 100 / whole_area) if whole_area > 0 else 0

                    b_mask = select_border_of_interest(mask, border_pixels, border_ratios[border_ratio_idx])
                    b_combined = cv2.bitwise_and(inverse_binary, b_mask)
                    b_area = np.count_nonzero(b_mask == 255)
                    b_pore_area = np.count_nonzero(b_combined == 255)
                    b_porosity = (b_pore_area * 100 / b_area) if b_area > 0 else 0

                if st.session_state.sample_type == "etched":
                    intersections, mean_intercept_px, overlay = calculate_grain_size(
                        img_cv,
                        mask,
                        grain_lines,
                        threshold
                    )
                    pixel_size_um = zoom_factor / img_cv.shape[1]
                    st.session_state.grain_size_um = mean_intercept_px * pixel_size_um
                    st.session_state.grain_intersections = intersections
                    st.session_state.grain_overlay = overlay
                else:
                    st.session_state.grain_size_um = None
                    st.session_state.grain_intersections = None
                    st.session_state.grain_overlay = None

                st.session_state.mask = mask
                st.session_state.pores_mask = combined
                st.session_state.border_mask = b_mask
                st.session_state.binary_img = binary
                st.session_state.full_porosity = porosity
                st.session_state.border_porosity = b_porosity
                st.session_state.needs_calc = False
                # Removed the st.rerun() here entirely!

        # Display Results Metric Cards
        if st.session_state.full_porosity is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric(t("metric_full_porosity"), f"{st.session_state.full_porosity:.2f}%")
            c2.metric(t("metric_border_porosity"), f"{st.session_state.border_porosity:.2f}%")

            if st.session_state.grain_size_um is not None:
                c3.metric(t("metric_grain_size"), f"{st.session_state.grain_size_um:.2f} µm")

        if st.session_state.zoom_notification:
            notif = st.session_state.zoom_notification
            msg = t(notif["key"], value=notif["value"]) if "value" in notif else t(notif["key"])
            if notif["type"] == "success":
                st.success(msg)
            elif notif["type"] == "warning":
                st.warning(msg)
            else:
                st.error(msg)

        # ==========================
        # 4. VISUALIZER RENDERING
        # ==========================
        view_keys = ["original_draw", "area_mask", "pores", "border"]
        if st.session_state.sample_type == "etched":
            view_keys.append("grain_grid")
        view_selection = st.radio(
            t("select_view_label"), view_keys,
            format_func=_localized_format_func(VIEW_LABEL_KEYS, st.session_state.language), horizontal=True,
            key="view_selection"
        )

        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        MAX_CANVAS_WIDTH = 1000

        if view_selection == "original_draw":
            st.info(t("draw_info"))

            if pil_img.width > MAX_CANVAS_WIDTH:
                scale_factor = pil_img.width / MAX_CANVAS_WIDTH
                display_width = MAX_CANVAS_WIDTH
                display_height = int(pil_img.height / scale_factor)
                display_img = pil_img.resize((display_width, display_height), Image.Resampling.LANCZOS)
            else:
                scale_factor = 1.0
                display_width = pil_img.width
                display_height = pil_img.height
                display_img = pil_img

            # Safely store canvas backgrounds to prevent garbage collector death
            st.session_state.current_canvas_bg = display_img

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=2,
                stroke_color="red",
                background_image=st.session_state.current_canvas_bg,
                update_streamlit=True,
                height=display_height,
                width=display_width,
                drawing_mode="rect",
                key="canvas_original",
            )

            col_region_btn, col_region_demo = st.columns([5, 1])
            with col_region_demo:
                demo_video_button(
                    "calculate_region_demo_btn", "calculate_region.mp4",
                    t("calc_region_demo_heading"), t("calc_region_demo_caption"), t("calc_region_demo_tooltip")
                )
            if col_region_btn.button(t("calc_region_button"), type="primary", use_container_width=True):
                if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                    last_shape = canvas_result.json_data["objects"][-1]

                    x0 = int(last_shape["left"] * scale_factor)
                    y0 = int(last_shape["top"] * scale_factor)
                    w = int(last_shape["width"] * scale_factor)
                    h = int(last_shape["height"] * scale_factor)

                    cropped_img = img_cv[y0:y0+h, x0:x0+w]

                    if cropped_img.size > 0:
                        reg_por, _, _, reg_comb, _, _, _ = analyze_porosity(
                            cropped_img,
                            use_area_of_interest=False,
                            border_ratio=0.10,
                            open_kernel_ratio=kernel_options[open_idx],
                            close_kernel_ratio=kernel_options[close_idx],
                            fast_mask_enabled=fast_mask,
                            processing_size=processing_size
                        )
                        st.success(t("region_result", value=reg_por))
                else:
                    st.warning(t("draw_rect_warning"))

        elif view_selection == "area_mask":
            if st.session_state.mask is not None:
                st.info(t("erase_info"))

                blended_arr = img_rgb.copy().astype(np.float32)
                active_mask = st.session_state.mask == 255

                blended_arr[active_mask] = blended_arr[active_mask] * 0.5 + np.array([255, 0, 0]) * 0.5

                PAD = 150
                padded_arr = np.pad(blended_arr, ((PAD, PAD), (PAD, PAD), (0, 0)), mode='constant', constant_values=40)
                mask_bg = Image.fromarray(padded_arr.astype(np.uint8))

                if mask_bg.width > MAX_CANVAS_WIDTH:
                    scale_factor_mask = mask_bg.width / MAX_CANVAS_WIDTH
                    display_width_mask = MAX_CANVAS_WIDTH
                    display_height_mask = int(mask_bg.height / scale_factor_mask)
                    display_img_mask = mask_bg.resize((display_width_mask, display_height_mask), Image.Resampling.LANCZOS)
                else:
                    scale_factor_mask = 1.0
                    display_width_mask = mask_bg.width
                    display_height_mask = mask_bg.height
                    display_img_mask = mask_bg

                # Safely store canvas backgrounds to prevent garbage collector death
                st.session_state.current_mask_bg = display_img_mask

                canvas_mask = st_canvas(
                    fill_color="rgba(255, 255, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="yellow",
                    background_image=st.session_state.current_mask_bg,
                    update_streamlit=True,
                    height=display_height_mask,
                    width=display_width_mask,
                    drawing_mode="rect",
                    key="canvas_mask",
                )

                col_c, col_d, col_demo = st.columns([3, 3, 2])
                with col_demo:
                    demo_video_button(
                        "erase_selection_demo_btn", "erase_selection.mp4",
                        t("erase_demo_heading"), t("erase_demo_caption"), t("erase_demo_tooltip")
                    )

                if col_c.button(t("erase_selection_button"), type="primary", use_container_width=True):
                    if canvas_mask.json_data is not None and len(canvas_mask.json_data["objects"]) > 0:
                        img_h, img_w = st.session_state.mask.shape

                        for shape in canvas_mask.json_data["objects"]:
                            x0_padded = int(shape["left"] * scale_factor_mask)
                            y0_padded = int(shape["top"] * scale_factor_mask)
                            w_padded = int(shape["width"] * scale_factor_mask)
                            h_padded = int(shape["height"] * scale_factor_mask)

                            x0 = x0_padded - PAD
                            y0 = y0_padded - PAD
                            x1 = x0 + w_padded
                            y1 = y0 + h_padded

                            x0 = max(0, min(x0, img_w))
                            y0 = max(0, min(y0, img_h))
                            x1 = max(0, min(x1, img_w))
                            y1 = max(0, min(y1, img_h))

                            if x1 > x0 and y1 > y0:
                                w_real = x1 - x0
                                h_real = y1 - y0
                                st.session_state.erased_boxes.append({'x': x0, 'y': y0, 'w': w_real, 'h': h_real})
                                st.session_state.mask[y0:y1, x0:x1] = 0

                        inverse_binary = cv2.bitwise_not(st.session_state.binary_img)
                        st.session_state.pores_mask = cv2.bitwise_and(inverse_binary, st.session_state.mask)

                        whole_area = np.count_nonzero(st.session_state.mask == 255)
                        pore_area = np.count_nonzero(st.session_state.pores_mask == 255)
                        st.session_state.full_porosity = (pore_area * 100 / whole_area) if whole_area > 0 else 0

                        # We still need a rerun here specifically to apply the mask changes,
                        # but because it's tied to an explicit button click AFTER load, it won't kill images
                        st.rerun()
                    else:
                        st.warning(t("draw_rect_warning2"))

                if col_d.button(t("reset_erasures_button"), use_container_width=True):
                    st.session_state.erased_boxes = []
                    st.warning(t("erasures_cleared_warning"))
            else:
                st.info(t("upload_to_process_mask"))

        elif view_selection == "pores":
            if st.session_state.pores_mask is not None:
                blended_arr = img_rgb.copy().astype(np.float32)
                active_mask = st.session_state.pores_mask == 255
                blended_arr[active_mask] = blended_arr[active_mask] * 0.4 + np.array([0, 150, 255]) * 0.6
                display_arr = blended_arr.astype(np.uint8)

                highlight_biggest = st.checkbox(
                    t("highlight_biggest_label"),
                    key="highlight_biggest_pores",
                    help=t("highlight_biggest_help")
                )
                if highlight_biggest:
                    contours, _ = cv2.findContours(
                        st.session_state.pores_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    biggest = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

                    if biggest:
                        pixel_size_um = (zoom_factor / img_cv.shape[1]) if zoom_factor and zoom_factor > 0 else 0
                        summary_lines = []

                        # Scale box/text proportionally to the image so labels stay
                        # legible on both small crops and multi-thousand-pixel scans.
                        img_h, img_w = display_arr.shape[:2]
                        ref_dim = min(img_h, img_w)
                        box_thickness = max(2, min(8, ref_dim // 500))
                        font_scale = max(0.6, min(3.0, ref_dim / 1000))
                        text_thickness = max(2, box_thickness - 1)
                        margin = max(6, ref_dim // 150)

                        for rank, contour in enumerate(biggest, start=1):
                            x, y, w, h = cv2.boundingRect(contour)
                            (_, _), radius = cv2.minEnclosingCircle(contour)
                            diameter_px = radius * 2

                            if pixel_size_um > 0:
                                diameter_label = f"{diameter_px * pixel_size_um:.1f} um"
                                summary_lines.append(f"#{rank}: {diameter_px * pixel_size_um:.1f} µm")
                            else:
                                diameter_label = f"{diameter_px:.0f} px"
                                summary_lines.append(f"#{rank}: {diameter_px:.0f} px")

                            cv2.rectangle(display_arr, (x, y), (x + w, y + h), (255, 0, 0), box_thickness)

                            text = f"#{rank} {diameter_label}"
                            (text_w, text_h), _ = cv2.getTextSize(
                                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
                            )
                            text_x = min(max(x, 0), max(img_w - text_w - margin, 0))
                            text_y = y - margin
                            if text_y - text_h < 0:
                                # Not enough room above the box -- place the label just below the top edge instead
                                text_y = y + text_h + margin
                            cv2.putText(
                                display_arr, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale, (255, 0, 0), text_thickness, cv2.LINE_AA
                            )

                        st.caption("  |  ".join(summary_lines))
                    else:
                        st.info(t("no_pores_info"))

                st.image(Image.fromarray(display_arr), use_container_width=True)

        elif view_selection == "border":
            if st.session_state.border_mask is not None:
                blended_arr = img_rgb.copy().astype(np.float32)
                active_mask = st.session_state.border_mask == 255
                blended_arr[active_mask] = blended_arr[active_mask] * 0.4 + np.array([255, 255, 0]) * 0.6

                st.image(Image.fromarray(blended_arr.astype(np.uint8)), use_container_width=True)

        elif view_selection == "grain_grid":
            if st.session_state.grain_overlay is not None:
                blended_arr = img_rgb.copy().astype(np.float32)
                overlay_rgba = st.session_state.grain_overlay

                alpha = overlay_rgba[:, :, 3] / 255.0
                alpha_rgb = np.stack([alpha, alpha, alpha], axis=-1)
                overlay_rgb = overlay_rgba[:, :, :3].astype(np.float32)

                blended_arr = (blended_arr * (1 - alpha_rgb)) + (overlay_rgb * alpha_rgb)

                st.image(Image.fromarray(blended_arr.astype(np.uint8)), use_container_width=True)

if __name__ == "__main__":
    main()
