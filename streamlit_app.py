import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pytesseract
import re
import shutil
import os

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
            st.session_state.zoom_notification = {
                "type": "success",
                "msg": f"✅ **Scale Detected:** Tesseract OCR successfully scanned the bottom 20% of the image and found a zoom factor of **{st.session_state.detected_zoom} µm**. The scale has been automatically updated in the sidebar."
            }
        else:
            st.session_state.detected_zoom = 5000
            st.session_state.zoom_notification = {
                "type": "warning",
                "msg": "⚠️ **Scale Not Found:** Tesseract OCR scanned the bottom of the image but could not detect any numbers. The zoom factor has been set to the default **5000 µm**."
            }
    except Exception:
        st.session_state.detected_zoom = 5000
        st.session_state.zoom_notification = {
            "type": "error",
            "msg": "❌ **OCR Error:** Tesseract OCR failed to run or is not properly configured on your system. Defaulting to **5000 µm**."
        }

    st.session_state.zoom_factor_input = st.session_state.detected_zoom

# Simplified stand-in for the Advanced Options open/close kernel ratio + iteration sliders:
# how large a dark region can be before it's still folded into the sample's
# area-of-interest mask. Picking a preset here writes directly into the same
# open_idx/close_idx/open_iters/close_iters session_state keys the Advanced Options
# sliders use, so adjusting those sliders afterward naturally overrides the preset --
# there's a single source of truth, whichever was set most recently.
HOLE_SIZE_LABELS = ["Small Pores", "Medium", "Large", "Big Pores"]
HOLE_SIZE_PRESETS = {
    "Small Pores": {"open_idx": 1, "close_idx": 0, "open_iters": 1, "close_iters": 1},
    "Medium":      {"open_idx": 1, "close_idx": 1, "open_iters": 1, "close_iters": 1},
    "Large":       {"open_idx": 1, "close_idx": 2, "open_iters": 1, "close_iters": 1},
    "Big Pores":   {"open_idx": 1, "close_idx": 3, "open_iters": 1, "close_iters": 1},
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
    'sample_type': 'Ungeätzt',
    'enable_ocr': False,
    'fast_mask': True,
    'processing_size': 512,
    'zoom_factor_input': 5000,
    'adjust_pore_size': False,
    'hole_size': 'Small Pores',
    'crop_legend': False,
    'crop_height': 100,
    'grain_lines': 5,
    'open_idx': 1,
    'close_idx': 0,
    'open_iters': 1,
    'close_iters': 1,
    'border_pixels': 0,
    'border_ratio_idx': 1,
    'threshold': 0,
    'mask_threshold': 0,
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

    st.title("Porosity & Grain Analysis Tool")
    st.markdown("---")
    
    # ==========================
    # 1. FILE UPLOAD & PRE-PROCESSING
    # ==========================
    # We must process the file *before* rendering the sidebar widgets 
    # so they can naturally adopt the updated zoom values without a st.rerun()
    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'bmp'])

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
    # 2. SIDEBAR SETTINGS (Rendered after processing upload)
    # ==========================
    if uploaded_file is not None:
        with st.sidebar:
            st.radio(
                "Sample Type",
                options=["Ungeätzt", "Geätzt"],
                key="sample_type",
                help="Ungeätzt (unetched): porosity analysis only. Geätzt (etched): grain boundaries are visible, so grain size is measured too."
            )

            st.subheader("Basic Settings")
            st.checkbox(
                "Auto-detect Scale (OCR)",
                key="enable_ocr",
                help="Uses Tesseract OCR to read the µm scale from the bottom of newly uploaded images. Requires Tesseract to be installed on this machine."
            )
            fast_mask = st.checkbox(
                "Fast Mask Processing",
                key="fast_mask",
                help="Downsizes the image before detecting the sample boundary, for speed. Turn off to classify the boundary at full resolution instead."
            )
            if fast_mask:
                processing_size = st.number_input(
                    "Processing Size (px)", min_value=128, max_value=2048, step=64, key="processing_size",
                    help="The image is downsized so its largest side is at most this many pixels before detecting the sample boundary. Higher = more precise but slower."
                )
            else:
                processing_size = st.session_state.processing_size
            zoom_factor = st.number_input(
                "Zoom Factor (µm)", step=10, key="zoom_factor_input",
                help="The real-world width of the image in micrometers, used to convert pixel measurements into grain size (µm). Auto-filled by OCR if enabled above."
            )

            st.subheader("Mask Setting")
            st.checkbox(
                "Adjust Pore Size",
                key="adjust_pore_size",
                help="Lets you tune how large a dark region can be before it's still folded into the sample area, instead of using the default."
            )
            if st.session_state.adjust_pore_size:
                st.select_slider(
                    "Pore Size", options=HOLE_SIZE_LABELS, key="hole_size", on_change=apply_pore_size_preset,
                    help="Small Pores only folds tiny specks into the sample area; Big Pores tolerates larger dark regions as part of the sample boundary. Adjusting Close/Open Kernel Ratio directly in Advanced Options below overrides this preset."
                )
            crop_legend = st.checkbox(
                "Crop Legend",
                key="crop_legend",
                help="Blacks out a strip at the bottom of the image (e.g. a scale bar or legend) before analysis, so it isn't mistaken for sample or pore area."
            )
            crop_height = st.number_input(
                "Crop Height (px from bottom)", min_value=1, key="crop_height",
                help="How many pixels tall the cropped-out strip is, measured up from the bottom of the image."
            ) if crop_legend else None

            with st.expander("Porosity Setting"):
                st.markdown("**Morphology Setting** (overrides the Mask Setting panel's Pore Size preset)")
                kernel_options = {0: 1/400, 1: 1/200, 2: 1/100, 3: 1/50, 4: 1/25, 5: 1/10}
                open_idx = st.slider(
                    "Open Kernel Ratio (0-5)", 0, 5, key="open_idx",
                    help="Removes small noise specks from the sample boundary mask. Higher = removes larger specks, but can erode real detail."
                )
                close_idx = st.slider(
                    "Close Kernel Ratio (0-5)", 0, 5, key="close_idx",
                    help="How large a dark region can be before it's still folded into the sample area mask. Higher = tolerates bigger dark regions as part of the sample."
                )
                open_iters = st.slider(
                    "Open Iterations", 0, 5, key="open_iters",
                    help="How many times the noise-removal (open) step repeats. Higher = more aggressive cleanup."
                )
                close_iters = st.slider(
                    "Close Iterations", 0, 5, key="close_iters",
                    help="How many times the hole-filling (close) step repeats. Higher = more aggressive filling."
                )

                st.markdown("**Border Porosity Setting**")
                border_pixels = st.number_input(
                    "Border Width (px)", min_value=0, key="border_pixels",
                    help="Fixed border thickness, in pixels, used for the Border Porosity measurement. Leave at 0 to use Border Ratio Level instead."
                )
                border_pixels = border_pixels if border_pixels > 0 else None

                border_ratios = {0: 0.05, 1: 0.10, 2: 0.15, 3: 0.20}
                border_ratio_idx = st.slider(
                    "Border Ratio Level (0-3)", 0, 3, key="border_ratio_idx",
                    help="Only used when Border Width is 0. Sets the border ring's thickness as a percentage of the sample's estimated radius -- higher levels use a thicker ring."
                )

                st.markdown("**Threshold Setting**")
                threshold = st.slider(
                    "Threshold (0 = Auto)", 0, 255, key="threshold",
                    help="Manual brightness cutoff (0-255) used to separate pores from solid material. 0 lets the app choose automatically (Otsu's method)."
                )
                mask_threshold = st.slider(
                    "Mask Threshold (0 = Auto)", 0, 255, key="mask_threshold",
                    help="Manual brightness cutoff (0-255) used to detect the sample's outer boundary (area of interest). 0 lets the app choose automatically (Otsu's method)."
                )

            if st.session_state.sample_type == "Geätzt":
                with st.expander("Grain Size Calculation"):
                    grain_lines = st.slider(
                        "Grain Size Grid Lines (H & V)", min_value=1, max_value=20, key="grain_lines",
                        help="Number of horizontal and vertical test lines used to measure grain size via the line-intercept method. More lines sample more of the image but take longer."
                    )
            else:
                grain_lines = st.session_state.grain_lines

            st.subheader("Analysis Actions")
            has_param_changes = any(
                st.session_state[key] != default for key, default in PARAM_DEFAULTS.items()
            )
            if st.button(
                "Reset Default Parameters",
                disabled=not has_param_changes,
                use_container_width=True
            ):
                st.session_state.pending_reset = True
                st.rerun()
            if st.button("recalculate", type="primary", use_container_width=True):
                st.session_state.needs_calc = True

    # ==========================
    # 3. MAIN CONTENT AREA (Calculations & Results)
    # ==========================
    if uploaded_file is not None:
        img_cv = st.session_state.original_img
        
        # --- Centralized Calculation Block ---
        if st.session_state.needs_calc:
            with st.spinner("Analyzing image..."):
                porosity, binary, mask, combined, b_porosity, b_combined, b_mask = analyze_porosity(
                    img_cv,
                    crop_legend_enabled=crop_legend,
                    crop_height=crop_height,
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
                    processing_size=processing_size
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
                
                if st.session_state.sample_type == "Geätzt":
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
            c1.metric("Full Image Porosity", f"{st.session_state.full_porosity:.2f}%")
            c2.metric("Border Porosity", f"{st.session_state.border_porosity:.2f}%")
            
            if st.session_state.grain_size_um is not None:
                c3.metric("Mean Grain Size", f"{st.session_state.grain_size_um:.2f} µm")

        if st.session_state.zoom_notification:
            if st.session_state.zoom_notification["type"] == "success":
                st.success(st.session_state.zoom_notification["msg"])
            elif st.session_state.zoom_notification["type"] == "warning":
                st.warning(st.session_state.zoom_notification["msg"])
            else:
                st.error(st.session_state.zoom_notification["msg"])

        # ==========================
        # 4. VISUALIZER RENDERING
        # ==========================
        view_options = ["Original / Draw", "Area Mask", "Pores", "Border"]
        if st.session_state.sample_type == "Geätzt":
            view_options.append("Grain Grid")
        view_selection = st.radio("Select View", view_options, horizontal=True)
        
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        MAX_CANVAS_WIDTH = 1000 
        
        if view_selection == "Original / Draw":
            st.info("Draw rectangles on the image below to outline specific regions. Then use the buttons below the image to process your selection.")
            
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
            
            if st.button("Calculate Selected Region Porosity", type="primary", use_container_width=True):
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
                        st.success(f"Selected Region Porosity: {reg_por:.2f}%")
                else:
                    st.warning("Please draw a rectangle first.")

        elif view_selection == "Area Mask":
            if st.session_state.mask is not None:
                st.info("Draw rectangles on the mask to erase regions. Click 'Erase Selection' to apply. Erased regions remain across calculations until reset.")
                
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
                
                col_c, col_d = st.columns(2)

                if col_c.button("Erase Selection", type="primary", use_container_width=True):
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
                        st.warning("Please draw at least one rectangle first.")
                
                if col_d.button("Reset Erasures", use_container_width=True):
                    st.session_state.erased_boxes = []
                    st.warning("Erasures cleared. Click 'recalculate' in the sidebar to restore the default mask.")
            else:
                st.info("Upload an image to process the mask.")

        elif view_selection == "Pores":
            if st.session_state.pores_mask is not None:
                blended_arr = img_rgb.copy().astype(np.float32)
                active_mask = st.session_state.pores_mask == 255
                blended_arr[active_mask] = blended_arr[active_mask] * 0.4 + np.array([0, 150, 255]) * 0.6
                
                st.image(Image.fromarray(blended_arr.astype(np.uint8)), use_container_width=True)

        elif view_selection == "Border":
            if st.session_state.border_mask is not None:
                blended_arr = img_rgb.copy().astype(np.float32)
                active_mask = st.session_state.border_mask == 255
                blended_arr[active_mask] = blended_arr[active_mask] * 0.4 + np.array([255, 255, 0]) * 0.6
                
                st.image(Image.fromarray(blended_arr.astype(np.uint8)), use_container_width=True)

        elif view_selection == "Grain Grid":
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