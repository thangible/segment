import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Import your existing backend logic
from segment import analyze_porosity, calculate_grain_size, select_border_of_interest

st.set_page_config(page_title="Porosity & Grain Analysis", layout="wide")

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
        'grain_result': None,
        'erased_boxes': [],
        'last_uploaded_name': None,
        'needs_calc': False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def main():
    init_session_state()
    
    st.title("Porosity & Grain Analysis Tool")
    st.markdown("---")
    
    # ==========================
    # SIDEBAR CONTROLS
    # ==========================
    with st.sidebar:
        st.header("Controls")
        
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'bmp'])
        
        st.subheader("Basic Settings")
        fast_mask = st.checkbox("Fast Mask Processing", value=True)
        modify_size = st.checkbox("Modify Processing Size")
        processing_size = st.number_input("Processing Size (px)", min_value=128, max_value=2048, step=64, value=512) if modify_size else 512
        
        with st.expander("Show Advanced Options"):
            crop_legend = st.checkbox("Crop Legend")
            crop_height = st.number_input("Crop Height (px from bottom)", min_value=1, value=100) if crop_legend else None
            
            st.markdown("**Grain Size Parameters**")
            zoom_factor = st.number_input("Image Width (µm) for Scale", value=5000, step=10)
            grain_lines = st.slider("Grain Size Grid Lines (H & V)", min_value=1, max_value=20, value=5)
            
            st.markdown("**Morphology Parameters**")
            kernel_options = {0: 1/400, 1: 1/200, 2: 1/100, 3: 1/50, 4: 1/25, 5: 1/10}
            open_idx = st.slider("Open Kernel Ratio (0-5)", 0, 5, 1)
            close_idx = st.slider("Close Kernel Ratio (0-5)", 0, 5, 1)
            open_iters = st.slider("Open Iterations", 0, 5, 1)
            close_iters = st.slider("Close Iterations", 0, 5, 1)
            
            border_pixels = st.number_input("Border Width (px)", min_value=0, value=0)
            border_pixels = border_pixels if border_pixels > 0 else None
            
            border_ratios = {0: 0.05, 1: 0.10, 2: 0.15, 3: 0.20}
            border_ratio_idx = st.slider("Border Ratio Level (0-3)", 0, 3, 1)
            
            threshold = st.slider("Threshold (0 = Auto)", 0, 255, 0)
            mask_threshold = st.slider("Mask Threshold (0 = Auto)", 0, 255, 0)
            
        st.subheader("Analysis Actions")
        
        col1, col2 = st.columns(2)
        btn_recalc = col1.button("Recalculate", type="primary", use_container_width=True)
        btn_grain = col2.button("Calculate Grain Size", type="secondary", use_container_width=True)

    # ==========================
    # MAIN CONTENT AREA
    # ==========================
    if uploaded_file is not None:
        
        # --- Auto-Calculation Trigger on New Upload ---
        if st.session_state.last_uploaded_name != uploaded_file.name:
            image = Image.open(uploaded_file)
            st.session_state.original_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            st.session_state.last_uploaded_name = uploaded_file.name
            st.session_state.erased_boxes = [] # Reset erasures for new image
            st.session_state.needs_calc = True
            
        img_cv = st.session_state.original_img
        
        if btn_recalc:
            st.session_state.needs_calc = True

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
                
                # Apply persistent erasures over the freshly calculated mask
                if st.session_state.erased_boxes:
                    for box in st.session_state.erased_boxes:
                        x0, y0, w, h = box['x'], box['y'], box['w'], box['h']
                        mask[y0:y0+h, x0:x0+w] = 0
                    
                    # Recalculate full porosity with modified mask
                    inverse_binary = cv2.bitwise_not(binary)
                    combined = cv2.bitwise_and(inverse_binary, mask)
                    whole_area = np.count_nonzero(mask == 255)
                    pore_area = np.count_nonzero(combined == 255)
                    porosity = (pore_area * 100 / whole_area) if whole_area > 0 else 0
                    
                    # Recalculate border porosity with modified mask
                    b_mask = select_border_of_interest(mask, border_pixels, border_ratios[border_ratio_idx])
                    b_combined = cv2.bitwise_and(inverse_binary, b_mask)
                    b_area = np.count_nonzero(b_mask == 255)
                    b_pore_area = np.count_nonzero(b_combined == 255)
                    b_porosity = (b_pore_area * 100 / b_area) if b_area > 0 else 0
                
                # Store results in session state
                st.session_state.mask = mask
                st.session_state.pores_mask = combined
                st.session_state.border_mask = b_mask
                st.session_state.binary_img = binary
                st.session_state.full_porosity = porosity
                st.session_state.border_porosity = b_porosity
                st.session_state.grain_overlay = None # Reset grain view
                st.session_state.needs_calc = False
                
                st.rerun() # Refresh immediately to show results

        # Action: Calculate Grain Size
        if btn_grain:
            with st.spinner("Calculating Grain Size..."):
                intersections, mean_intercept_px, overlay = calculate_grain_size(
                    img_cv, 
                    st.session_state.mask, 
                    grain_lines, 
                    threshold
                )
                
                pixel_size_um = zoom_factor / img_cv.shape[1]
                mean_intercept_um = mean_intercept_px * pixel_size_um
                
                st.session_state.grain_overlay = overlay
                st.session_state.grain_result = f"Intersections: {intersections} | Mean Intercept: {mean_intercept_um:.2f} µm"

        # Display Results Metric Cards
        if st.session_state.full_porosity is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Full Image Porosity", f"{st.session_state.full_porosity:.2f}%")
            c2.metric("Border Porosity", f"{st.session_state.border_porosity:.2f}%")
            if st.session_state.grain_overlay is not None:
                c3.success(st.session_state.grain_result)

        # Tabs for different visualization layers
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Original / Draw", "Area Mask", "Pores", "Border", "Grain Grid"])
        
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        MAX_CANVAS_WIDTH = 1000  # Sets maximum width for both visual canvases
        
        with tab1:
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

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=2,
                stroke_color="red",
                background_image=display_img,
                update_streamlit=True,
                height=display_height,
                width=display_width,
                drawing_mode="rect",
                key="canvas_original",
            )
            
            if st.button("Calculate Selected Region Porosity"):
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

        with tab2:
            if st.session_state.mask is not None:
                st.info("Draw rectangles on the mask to erase regions. Click 'Erase Selection' to apply. Erased regions remain across calculations until reset.")
                
                red_mask = np.zeros((*st.session_state.mask.shape, 4), dtype=np.uint8)
                red_mask[st.session_state.mask == 255] = [255, 0, 0, 127]
                mask_bg = Image.alpha_composite(pil_img.convert("RGBA"), Image.fromarray(red_mask))
                
                mask_bg = mask_bg.convert("RGB")
                
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

                canvas_mask = st_canvas(
                    fill_color="rgba(255, 255, 0, 0.3)", 
                    stroke_width=2,
                    stroke_color="yellow", 
                    background_image=display_img_mask,
                    update_streamlit=True,
                    height=display_height_mask,
                    width=display_width_mask,
                    drawing_mode="rect",
                    key="canvas_mask",
                )
                
                col_c, col_d = st.columns(2)
                
                if col_c.button("Erase Selection"):
                    if canvas_mask.json_data is not None and len(canvas_mask.json_data["objects"]) > 0:
                        
                        for shape in canvas_mask.json_data["objects"]:
                            x0 = int(shape["left"] * scale_factor_mask)
                            y0 = int(shape["top"] * scale_factor_mask)
                            w = int(shape["width"] * scale_factor_mask)
                            h = int(shape["height"] * scale_factor_mask)
                            
                            st.session_state.erased_boxes.append({'x': x0, 'y': y0, 'w': w, 'h': h})
                            st.session_state.mask[y0:y0+h, x0:x0+w] = 0
                        
                        inverse_binary = cv2.bitwise_not(st.session_state.binary_img)
                        st.session_state.pores_mask = cv2.bitwise_and(inverse_binary, st.session_state.mask)
                        
                        whole_area = np.count_nonzero(st.session_state.mask == 255)
                        pore_area = np.count_nonzero(st.session_state.pores_mask == 255)
                        st.session_state.full_porosity = (pore_area * 100 / whole_area) if whole_area > 0 else 0
                        
                        st.rerun()
                    else:
                        st.warning("Please draw at least one rectangle first.")
                
                if col_d.button("Reset Erasures"):
                    st.session_state.erased_boxes = []
                    st.warning("Erasures cleared. Click 'Recalculate' in the sidebar to restore the default mask.")
            else:
                st.info("Upload an image to process the mask.")

        with tab3:
            if st.session_state.pores_mask is not None:
                blue_mask = np.zeros((*st.session_state.pores_mask.shape, 4), dtype=np.uint8)
                blue_mask[st.session_state.pores_mask == 255] = [0, 150, 255, 200]
                st.image(Image.alpha_composite(pil_img.convert("RGBA"), Image.fromarray(blue_mask)), use_column_width=True)

        with tab4:
            if st.session_state.border_mask is not None:
                yellow_mask = np.zeros((*st.session_state.border_mask.shape, 4), dtype=np.uint8)
                yellow_mask[st.session_state.border_mask == 255] = [255, 255, 0, 160]
                st.image(Image.alpha_composite(pil_img.convert("RGBA"), Image.fromarray(yellow_mask)), use_column_width=True)

        with tab5:
            if st.session_state.grain_overlay is not None:  
                st.image(Image.alpha_composite(pil_img.convert("RGBA"), Image.fromarray(st.session_state.grain_overlay)), use_column_width=True)
    else:
        st.info("Upload an image to begin analysis.")

if __name__ == "__main__":
    main()