import cv2
import numpy as np


def crop_legend(img, img_path, png_legend_ratio=1/45, 
                bmp_legend_height=0.08, bmp_legend_height_value=200,
                bmp_legend_width=0.4, bmp_legend_width_value=3000,
                crop_height=None):
    """
    Crop legend from microscopy images based on file type
    """
    if crop_height is not None:
        img_copy = img.copy()
        height = img_copy.shape[0]
        start_row = max(0, height - crop_height)
        img_copy[start_row:, :] = 0
        return img_copy
    
    if img_path.endswith('.jpg') or img_path.endswith('.png'):
        legend_crop_value = int(img.shape[0] * png_legend_ratio)
        cropped_img = img[legend_crop_value:, :]
    elif img_path.endswith('.bmp'):
        height_crop = int(img.shape[0] * bmp_legend_height) if img.shape[0] < 5000 else bmp_legend_height_value
        width_crop = int(img.shape[1] * bmp_legend_width) if img.shape[1] < 2000 else bmp_legend_width_value
        img_copy = img.copy()
        img_copy[height_crop:, width_crop:] = 0
        cropped_img = img_copy
    else:
        cropped_img = img
    return cropped_img


def select_area_of_interest(processed_img, mask_threshold, open_kernel_ratio=1/200, close_kernel_ratio=1/200, open_iterations=1, close_iterations=1, processing_size=512):
    """
    Select area of interest using morphological operations with image resizing for speed optimization
    """
    original_height, original_width = processed_img.shape[:2]
    max_dimension = max(original_height, original_width)
    
    if max_dimension > processing_size:
        resize_factor = processing_size / max_dimension
        new_width = int(original_width * resize_factor)
        new_height = int(original_height * resize_factor)
        resized_img = cv2.resize(processed_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized_img = processed_img
        resize_factor = 1.0
        new_width, new_height = original_width, original_height
    
    if mask_threshold > 0:
        _, binary_img = cv2.threshold(resized_img, mask_threshold, 255, cv2.THRESH_BINARY)
    else:
        _, binary_img = cv2.threshold(resized_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    mask = binary_img.copy()
    
    if open_iterations > 0:
        open_kernel_size = max(1, int(new_height * open_kernel_ratio))
        open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=open_iterations)
    
    if close_iterations > 0:
        close_kernel_size = max(1, int(new_height * close_kernel_ratio))
        close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)
    
    if resize_factor != 1.0:
        mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    return mask


def select_border_of_interest(mask, border_pixels=None, border_ratio=0.10):
    """
    Select border area of interest using erosion and subtraction
    """
    if border_pixels is not None:
        erosion_size = border_pixels
    else:
        mask_area = np.count_nonzero(mask == 255)
        if mask_area == 0:
            return np.zeros_like(mask)
        estimated_radius = np.sqrt(mask_area / np.pi)
        erosion_size = max(1, int(estimated_radius * border_ratio))
    
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    border_mask = cv2.subtract(mask, eroded_mask)
    
    return border_mask


def calculate_border_porosity_from_images(original_img, mask, border_pixels=None, border_ratio=0.10, manual_threshold=0):
    border_mask = select_border_of_interest(mask, border_pixels, border_ratio)
    border_area_counts = np.count_nonzero(border_mask == 255)
    
    if border_area_counts == 0:
        return 0, np.zeros_like(original_img), border_mask
    
    if manual_threshold > 0:
        _, binary_img = cv2.threshold(original_img, manual_threshold, 255, cv2.THRESH_BINARY)
    else:
        _, binary_img = cv2.threshold(original_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    inverse_binary = cv2.bitwise_not(binary_img)
    border_combined_mask = cv2.bitwise_and(inverse_binary, border_mask)
    border_pore_area_counts = np.count_nonzero(border_combined_mask == 255)
    border_porosity = border_pore_area_counts * 100 / border_area_counts
    
    return border_porosity, border_combined_mask, border_mask


def calculate_porosity_from_images(original_img, mask, manual_threshold=0):
    if manual_threshold > 0:
        _, binary_img = cv2.threshold(original_img, manual_threshold, 255, cv2.THRESH_BINARY)
    else:
        _, binary_img = cv2.threshold(original_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    whole_area_counts = np.count_nonzero(mask == 255)
    inverse_binary = cv2.bitwise_not(binary_img)
    combined_mask = cv2.bitwise_and(inverse_binary, mask)
    pore_area_counts = np.count_nonzero(combined_mask == 255)
    porosity = pore_area_counts * 100 / whole_area_counts if whole_area_counts > 0 else 0
    
    return porosity, combined_mask, binary_img


def analyze_porosity(img_input, crop_legend_enabled=False, open_kernel_ratio=1/200, close_kernel_ratio=1/200,
                    manual_threshold=0, mask_threshold=0, use_area_of_interest=True, open_iterations=1, close_iterations=1, 
                    border_pixels=None, border_ratio=0.10, fast_mask_enabled=True, processing_size=512,
                    png_legend_ratio=1/45, bmp_legend_height=0.08, bmp_legend_height_value=200,
                    bmp_legend_width=0.4, bmp_legend_width_value=3000, crop_height=None):
    
    if isinstance(img_input, str):
        og_img = cv2.imread(img_input)
        img_path = img_input
    else:
        og_img = img_input
        img_path = "image.png"
     
    img = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY)
    
    if crop_legend_enabled:
        processed_img = crop_legend(img, img_path, png_legend_ratio, 
                                   bmp_legend_height, bmp_legend_height_value,
                                   bmp_legend_width, bmp_legend_width_value,
                                   crop_height)
    else:
        processed_img = img
    
    if use_area_of_interest:
        if fast_mask_enabled:
            mask = select_area_of_interest(processed_img, mask_threshold, open_kernel_ratio, close_kernel_ratio, open_iterations, close_iterations, processing_size)
        else:
            mask = select_area_of_interest(processed_img, mask_threshold, open_kernel_ratio, close_kernel_ratio, open_iterations, close_iterations, processing_size=max(processed_img.shape))
    else:
        mask = np.ones_like(processed_img) * 255
    
    porosity, combined_mask, binary_img = calculate_porosity_from_images(processed_img, mask, manual_threshold)
    
    border_porosity, border_combined_mask, border_mask = calculate_border_porosity_from_images(
        processed_img, mask, border_pixels, border_ratio, manual_threshold)
    
    return porosity, binary_img, mask, combined_mask, border_porosity, border_combined_mask, border_mask


def calculate_grain_size(img_input, mask=None, num_lines=5, manual_threshold=0):
    """
    Calculate grain size using the statistical Line Intercept Method (Linienschnittverfahren).
    Applies test grids ONLY within the active Area of Interest mask.
    """
    if isinstance(img_input, str):
        og_img = cv2.imread(img_input)
    else:
        og_img = img_input
        
    if len(og_img.shape) == 3:
        img_gray = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = og_img.copy()
        
    height, width = img_gray.shape
    
    # If no mask is provided, treat the entire image as the area of interest
    if mask is None:
        mask = np.ones((height, width), dtype=np.uint8) * 255

    # Canny Edge Detection
    if manual_threshold > 0:
        _, binary = cv2.threshold(img_gray, manual_threshold, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(binary, 50, 150)
    else:
        blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
        v, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        edges = cv2.Canny(blurred, lower, upper)

    # Restrict edges strictly to the mask area
    edges = cv2.bitwise_and(edges, mask)

    # Create RGBA overlay (transparent background)
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    # Highlight grain boundaries inside the mask in bright green
    overlay[edges == 255] = [0, 255, 0, 180]

    h_spacing = height // (num_lines + 1)
    v_spacing = width // (num_lines + 1)

    total_intersections = 0
    total_length_pixels = 0

    # Process horizontal test lines
    for i in range(1, num_lines + 1):
        y = i * h_spacing
        line_mask = mask[y, :]
        line_edges = edges[y, :]
        
        # Calculate length of the test line that actually falls inside the mask
        length_in_mask = np.count_nonzero(line_mask)
        total_length_pixels += length_in_mask
        
        # Count discrete 0 -> 255 transitions to avoid boundary thickness duplication
        binary_line = (line_edges > 127).astype(np.int8)
        intersections = np.sum(np.diff(binary_line) > 0)
        total_intersections += intersections
        
        # Draw red guiding line ONLY where the mask is active
        active_indices = np.where(line_mask > 0)[0]
        if len(active_indices) > 0:
            overlay[y, active_indices] = [0, 0, 255, 180]

    # Process vertical test lines
    for i in range(1, num_lines + 1):
        x = i * v_spacing
        line_mask = mask[:, x]
        line_edges = edges[:, x]
        
        length_in_mask = np.count_nonzero(line_mask)
        total_length_pixels += length_in_mask
        
        binary_line = (line_edges > 127).astype(np.int8)
        intersections = np.sum(np.diff(binary_line) > 0)
        total_intersections += intersections
        
        active_indices = np.where(line_mask > 0)[0]
        if len(active_indices) > 0:
            overlay[active_indices, x] = [0, 0, 255, 180]

    mean_intercept = total_length_pixels / total_intersections if total_intersections > 0 else 0

    return total_intersections, mean_intercept, overlay