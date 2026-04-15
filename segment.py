import cv2
import numpy as np


def crop_legend(img, img_path, png_legend_ratio=1/45, 
                bmp_legend_height=0.08, bmp_legend_height_value=200,
                bmp_legend_width=0.4, bmp_legend_width_value=3000,
                crop_height=None):
    """
    Crop legend from microscopy images based on file type
    
    Parameters:
    - img: grayscale image
    - img_path: path to the image file
    - png_legend_ratio: ratio for cropping PNG legend (default 1/45)
    - bmp_legend_height: ratio for BMP legend height cropping (default 0.08)
    - bmp_legend_height_value: absolute value for BMP legend height (default 200)
    - bmp_legend_width: ratio for BMP legend width cropping (default 0.4)
    - bmp_legend_width_value: absolute value for BMP legend width (default 3000)
    - crop_height: specific number of pixels to black out from bottom (default None)
    """
    # If crop_height is specified, black out the bottom portion
    if crop_height is not None:
        img_copy = img.copy()
        height = img_copy.shape[0]
        # Black out pixels from (height - crop_height) to the bottom
        start_row = max(0, height - crop_height)
        img_copy[start_row:, :] = 0
        return img_copy
    
    # Original logic for different file types
    if img_path.endswith('.jpg') or img_path.endswith('.png'):
        legend_crop_value = int(img.shape[0] * png_legend_ratio)
        cropped_img = img[legend_crop_value:, :]  # crop the legend
    elif img_path.endswith('.bmp'):
        height_crop = int(img.shape[0] * bmp_legend_height) if img.shape[0] < 5000 else bmp_legend_height_value
        width_crop = int(img.shape[1] * bmp_legend_width) if img.shape[1] < 2000 else bmp_legend_width_value
        img_copy = img.copy()
        img_copy[height_crop:, width_crop:] = 0  # Replace the bottom right corner with a black square
        cropped_img = img_copy
    else:
        cropped_img = img
    return cropped_img


def select_area_of_interest(processed_img, mask_threshold, open_kernel_ratio=1/200, close_kernel_ratio=1/200, open_iterations=1, close_iterations=1, processing_size=512):
    """
    Select area of interest using morphological operations with image resizing for speed optimization
    
    Parameters:
    - processed_img: processed grayscale image
    - mask_threshold: threshold value for creating area of interest mask (0 for OTSU)
    - open_kernel_ratio: ratio for morphological opening kernel size
    - close_kernel_ratio: ratio for morphological closing kernel size
    - open_iterations: number of iterations for opening operations (default 1, 0 to skip)
    - close_iterations: number of iterations for closing operations (default 1, 0 to skip)
    - processing_size: target size for processing (default 1024 pixels for largest dimension)
    
    Returns:
    - mask: binary mask of the area of interest (original size)
    """
    
    # Save original image size
    original_height, original_width = processed_img.shape[:2]
    
    # Calculate resize factor to make largest dimension equal to processing_size
    max_dimension = max(original_height, original_width)
    if max_dimension > processing_size:
        resize_factor = processing_size / max_dimension
        new_width = int(original_width * resize_factor)
        new_height = int(original_height * resize_factor)
        
        # Resize image for processing
        resized_img = cv2.resize(processed_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        # Image is already small enough, use original
        resized_img = processed_img
        resize_factor = 1.0
        new_width, new_height = original_width, original_height
    
    # Apply thresholding on resized image
    if mask_threshold > 0:
        _, binary_img = cv2.threshold(resized_img, mask_threshold, 255, cv2.THRESH_BINARY)
    else:
        _, binary_img = cv2.threshold(resized_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    mask = binary_img.copy()
    
    # Apply opening operation if iterations > 0
    if open_iterations > 0:
        # Scale kernel size according to resized image
        open_kernel_size = max(1, int(new_height * open_kernel_ratio))
        open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=open_iterations)
    
    # Apply closing operation if iterations > 0
    if close_iterations > 0:
        # Scale kernel size according to resized image
        close_kernel_size = max(1, int(new_height * close_kernel_ratio))
        close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)
    
    # Resize mask back to original image size
    if resize_factor != 1.0:
        mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        # Ensure binary values after resize
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    return mask


def select_border_of_interest(mask, border_pixels=None, border_ratio=0.10):
    """
    Select border area of interest using erosion and subtraction
    
    Parameters:
    - mask: area of interest mask
    - border_pixels: specific number of pixels for border width (default None)
    - border_ratio: ratio of border width to mask area (default 0.10 = 10%)
    
    Returns:
    - border_mask: binary mask of the border area
    """
    if border_pixels is not None:
        # Use specific pixel count for erosion kernel
        erosion_size = border_pixels
    else:
        # Calculate erosion size based on ratio
        # Use square root of area to get approximate radius, then apply ratio
        mask_area = np.count_nonzero(mask == 255)
        if mask_area == 0:
            return np.zeros_like(mask)
        
        # Estimate radius from area and apply border ratio
        estimated_radius = np.sqrt(mask_area / np.pi)
        erosion_size = max(1, int(estimated_radius * border_ratio))
    
    # Create erosion kernel
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    
    # Erode the mask to get inner area
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    
    # Subtract eroded mask from original to get border
    border_mask = cv2.subtract(mask, eroded_mask)
    
    return border_mask


def calculate_border_porosity_from_images(original_img, mask, border_pixels=None, border_ratio=0.10, manual_threshold=0):
    """
    Calculate border porosity from original image and mask
    
    Parameters:
    - original_img: original grayscale image
    - mask: area of interest mask
    - border_pixels: specific number of pixels for border width (default None)
    - border_ratio: ratio of border width to mask area (default 0.10 = 10%)
    - manual_threshold: manual threshold value, 0 for OTSU (default 0)
    
    Returns:
    - border_porosity: border porosity percentage
    - border_combined_mask: the combined mask showing pores in border area
    - border_mask: the border area mask
    """
    # Get border mask from the interest mask
    border_mask = select_border_of_interest(mask, border_pixels, border_ratio)
    
    # Calculate the border area counts
    border_area_counts = np.count_nonzero(border_mask == 255)
    
    if border_area_counts == 0:
        return 0, np.zeros_like(original_img), border_mask
    
    # Apply thresholding to get binary image
    if manual_threshold > 0:
        _, binary_img = cv2.threshold(original_img, manual_threshold, 255, cv2.THRESH_BINARY)
    else:
        _, binary_img = cv2.threshold(original_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Create an inverse of the binary image
    inverse_binary = cv2.bitwise_not(binary_img)
    
    # Combine the inverse binary image with the border mask
    border_combined_mask = cv2.bitwise_and(inverse_binary, border_mask)
    
    # Calculate the pore area counts in border
    border_pore_area_counts = np.count_nonzero(border_combined_mask == 255)
    
    # Calculate border porosity
    border_porosity = border_pore_area_counts * 100 / border_area_counts
    
    return border_porosity, border_combined_mask, border_mask


def calculate_porosity_from_images(original_img, mask, manual_threshold=0):
    """
    Calculate porosity from original image and mask with custom threshold
    
    Parameters:
    - original_img: original grayscale image
    - mask: area of interest mask
    - manual_threshold: manual threshold value, 0 for OTSU (default 0)
    
    Returns:
    - porosity: porosity percentage
    - combined_mask: the combined mask showing pores
    - binary_img: the thresholded binary image used for calculation
    """
    # Apply thresholding to get binary image
    if manual_threshold > 0:
        _, binary_img = cv2.threshold(original_img, manual_threshold, 255, cv2.THRESH_BINARY)
    else:
        _, binary_img = cv2.threshold(original_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Calculate the whole area counts
    whole_area_counts = np.count_nonzero(mask == 255)
    
    # Create an inverse of the binary image
    inverse_binary = cv2.bitwise_not(binary_img)
    
    # Combine the inverse binary image with the mask
    combined_mask = cv2.bitwise_and(inverse_binary, mask)
    
    # Calculate the pore area counts
    pore_area_counts = np.count_nonzero(combined_mask == 255)
    
    # Calculate porosity
    porosity = pore_area_counts * 100 / whole_area_counts if whole_area_counts > 0 else 0
    
    return porosity, combined_mask, binary_img


def analyze_porosity(img_input, crop_legend_enabled=False, open_kernel_ratio=1/200, close_kernel_ratio=1/200,
                    manual_threshold=0, mask_threshold=0, use_area_of_interest=True, open_iterations=1, close_iterations=1, 
                    border_pixels=None, border_ratio=0.10, fast_mask_enabled=True, processing_size=512,
                    png_legend_ratio=1/45, bmp_legend_height=0.08, bmp_legend_height_value=200,
                    bmp_legend_width=0.4, bmp_legend_width_value=3000, crop_height=None):
    """
    Analyze porosity from microscopy images
    
    Parameters:
    - img_input: path to the image file (string) or cv2 image array (numpy.ndarray)
    - crop_legend_enabled: whether to crop legend (default False)
    - open_kernel_ratio: ratio for morphological opening kernel size (default 1/200)
    - close_kernel_ratio: ratio for morphological closing kernel size (default 1/200)
    - manual_threshold: manual threshold value for pore detection, 0 for OTSU (default 0)
    - mask_threshold: threshold value for area of interest mask, 0 for OTSU (default 0)
    - use_area_of_interest: whether to use area of interest selection (default True)
    - open_iterations: number of iterations for opening operations (default 1, 0 to skip)
    - close_iterations: number of iterations for closing operations (default 1, 0 to skip)
    - border_pixels: specific number of pixels for border width (default None)
    - border_ratio: ratio of border width to mask area (default 0.10 = 10%)
    - fast_mask_enabled: whether to use fast mask processing with image resizing (default True)
    - processing_size: target size for fast mask processing (default 512)
    - png_legend_ratio: ratio for cropping PNG legend (default 1/45)
    - bmp_legend_height: ratio for BMP legend height cropping (default 0.08)
    - bmp_legend_height_value: absolute value for BMP legend height (default 200)
    - bmp_legend_width: ratio for BMP legend width cropping (default 0.4)
    - bmp_legend_width_value: absolute value for BMP legend width (default 3000)
    - crop_height: specific number of pixels to black out from bottom (default None)
    
    Returns:
    - porosity: porosity percentage
    - binary_img: binary thresholded image
    - mask: area of interest mask (or full image mask if use_area_of_interest=False)
    - combined_mask: pores mask
    - border_porosity: border porosity percentage
    - border_combined_mask: pores in border area mask
    - border_mask: border area mask
    """
    
    # Handle input - either file path or cv2 image
    if isinstance(img_input, str):
        # Input is a file path
        og_img = cv2.imread(img_input)
        img_path = img_input  # Keep path for legend cropping logic
    else:
        # Input is a cv2 image array
        og_img = img_input
        img_path = "image.png"  # Default path for legend cropping logic
     
    # Convert to grayscale
    img = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY)
    
    # Crop the legend based on file type (optional)
    if crop_legend_enabled:
        processed_img = crop_legend(img, img_path, png_legend_ratio, 
                                   bmp_legend_height, bmp_legend_height_value,
                                   bmp_legend_width, bmp_legend_width_value,
                                   crop_height)
    else:
        processed_img = img
    
    # Select area of interest using morphological operations (optional)
    if use_area_of_interest:
        if fast_mask_enabled:
            mask = select_area_of_interest(processed_img, mask_threshold, open_kernel_ratio, close_kernel_ratio, open_iterations, close_iterations, processing_size)
        else:
            # Use original size processing (no resizing)
            mask = select_area_of_interest(processed_img, mask_threshold, open_kernel_ratio, close_kernel_ratio, open_iterations, close_iterations, processing_size=max(processed_img.shape))
    else:
        # Use the entire image as the mask
        mask = np.ones_like(processed_img) * 255
    
    # Calculate porosity and get combined mask
    porosity, combined_mask, binary_img = calculate_porosity_from_images(processed_img, mask, manual_threshold)
    
    # Calculate border porosity
    border_porosity, border_combined_mask, border_mask = calculate_border_porosity_from_images(
        processed_img, mask, border_pixels, border_ratio, manual_threshold)
    
    return porosity, binary_img, mask, combined_mask, border_porosity, border_combined_mask, border_mask