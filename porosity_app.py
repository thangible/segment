import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
from segment import analyze_porosity

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Global variables to store image data and processing results
uploaded_image = None
image_path = None
current_mask = None

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Porosity Analysis Tool", className="text-center mb-4"),
            html.Hr(),
        ])
    ]),
    
    # Main content - split screen layout
    dbc.Row([
        # Left column - Controls
        dbc.Col([
            html.H3("Controls", className="mb-3"),
            
            # File upload component
            html.Div([
                html.H5("Upload Image", className="mb-2"),
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Image Files'),
                        html.Br(),
                        'Supported formats: .jpg, .png, .bmp'
                    ]),
                    style={
                        'width': '100%',
                        'height': '80px',
                        'lineHeight': '80px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px 0'
                    },
                    multiple=False
                ),
                html.Div(id='upload-status', className="mb-3"),
            ], className="mb-4"),
            
            # Control buttons section
            html.Div([
                html.H5("Processing Options", className="mb-3"),
                
                # Checkbox for legend cropping
                dbc.Checklist(
                    options=[
                        {"label": " Crop Legend", "value": "crop_legend"}
                    ],
                    value=[],  # Default: not checked
                    id="crop-legend-checkbox",
                    className="mb-3"
                ),
                
                # Crop height input (only visible when crop legend is checked)
                html.Div([
                    html.Label("Crop Height (pixels from bottom)", className="form-label"),
                    dbc.Input(
                        id="crop-height-input",
                        type="number",
                        placeholder="e.g. 100",
                        min=1,
                        max=2000,
                        value=None,
                        className="mb-2"
                    ),
                    html.Small("Leave empty for automatic legend detection", className="form-text text-muted"),
                ], id="crop-height-container", className="mb-3", style={"display": "none"}),
                
                # Checkbox for fast mask processing
                dbc.Checklist(
                    options=[
                        {"label": " Fast Mask Processing", "value": "fast_mask"}
                    ],
                    value=["fast_mask"],  # Default: checked
                    id="fast-mask-checkbox",
                    className="mb-3"
                ),
                
                # Processing size input
                html.Div([
                    html.Label("Processing Size (pixels for largest dimension)", className="form-label"),
                    dbc.Input(
                        id="processing-size-input",
                        type="number",
                        placeholder="512",
                        min=128,
                        max=2048,
                        step=64,
                        value=512,
                        className="mb-2"
                    ),
                    html.Small("Smaller values = faster processing, larger values = better quality", className="form-text text-muted"),
                ], className="mb-3"),
                
                # Open kernel ratio slider
                html.Div([
                    html.Label("Open Kernel Ratio (to image size)", className="form-label"),
                    dcc.Slider(
                        id="open-kernel-ratio-slider",
                        min=0,
                        max=5,
                        step=1,
                        value=1,  # Default to 1/200
                        marks={
                            0: "1/400",
                            1: "1/200",
                            2: "1/100", 
                            3: "1/50",
                            4: "1/25",
                            5: "1/10"
                        },
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], className="mb-3"),
                
                # Close kernel ratio slider
                html.Div([
                    html.Label("Close Kernel Ratio (to image size)", className="form-label"),
                    dcc.Slider(
                        id="close-kernel-ratio-slider",
                        min=0,
                        max=5,
                        step=1,
                        value=1,  # Default to 1/200
                        marks={
                            0: "1/400",
                            1: "1/200",
                            2: "1/100", 
                            3: "1/50",
                            4: "1/25",
                            5: "1/10"
                        },
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], className="mb-3"),
                
                # Open iterations slider
                html.Div([
                    html.Label("Open Iterations (0 = Skip)", className="form-label"),
                    dcc.Slider(
                        id="open-iterations-slider",
                        min=0,
                        max=5,
                        step=1,
                        value=1,  # Default to 1 iteration
                        marks={
                            0: "0",
                            1: "1",
                            2: "2",
                            3: "3", 
                            4: "4",
                            5: "5"
                        },
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], className="mb-3"),
                
                # Close iterations slider
                html.Div([
                    html.Label("Close Iterations (0 = Skip)", className="form-label"),
                    dcc.Slider(
                        id="close-iterations-slider",
                        min=0,
                        max=5,
                        step=1,
                        value=1,  # Default to 1 iteration
                        marks={
                            0: "0",
                            1: "1",
                            2: "2",
                            3: "3", 
                            4: "4",
                            5: "5"
                        },
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], className="mb-3"),
                
                # Border pixels input
                html.Div([
                    html.Label("Border Width (pixels, leave empty for ratio)", className="form-label"),
                    dbc.Input(
                        id="border-pixels-input",
                        type="number",
                        placeholder="e.g. 20",
                        min=1,
                        value=None,
                        className="mb-2"
                    ),
                ], className="mb-3"),
                
                # Border ratio slider
                html.Div([
                    html.Label("Border Ratio (% of area radius)", className="form-label"),
                    dcc.Slider(
                        id="border-ratio-slider",
                        min=0,
                        max=3,
                        step=1,
                        value=1,  # Default to 10%
                        marks={
                            0: "5%",
                            1: "10%",
                            2: "15%",
                            3: "20%"
                        },
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], className="mb-3"),
                
                # Threshold slider
                html.Div([
                    html.Label("Pore Threshold (0 = Auto OTSU)", className="form-label"),
                    dcc.Slider(
                        id="threshold-slider",
                        min=0,
                        max=255,
                        step=1,
                        value=0,  # Default to OTSU
                        marks={
                            0: "Auto",
                            64: "64",
                            128: "128", 
                            192: "192",
                            255: "255"
                        },
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], className="mb-3"),
                
                # Mask Threshold slider
                html.Div([
                    html.Label("Mask Threshold - Area of Interest (0 = Auto OTSU)", className="form-label"),
                    dcc.Slider(
                        id="mask-threshold-slider",
                        min=0,
                        max=255,
                        step=1,
                        value=0,  # Default to OTSU
                        marks={
                            0: "Auto",
                            64: "64",
                            128: "128", 
                            192: "192",
                            255: "255"
                        },
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], className="mb-3"),
                
                html.H5("Analysis Options", className="mb-3"),
                dbc.Button("Calculate Full Image Porosity", 
                          id="btn-full-porosity", 
                          color="primary", 
                          className="w-100 mb-2",
                          size="md"),
                
                dbc.Button("Calculate Selected Region Porosity", 
                          id="btn-region-porosity", 
                          color="success", 
                          className="w-100 mb-2",
                          size="md"),
                
                # Calculating indicator
                html.Div(id="calculating-indicator", className="mb-3"),
            ], className="mb-4"),
            
            # # Instructions
            # html.Div([
            #     html.H5("Instructions", className="mb-2"),
            #     html.P("1. Upload an image using the upload area above"),
            #     html.P("2. Adjust processing options (legend cropping, kernel ratio, threshold)"),
            #     html.P("3. Click 'Calculate Full Image Porosity' to analyze with current settings"),
            #     html.P("4. Use red button (🔴) to toggle area of interest mask"),
            #     html.P("5. Use blue button (🔵) to toggle pores visualization"),
            #     html.P("6. Draw a rectangle on the image to select a region"),
            #     html.P("7. Click 'Calculate Selected Region Porosity' to analyze the selected area"),
            #     html.P("8. Use mouse wheel to zoom in/out on the image"),
            #     html.P("9. Drawing a new region will clear mask visualizations"),
            # ], className="alert alert-light"),
            
        ], width=4, style={"height": "100vh", "overflow-y": "auto", "padding-right": "20px"}),
        
        # Right column - Image and Results
        dbc.Col([
            html.H3("Image Analysis", className="mb-3"),
            
            # Toggle buttons above the graph
            html.Div([
                dbc.ButtonGroup([
                    dbc.Button(
                        "🔴 Mask", 
                        id="mask-toggle-btn",
                        color="secondary",
                        outline=True,
                        size="sm"
                    ),
                    dbc.Button(
                        "🔵 Pores", 
                        id="pores-toggle-btn",
                        color="secondary", 
                        outline=True,
                        size="sm"
                    ),
                    dbc.Button(
                        "🟡 Border", 
                        id="border-toggle-btn",
                        color="secondary", 
                        outline=True,
                        size="sm"
                    ),
                    dbc.Button(
                        "🗑️ Clear Areas", 
                        id="clear-rectangles-btn",
                        color="warning",
                        outline=True,
                        size="sm"
                    ),
                ], className="mb-2")
            ], className="d-flex justify-content-center"),
            
            # Image display with interactive selection
            dcc.Graph(
                id='image-graph',
                style={
                    'height': '70vh', 
                    'width': '100%',
                    'border': '1px solid #ddd'
                },
                config={
                    'modeBarButtonsToAdd': ['drawrect'],
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d'],
                    'displaylogo': False,
                    'scrollZoom': True,
                    'doubleClick': 'reset+autosize'
                }
            ),
            
            # Results display
            html.Div([
                html.H5("Results", className="mb-2"),
                dbc.Alert(id="full-image-result", color="info", style={"display": "none"}),
                dbc.Alert(id="region-result", color="success", style={"display": "none"}),
            ], className="mt-3"),
            
        ], width=8, style={"height": "100vh", "overflow-y": "auto", "padding-left": "20px"}),
    ], style={"margin-top": "20px"}),
    
    # Store components for data persistence
    dcc.Store(id='image-store'),
    dcc.Store(id='selection-store'),
    dcc.Store(id='mask-store'),  # Area of interest mask (base64 encoded)
    dcc.Store(id='pores-mask-store'),  # Pores mask (base64 encoded)
    dcc.Store(id='border-mask-store'),  # Border mask (base64 encoded)
    dcc.Store(id='binary-img-store'),  # Binary image for processing
    dcc.Store(id='mask-toggle-store', data=False),  # Store mask toggle state
    dcc.Store(id='pores-toggle-store', data=False),  # Store pores toggle state
    dcc.Store(id='border-toggle-store', data=False),  # Store border toggle state
    
    # Floating notification
    html.Div(
        id="notification",
        style={
            "position": "fixed",
            "bottom": "20px",
            "left": "20px",
            "zIndex": 9999,
            "display": "none"
        }
    ),
    
    # Interval for auto-hiding notifications
    dcc.Interval(id="notification-interval", interval=3000, n_intervals=0, disabled=True),
    
], fluid=True)

def encode_mask_to_base64(mask):
    """Convert numpy array mask to base64 string for storage"""
    if mask is None:
        return None
    # Encode mask as PNG to base64
    _, buffer = cv2.imencode('.png', mask)
    mask_b64 = base64.b64encode(buffer).decode()
    return mask_b64

def decode_mask_from_base64(mask_b64):
    """Convert base64 string back to numpy array mask"""
    if mask_b64 is None:
        return None
    # Decode base64 to numpy array
    mask_bytes = base64.b64decode(mask_b64)
    mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
    return mask

def parse_contents(contents):
    """Parse uploaded file contents"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Convert to PIL Image
    img = Image.open(io.BytesIO(decoded))
    
    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    return img, img_cv

def create_image_figure(img_pil, title="Uploaded Image"):
    """Create a plotly figure from PIL image"""
    fig = go.Figure()
    
    # Get image dimensions
    img_width, img_height = img_pil.width, img_pil.height
    
    # Add image
    fig.add_layout_image(
        dict(
            source=img_pil,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=img_width,
            sizey=img_height,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )
    
    # Configure layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            range=[0, img_width], 
            showgrid=True, 
            zeroline=False,
            scaleanchor="y",
            scaleratio=1,
            constrain="domain",
            rangemode="normal",
            fixedrange=False,
            constraintoward="center"
        ),
        yaxis=dict(
            range=[img_height, 0], 
            showgrid=True, 
            zeroline=False,
            constrain="domain",
            rangemode="normal",
            fixedrange=False,
            constraintoward="center"
        ),
        showlegend=False,
        dragmode="pan",  # Enable panning by default, users can switch to drawrect via modebar
        newshape=dict(
            fillcolor="rgba(255,0,0,0.3)",
            fillrule="evenodd",
            line=dict(color="red", width=2),
            opacity=0.7
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        autosize=True
    )
    
    return fig

def crop_image_from_selection(img_cv, selection_data):
    """Crop image based on selection coordinates"""
    if not selection_data or 'shapes' not in selection_data or len(selection_data['shapes']) == 0:
        return img_cv
    
    # Get the last drawn rectangle
    shape = selection_data['shapes'][-1]
    
    # Extract coordinates
    x0 = int(shape['x0'])
    y0 = int(shape['y0'])
    x1 = int(shape['x1'])
    y1 = int(shape['y1'])
    
    # Ensure coordinates are within image bounds
    height, width = img_cv.shape[:2]
    x0 = max(0, min(x0, width))
    x1 = max(0, min(x1, width))
    y0 = max(0, min(y0, height))
    y1 = max(0, min(y1, height))
    
    # Crop the image
    cropped = img_cv[y0:y1, x0:x1]
    
    return cropped

@app.callback(
    [Output('image-store', 'data'),
     Output('image-graph', 'figure'),
     Output('upload-status', 'children'),
     Output('notification', 'children', allow_duplicate=True),
     Output('notification', 'style', allow_duplicate=True),
     Output('notification-interval', 'disabled', allow_duplicate=True)],
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
    prevent_initial_call=True
)
def update_image(contents, filename):
    """Update image display when new file is uploaded"""
    if contents is None:
        return None, {}, "", "", {"position": "fixed", "bottom": "20px", "left": "20px", "zIndex": 9999, "display": "none"}, True
    
    try:
        # Parse the uploaded image
        img_pil, img_cv = parse_contents(contents)
        
        # Store image data (as base64 for persistence)
        img_buffer = io.BytesIO()
        img_pil.save(img_buffer, format='PNG')
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Create figure
        fig = create_image_figure(img_pil, f"Uploaded Image: {filename}")
        
        # Prepare results
        status = dbc.Alert(f"Successfully loaded {filename}", color="success")
        
        # Create success notification
        success_notification = dbc.Alert(
            [
                html.I(className="fas fa-check me-2"),  # Check icon
                f"Image loaded: {filename}"
            ], 
            color="success",
            className="d-flex align-items-center"
        )
        
        success_style = {
            "position": "fixed",
            "bottom": "20px", 
            "left": "20px",
            "zIndex": 9999,
            "display": "block",
            "minWidth": "150px"
        }
        
        return img_b64, fig, status, success_notification, success_style, False
        
    except Exception as e:
        error_msg = dbc.Alert(f"Error loading image: {str(e)}", color="danger")
        
        # Create error notification  
        error_notification = dbc.Alert(
            [
                html.I(className="fas fa-times me-2"),  # X icon
                "Error loading image!"
            ], 
            color="danger",
            className="d-flex align-items-center"
        )
        
        error_style = {
            "position": "fixed",
            "bottom": "20px", 
            "left": "20px",
            "zIndex": 9999,
            "display": "block",
            "minWidth": "150px"
        }
        
        return None, {}, error_msg, error_notification, error_style, False

@app.callback(
    Output('selection-store', 'data'),
    Input('image-graph', 'relayoutData'),
    State('selection-store', 'data')
)
def store_selection(relayoutData, current_selection):
    """Store selection data when user draws rectangle"""
    if relayoutData is None:
        return current_selection
    
    # Debug: print what we're getting
    # print("relayoutData keys:", list(relayoutData.keys()) if relayoutData else None)
    
    # Only update for actual shape drawing events
    # Ignore pan/zoom events which have axis range changes
    if relayoutData and 'shapes' in relayoutData:
        # Check if this is a pan/zoom event by looking for axis range changes
        has_axis_changes = any(key.startswith(('xaxis.', 'yaxis.')) for key in relayoutData.keys())
        
        # If there are axis changes, this is pan/zoom, so ignore
        if has_axis_changes:
            return current_selection
        
        # If shapes is present without axis changes, it's likely a drawing event
        return relayoutData
    
    # For all other cases, don't update
    return current_selection

@app.callback(
    [Output('mask-toggle-store', 'data', allow_duplicate=True),
     Output('pores-toggle-store', 'data', allow_duplicate=True)],
    Input('image-graph', 'relayoutData'),
    [State('mask-toggle-store', 'data'),
     State('pores-toggle-store', 'data')],
    prevent_initial_call=True
)
def clear_masks_on_new_rectangle(relayoutData, current_mask_state, current_pores_state):
    """Clear mask toggles only when user draws a new rectangle"""
    if relayoutData is None:
        return dash.no_update, dash.no_update
    
    # Only clear toggles if shapes were actually drawn (new rectangle created)
    if 'shapes' in relayoutData and len(relayoutData.get('shapes', [])) > 0:
        # Check if this is a new shape being drawn (not just a layout change)
        if any(key.startswith('shapes') for key in relayoutData.keys() if key != 'shapes'):
            return False, False
    
    return dash.no_update, dash.no_update

@app.callback(
    [Output('full-image-result', 'children'),
     Output('full-image-result', 'style'),
     Output('calculating-indicator', 'children', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True),
     Output('notification', 'style', allow_duplicate=True),
     Output('notification-interval', 'disabled', allow_duplicate=True),
     Output('mask-toggle-store', 'data', allow_duplicate=True),
     Output('pores-toggle-store', 'data', allow_duplicate=True),
     Output('border-toggle-store', 'data', allow_duplicate=True),
     Output('mask-store', 'data'),
     Output('pores-mask-store', 'data'),
     Output('border-mask-store', 'data')],
    Input('btn-full-porosity', 'n_clicks'),
    [State('image-store', 'data'),
     State('crop-legend-checkbox', 'value'),
     State('crop-height-input', 'value'),
     State('fast-mask-checkbox', 'value'),
     State('processing-size-input', 'value'),
     State('open-kernel-ratio-slider', 'value'),
     State('close-kernel-ratio-slider', 'value'),
     State('open-iterations-slider', 'value'),
     State('close-iterations-slider', 'value'),
     State('border-pixels-input', 'value'),
     State('border-ratio-slider', 'value'),
     State('threshold-slider', 'value'),
     State('mask-threshold-slider', 'value'),
     State('mask-toggle-store', 'data'),
     State('pores-toggle-store', 'data'),
     State('border-toggle-store', 'data')],
    prevent_initial_call=True
)
def calculate_full_porosity(btn_clicks, image_data, crop_legend_value, crop_height, fast_mask_value, processing_size, open_kernel_ratio_idx, close_kernel_ratio_idx, open_iterations, close_iterations, border_pixels, border_ratio_idx, threshold_value, mask_threshold_value, current_mask_state, current_pores_state, current_border_state):
    """Calculate and display full image porosity"""
    # Only calculate if button was clicked and we have image data
    if btn_clicks is None or image_data is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    crop_legend_enabled = 'crop_legend' in (crop_legend_value or [])
    fast_mask_enabled = 'fast_mask' in (fast_mask_value or [])
    
    # Use default processing size if not provided
    if processing_size is None or processing_size < 128:
        processing_size = 512
    
    # Convert kernel ratio indices to actual values
    kernel_ratios = [1/400, 1/200, 1/100, 1/50, 1/25, 1/10]
    open_kernel_ratio = kernel_ratios[open_kernel_ratio_idx] if 0 <= open_kernel_ratio_idx < len(kernel_ratios) else 1/200
    close_kernel_ratio = kernel_ratios[close_kernel_ratio_idx] if 0 <= close_kernel_ratio_idx < len(kernel_ratios) else 1/200
    
    # Convert border ratio index to actual value
    border_ratios = [0.05, 0.10, 0.15, 0.20]
    border_ratio = border_ratios[border_ratio_idx] if 0 <= border_ratio_idx < len(border_ratios) else 0.10
    
    try:
        # Decode stored image
        img_bytes = base64.b64decode(image_data)
        img_pil = Image.open(io.BytesIO(img_bytes))
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Analyze porosity for full image using cv2 image directly
        porosity, binary_img, mask, combined_mask, border_porosity, border_combined_mask, border_mask = analyze_porosity(
            img_cv, 
            crop_legend_enabled=crop_legend_enabled,
            crop_height=crop_height,
            open_kernel_ratio=open_kernel_ratio,
            close_kernel_ratio=close_kernel_ratio,
            manual_threshold=threshold_value,
            mask_threshold=mask_threshold_value,
            use_area_of_interest=True,  # Use area of interest for full image analysis
            open_iterations=open_iterations,
            close_iterations=close_iterations,
            border_pixels=border_pixels,
            border_ratio=border_ratio,
            fast_mask_enabled=fast_mask_enabled,
            processing_size=processing_size
        )
        
        # Store masks in base64 format for in-memory caching
        mask_b64 = encode_mask_to_base64(mask)  # Area of interest mask
        pores_mask_b64 = encode_mask_to_base64(combined_mask)  # Pores mask  
        border_mask_b64 = encode_mask_to_base64(border_mask)  # Border mask
        
        result = f"Full Image Porosity: {porosity:.2f}% | Border Porosity: {border_porosity:.2f}%"
        
        # Create success notification
        success_notification = dbc.Alert(
            [
                html.I(className="fas fa-check me-2"),  # Check icon
                "Success!"
            ], 
            color="success",
            className="d-flex align-items-center"
        )
        
        success_style = {
            "position": "fixed",
            "bottom": "20px", 
            "left": "20px",
            "zIndex": 9999,
            "display": "block",
            "minWidth": "150px"
        }
        
        # If any toggle is currently active, keep it active to refresh the visualization
        # with the newly calculated masks
        if current_mask_state:
            return result, {"display": "block"}, "", success_notification, success_style, False, True, False, False, mask_b64, pores_mask_b64, border_mask_b64
        elif current_pores_state:
            return result, {"display": "block"}, "", success_notification, success_style, False, False, True, False, mask_b64, pores_mask_b64, border_mask_b64
        elif current_border_state:
            return result, {"display": "block"}, "", success_notification, success_style, False, False, False, True, mask_b64, pores_mask_b64, border_mask_b64
        else:
            # No toggles are active, don't change anything
            return result, {"display": "block"}, "", success_notification, success_style, False, dash.no_update, dash.no_update, dash.no_update, mask_b64, pores_mask_b64, border_mask_b64
        
    except Exception as e:
        error_result = f"Error calculating full image porosity: {str(e)}"
        
        # Create error notification  
        error_notification = dbc.Alert(
            [
                html.I(className="fas fa-times me-2"),  # X icon
                "Error!"
            ], 
            color="danger",
            className="d-flex align-items-center"
        )
        
        error_style = {
            "position": "fixed",
            "bottom": "20px", 
            "left": "20px",
            "zIndex": 9999,
            "display": "block",
            "minWidth": "150px"
        }
        
        return error_result, {"display": "block"}, "", error_notification, error_style, False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Add callback for calculating porosity on button click (manual)
@app.callback(
    [Output('region-result', 'children'),
     Output('region-result', 'style'),
     Output('pores-toggle-store', 'data', allow_duplicate=True),
     Output('image-graph', 'figure', allow_duplicate=True)],
    Input('btn-region-porosity', 'n_clicks'),
    [State('selection-store', 'data'),
     State('image-store', 'data'),
     State('crop-legend-checkbox', 'value'),
     State('crop-height-input', 'value'),
     State('fast-mask-checkbox', 'value'),
     State('processing-size-input', 'value'),
     State('open-kernel-ratio-slider', 'value'),
     State('close-kernel-ratio-slider', 'value'),
     State('open-iterations-slider', 'value'),
     State('close-iterations-slider', 'value'),
     State('threshold-slider', 'value'),
     State('mask-threshold-slider', 'value'),
     State('pores-toggle-store', 'data'),
     State('image-graph', 'figure'),
     State('image-graph', 'relayoutData')],
    prevent_initial_call=True
)
def calculate_region_porosity(btn_clicks, selection_data, image_data, crop_legend_value, crop_height, fast_mask_value, processing_size, open_kernel_ratio_idx, close_kernel_ratio_idx, open_iterations, close_iterations, threshold_value, mask_threshold_value, current_pores_state, current_figure, relayout_data):
    """Calculate porosity when button is clicked for selected region"""
    if btn_clicks is None or not selection_data or not image_data or 'shapes' not in selection_data:
        return "", {"display": "none"}, dash.no_update, dash.no_update
    
    crop_legend_enabled = 'crop_legend' in (crop_legend_value or [])
    fast_mask_enabled = 'fast_mask' in (fast_mask_value or [])
    
    # Use default processing size if not provided
    if processing_size is None or processing_size < 128:
        processing_size = 512
    
    # Convert kernel ratio indices to actual values
    kernel_ratios = [1/400, 1/200, 1/100, 1/50, 1/25, 1/10]
    open_kernel_ratio = kernel_ratios[open_kernel_ratio_idx] if 0 <= open_kernel_ratio_idx < len(kernel_ratios) else 1/200
    close_kernel_ratio = kernel_ratios[close_kernel_ratio_idx] if 0 <= close_kernel_ratio_idx < len(kernel_ratios) else 1/200
    
    try:
        # Decode stored image
        img_bytes = base64.b64decode(image_data)
        img_pil = Image.open(io.BytesIO(img_bytes))
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Crop image based on selection
        cropped_img = crop_image_from_selection(img_cv, selection_data)
        
        if cropped_img.size > 0:
            # Analyze porosity directly on cropped image (no file I/O)
            porosity, binary_img, mask, combined_mask, border_porosity, border_combined_mask, border_mask = analyze_porosity(
                cropped_img, 
                crop_legend_enabled=False,  # Never crop legend for region analysis
                crop_height=None,  # Never use crop height for region analysis
                open_kernel_ratio=open_kernel_ratio,
                close_kernel_ratio=close_kernel_ratio,
                manual_threshold=threshold_value,  # Use manual threshold for pore analysis
                mask_threshold=0,  # Use default OTSU for region analysis
                use_area_of_interest=False,  # Skip area of interest for region analysis
                open_iterations=open_iterations,
                close_iterations=close_iterations,
                border_pixels=None,  # Default border parameters for region analysis
                border_ratio=0.10,
                fast_mask_enabled=fast_mask_enabled,  # Use user's fast mask setting
                processing_size=processing_size
            )
            
            # Create a full-size pores mask for the region by mapping back to original image coordinates
            if current_pores_state and selection_data and 'shapes' in selection_data:
                # Get the selection coordinates
                shape = selection_data['shapes'][-1]
                x0 = int(shape['x0'])
                y0 = int(shape['y0']) 
                x1 = int(shape['x1'])
                y1 = int(shape['y1'])
                
                # Ensure coordinates are within image bounds
                height, width = img_cv.shape[:2]
                x0 = max(0, min(x0, width))
                x1 = max(0, min(x1, width))
                y0 = max(0, min(y0, height))
                y1 = max(0, min(y1, height))
                
                # Create a full-size mask with zeros
                full_size_pores_mask = np.zeros((height, width), dtype=np.uint8)
                
                # Resize the region pores mask to match the selected area size
                region_height = y1 - y0
                region_width = x1 - x0
                if region_height > 0 and region_width > 0:
                    resized_pores_mask = cv2.resize(combined_mask, (region_width, region_height), interpolation=cv2.INTER_NEAREST)
                    # Place it in the correct position in the full-size mask
                    full_size_pores_mask[y0:y1, x0:x1] = resized_pores_mask
                
                # Create updated figure with pores overlay while preserving zoom/pan state
                fig = create_image_figure(img_pil, "Pores Mask (Bright Blue = Pores)")
                
                # Add blue pores mask overlay directly from in-memory mask
                blue_mask = np.zeros((full_size_pores_mask.shape[0], full_size_pores_mask.shape[1], 4), dtype=np.uint8)
                blue_mask[full_size_pores_mask == 255] = [0, 150, 255, 200]  # Brighter blue with higher opacity
                blue_mask_pil = Image.fromarray(blue_mask, 'RGBA')
                
                fig.add_layout_image(
                    dict(
                        source=blue_mask_pil,
                        xref="x",
                        yref="y",
                        x=0,
                        y=0,
                        sizex=img_pil.width,
                        sizey=img_pil.height,
                        sizing="stretch",
                        opacity=1.0,
                        layer="above"
                    )
                )
                
                # Preserve zoom/pan state from current figure and relayout data
                if current_figure and 'layout' in current_figure:
                    if 'xaxis' in current_figure['layout'] and 'range' in current_figure['layout']['xaxis']:
                        fig.update_layout(xaxis=dict(range=current_figure['layout']['xaxis']['range']))
                    if 'yaxis' in current_figure['layout'] and 'range' in current_figure['layout']['yaxis']:
                        fig.update_layout(yaxis=dict(range=current_figure['layout']['yaxis']['range']))
                
                # Also check relayout_data for more recent zoom/pan changes
                if relayout_data:
                    if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
                        fig.update_layout(xaxis=dict(range=[relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]))
                    if 'yaxis.range[0]' in relayout_data and 'yaxis.range[1]' in relayout_data:
                        fig.update_layout(yaxis=dict(range=[relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]))
                    elif 'xaxis.range' in relayout_data:
                        fig.update_layout(xaxis=dict(range=relayout_data['xaxis.range']))
                    if 'yaxis.range' in relayout_data:
                        fig.update_layout(yaxis=dict(range=relayout_data['yaxis.range']))
                
                # Preserve any drawn rectangles from selection_data
                if selection_data and 'shapes' in selection_data:
                    fig.update_layout(shapes=selection_data['shapes'])
                
                # Keep pores toggle active to refresh visualization
                pores_toggle_state = True
                updated_figure = fig
            else:
                pores_toggle_state = current_pores_state
                updated_figure = dash.no_update
            
            result = f"Selected Region Porosity: {porosity:.2f}%"
            
            return result, {"display": "block"}, pores_toggle_state, updated_figure
        else:
            return "Invalid selection area", {"display": "block"}, dash.no_update, dash.no_update
            
    except Exception as e:
        error_result = f"Error calculating region porosity: {str(e)}"
        return error_result, {"display": "block"}, dash.no_update, dash.no_update

@app.callback(
    Output('image-graph', 'figure', allow_duplicate=True),
    [Input('mask-toggle-store', 'data'),
     Input('pores-toggle-store', 'data'),
     Input('border-toggle-store', 'data'),
     Input('image-store', 'data')],
    [State('selection-store', 'data'),
     State('image-graph', 'relayoutData'),
     State('mask-store', 'data'),
     State('pores-mask-store', 'data'),
     State('border-mask-store', 'data')],
    prevent_initial_call=True
)
def toggle_mask_view(show_mask, show_pores_enabled, show_border_enabled, image_data, selection_data, relayout_data, mask_b64, pores_mask_b64, border_mask_b64):
    """Fast toggle between image layers using cached masks while preserving drawn rectangles"""
    if image_data is None:
        return dash.no_update
    
    try:
        # Decode stored image
        img_bytes = base64.b64decode(image_data)
        img_pil = Image.open(io.BytesIO(img_bytes))
        
        title = "Uploaded Image"
        
        # Create figure with original image as base layer
        fig = create_image_figure(img_pil, title)
        
        # Add red area of interest mask if requested and mask exists
        if show_mask and not show_pores_enabled and not show_border_enabled:
            if mask_b64:
                try:
                    # Decode mask from base64
                    mask = decode_mask_from_base64(mask_b64)
                    if mask is not None:
                        red_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                        red_mask[mask == 255] = [255, 0, 0, 127]  # Red with 50% opacity
                        red_mask_pil = Image.fromarray(red_mask, 'RGBA')
                        
                        fig.add_layout_image(
                            dict(
                                source=red_mask_pil,
                                xref="x",
                                yref="y", 
                                x=0,
                                y=0,
                                sizex=img_pil.width,
                                sizey=img_pil.height,
                                sizing="stretch",
                                opacity=1.0,
                                layer="above"
                            )
                        )
                        title = "Area of Interest Mask (Red = Analysis Area)"
                except Exception:
                    pass  # Ignore if mask decode fails
        
        # Add blue pores mask if requested and mask exists
        elif show_pores_enabled and not show_mask and not show_border_enabled:
            if pores_mask_b64:
                try:
                    # Decode pores mask from base64
                    pores_mask = decode_mask_from_base64(pores_mask_b64)
                    if pores_mask is not None:
                        blue_mask = np.zeros((pores_mask.shape[0], pores_mask.shape[1], 4), dtype=np.uint8)
                        blue_mask[pores_mask == 255] = [0, 150, 255, 200]  # Brighter blue with higher opacity
                        blue_mask_pil = Image.fromarray(blue_mask, 'RGBA')
                        
                        fig.add_layout_image(
                            dict(
                                source=blue_mask_pil,
                                xref="x",
                                yref="y",
                                x=0,
                                y=0,
                                sizex=img_pil.width,
                                sizey=img_pil.height,
                                sizing="stretch",
                                opacity=1.0,
                                layer="above"
                            )
                        )
                        title = "Pores Mask (Bright Blue = Pores)"
                except Exception:
                    pass  # Ignore if pores mask decode fails
        
        # Add yellow border mask if requested and mask exists
        elif show_border_enabled and not show_mask and not show_pores_enabled:
            if border_mask_b64:
                try:
                    # Decode border mask from base64
                    border_mask = decode_mask_from_base64(border_mask_b64)
                    if border_mask is not None:
                        yellow_mask = np.zeros((border_mask.shape[0], border_mask.shape[1], 4), dtype=np.uint8)
                        yellow_mask[border_mask == 255] = [255, 255, 0, 160]  # Yellow with opacity
                        yellow_mask_pil = Image.fromarray(yellow_mask, 'RGBA')
                        
                        fig.add_layout_image(
                            dict(
                                source=yellow_mask_pil,
                                xref="x",
                                yref="y",
                                x=0,
                                y=0,
                                sizex=img_pil.width,
                                sizey=img_pil.height,
                                sizing="stretch",
                                opacity=1.0,
                                layer="above"
                            )
                        )
                        title = "Border Mask (Yellow = Border Ring)"
                except Exception:
                    pass  # Ignore if border mask decode fails
        
        # Configure layout 
        fig.update_layout(
            title=title,
            xaxis=dict(
                range=[0, img_pil.width], 
                showgrid=True, 
                zeroline=False,
                scaleanchor="y",
                scaleratio=1,
                constrain="domain",
                rangemode="normal",
                fixedrange=False,
                constraintoward="center"
            ),
            yaxis=dict(
                range=[img_pil.height, 0], 
                showgrid=True, 
                zeroline=False,
                constrain="domain",
                rangemode="normal",
                fixedrange=False,
                constraintoward="center"
            ),
            showlegend=False,
            dragmode="pan",
            newshape=dict(
                fillcolor="rgba(255,0,0,0.3)",
                fillrule="evenodd",
                line=dict(color="red", width=2),
                opacity=0.7
            ),
            margin=dict(l=40, r=40, t=40, b=40),
            autosize=True
        )
        
        # Preserve zoom/pan state from relayoutData if it exists
        if relayout_data:
            # Check for zoom/pan data and apply it
            if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
                fig.update_layout(
                    xaxis=dict(range=[relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']])
                )
            if 'yaxis.range[0]' in relayout_data and 'yaxis.range[1]' in relayout_data:
                fig.update_layout(
                    yaxis=dict(range=[relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']])
                )
            # Also check for range updates in different format
            elif 'xaxis.range' in relayout_data:
                fig.update_layout(xaxis=dict(range=relayout_data['xaxis.range']))
            if 'yaxis.range' in relayout_data:
                fig.update_layout(yaxis=dict(range=relayout_data['yaxis.range']))
        
        # Preserve any drawn rectangles from selection_data
        if selection_data and 'shapes' in selection_data:
            fig.update_layout(shapes=selection_data['shapes'])
        
        return fig
        
    except Exception as e:
        return dash.no_update

# Callback to show updating notification when calculation starts
@app.callback(
    [Output('notification', 'children'),
     Output('notification', 'style')],
    Input('btn-full-porosity', 'n_clicks'),
    State('image-store', 'data'),
    prevent_initial_call=True
)
def show_updating_notification(btn_clicks, image_data):
    """Show updating notification when calculation button is clicked"""
    if btn_clicks is None or image_data is None:
        return "", {"position": "fixed", "bottom": "20px", "left": "20px", "zIndex": 9999, "display": "none"}
    
    updating_notification = dbc.Alert(
        [
            html.I(className="fas fa-spinner fa-spin me-2"),  # Spinner icon
            "Updating..."
        ], 
        color="warning",
        className="d-flex align-items-center"
    )
    
    style = {
        "position": "fixed",
        "bottom": "20px", 
        "left": "20px",
        "zIndex": 9999,
        "display": "block",
        "minWidth": "150px"
    }
    
    return updating_notification, style

# Callback to auto-hide notifications
@app.callback(
    [Output('notification', 'style', allow_duplicate=True),
     Output('notification-interval', 'disabled', allow_duplicate=True)],
    Input('notification-interval', 'n_intervals'),
    prevent_initial_call=True
)
def hide_notification(n_intervals):
    """Automatically hide notification after interval"""
    hidden_style = {
        "position": "fixed",
        "bottom": "20px",
        "left": "20px", 
        "zIndex": 9999,
        "display": "none"
    }
    return hidden_style, True

# Callback to handle toggle button clicks
@app.callback(
    [Output('mask-toggle-store', 'data'),
     Output('pores-toggle-store', 'data'),
     Output('border-toggle-store', 'data')],
    [Input('mask-toggle-btn', 'n_clicks'),
     Input('pores-toggle-btn', 'n_clicks'),
     Input('border-toggle-btn', 'n_clicks')],
    [State('mask-toggle-store', 'data'),
     State('pores-toggle-store', 'data'),
     State('border-toggle-store', 'data')],
    prevent_initial_call=True
)
def handle_toggle_button_clicks(mask_clicks, pores_clicks, border_clicks, current_mask_state, current_pores_state, current_border_state):
    """Handle clicks on toggle buttons - only one can be active at a time (mutually exclusive)"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'mask-toggle-btn':
        # If mask is currently off, turn it on and turn others off
        if not current_mask_state:
            return True, False, False
        # If mask is currently on, turn it off (and leave others off)
        else:
            return False, False, False
    elif button_id == 'pores-toggle-btn':
        # If pores is currently off, turn it on and turn others off
        if not current_pores_state:
            return False, True, False
        # If pores is currently on, turn it off (and leave others off)
        else:
            return False, False, False
    elif button_id == 'border-toggle-btn':
        # If border is currently off, turn it on and turn others off
        if not current_border_state:
            return False, False, True
        # If border is currently on, turn it off (and leave others off)
        else:
            return False, False, False
    
    return dash.no_update, dash.no_update, dash.no_update

# Callback to update button appearance based on toggle state
@app.callback(
    [Output('mask-toggle-btn', 'color'),
     Output('mask-toggle-btn', 'outline'),
     Output('pores-toggle-btn', 'color'),
     Output('pores-toggle-btn', 'outline'),
     Output('border-toggle-btn', 'color'),
     Output('border-toggle-btn', 'outline')],
    [Input('mask-toggle-store', 'data'),
     Input('pores-toggle-store', 'data'),
     Input('border-toggle-store', 'data')],
    prevent_initial_call=True
)
def update_button_appearance(mask_active, pores_active, border_active):
    """Update button colors based on toggle states"""
    mask_color = "danger" if mask_active else "secondary"
    mask_outline = not mask_active
    pores_color = "primary" if pores_active else "secondary"  
    pores_outline = not pores_active
    border_color = "warning" if border_active else "secondary"
    border_outline = not border_active
    
    return mask_color, mask_outline, pores_color, pores_outline, border_color, border_outline

# Callback to clear rectangles
@app.callback(
    [Output('selection-store', 'data', allow_duplicate=True),
     Output('region-result', 'style', allow_duplicate=True),
     Output('image-graph', 'figure', allow_duplicate=True)],
    Input('clear-rectangles-btn', 'n_clicks'),
    [State('image-store', 'data'),
     State('mask-toggle-store', 'data'),
     State('pores-toggle-store', 'data'),
     State('border-toggle-store', 'data'),
     State('mask-store', 'data'),
     State('pores-mask-store', 'data'),
     State('border-mask-store', 'data')],
    prevent_initial_call=True
)
def clear_rectangles(clear_clicks, image_data, show_mask, show_pores, show_border, mask_b64, pores_mask_b64, border_mask_b64):
    """Clear all drawn rectangles and region results"""
    if clear_clicks is None or image_data is None:
        return dash.no_update, dash.no_update, dash.no_update
    
    try:
        # Decode stored image
        img_bytes = base64.b64decode(image_data)
        img_pil = Image.open(io.BytesIO(img_bytes))
        
        # Create fresh figure without rectangles but preserve current mask overlays
        fig = create_image_figure(img_pil, "Uploaded Image")
        
        # Re-add mask overlays if they were active
        title = "Uploaded Image"
        
        # Add red area of interest mask if active
        if show_mask and mask_b64:
            try:
                # Decode mask from base64
                mask = decode_mask_from_base64(mask_b64)
                if mask is not None:
                    red_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                    red_mask[mask == 255] = [255, 0, 0, 127]  # Red with 50% opacity
                    red_mask_pil = Image.fromarray(red_mask, 'RGBA')
                    
                    fig.add_layout_image(
                        dict(
                            source=red_mask_pil,
                            xref="x",
                            yref="y", 
                            x=0,
                            y=0,
                            sizex=img_pil.width,
                            sizey=img_pil.height,
                            sizing="stretch",
                            opacity=1.0,
                            layer="above"
                        )
                    )
                    title = "Area of Interest Mask (Red = Analysis Area)"
            except Exception:
                pass
        
        # Add blue pores mask if active (mutually exclusive with other masks)
        elif show_pores and pores_mask_b64:
            try:
                # Decode pores mask from base64
                pores_mask = decode_mask_from_base64(pores_mask_b64)
                if pores_mask is not None:
                    blue_mask = np.zeros((pores_mask.shape[0], pores_mask.shape[1], 4), dtype=np.uint8)
                    blue_mask[pores_mask == 255] = [0, 150, 255, 200]  # Brighter blue with higher opacity
                    blue_mask_pil = Image.fromarray(blue_mask, 'RGBA')
                    
                    fig.add_layout_image(
                        dict(
                            source=blue_mask_pil,
                            xref="x",
                            yref="y",
                            x=0,
                            y=0,
                            sizex=img_pil.width,
                            sizey=img_pil.height,
                            sizing="stretch",
                            opacity=1.0,
                            layer="above"
                        )
                    )
                    title = "Pores Mask (Bright Blue = Pores)"
            except Exception:
                pass
        
        # Add yellow border mask if active (mutually exclusive with other masks)
        elif show_border and border_mask_b64:
            try:
                # Decode border mask from base64
                border_mask = decode_mask_from_base64(border_mask_b64)
                if border_mask is not None:
                    yellow_mask = np.zeros((border_mask.shape[0], border_mask.shape[1], 4), dtype=np.uint8)
                    yellow_mask[border_mask == 255] = [255, 255, 0, 160]  # Yellow with opacity
                    yellow_mask_pil = Image.fromarray(yellow_mask, 'RGBA')
                    
                    fig.add_layout_image(
                        dict(
                            source=yellow_mask_pil,
                            xref="x",
                            yref="y",
                            x=0,
                            y=0,
                            sizex=img_pil.width,
                            sizey=img_pil.height,
                            sizing="stretch",
                            opacity=1.0,
                            layer="above"
                        )
                    )
                    title = "Border Mask (Yellow = Border Ring)"
            except Exception:
                pass
        
        # Update figure title
        fig.update_layout(title=title)
        
        # Clear selection data and hide region results
        return None, {"display": "none"}, fig
        
    except Exception as e:
        return dash.no_update, dash.no_update, dash.no_update

# Callback to show/hide crop height input based on legend cropping checkbox
@app.callback(
    Output("crop-height-container", "style"),
    [Input("crop-legend-checkbox", "value")]
)
def toggle_crop_height_input(crop_legend_value):
    """Show/hide crop height input based on legend cropping checkbox"""
    if crop_legend_value and "crop_legend" in crop_legend_value:
        return {"display": "block"}
    else:
        return {"display": "none"}

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)
