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
from segment import analyze_porosity, select_border_of_interest, calculate_grain_size

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
            html.H1("Porosity & Grain Analysis Tool", className="text-center mb-4"),
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
                html.H5("Basic Settings", className="mb-3"),
                
                # Checkbox for fast mask processing
                dbc.Checklist(
                    options=[
                        {"label": " Fast Mask Processing", "value": "fast_mask"}
                    ],
                    value=["fast_mask"],  # Default: checked
                    id="fast-mask-checkbox",
                    className="mb-2"
                ),

                # Advanced toggle for processing size
                dbc.Checklist(
                    options=[{"label": " Modify Processing Size", "value": "show_size"}],
                    value=[],
                    id="show-size-checkbox",
                    className="mb-2 text-muted",
                    style={"fontSize": "0.9em"}
                ),
                
                # Processing size input (hidden by default)
                html.Div([
                    html.Label("Processing Size (pixels for largest dimension)", className="form-label", style={"fontSize": "0.9em"}),
                    dbc.Input(
                        id="processing-size-input",
                        type="number",
                        placeholder="512",
                        min=128,
                        max=2048,
                        step=64,
                        value=512,
                        className="mb-2",
                        size="sm"
                    ),
                    html.Small("Smaller values = faster processing, larger values = better quality", className="form-text text-muted"),
                ], id="processing-size-container", className="mb-3", style={"display": "none", "marginLeft": "25px"}),
                
                html.Hr(),

                # Advanced Options Toggle
                dbc.Checklist(
                    options=[{"label": " Show Advanced Options", "value": "show_advanced"}],
                    value=[],
                    id="advanced-options-checkbox",
                    className="mb-3 fw-bold"
                ),
                
                # Advanced Options Container (Hidden by Default)
                html.Div([
                    html.Div([
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
                        
                        html.Hr(),
                        html.H6("Grain Size Parameters", className="mb-2 text-muted"),
                        
                        # Zoom Factor for Grain Size
                        html.Div([
                            html.Label("Image Width (µm) for Scale", className="form-label"),
                            dbc.Input(
                                id="zoom-factor-input",
                                type="number",
                                value=5000,
                                step=10,
                                className="mb-2"
                            ),
                            html.Small("Physical width of the image to calculate real grain size", className="form-text text-muted"),
                        ], className="mb-3"),

                        # Grid Lines for Grain Size
                        html.Div([
                            html.Label("Grain Size Grid Lines (H & V)", className="form-label"),
                            dcc.Slider(
                                id="grain-lines-slider",
                                min=1,
                                max=20,
                                step=1,
                                value=5,  # Default 5 lines
                                marks={
                                    1: "1",
                                    5: "5",
                                    10: "10",
                                    15: "15",
                                    20: "20"
                                },
                                tooltip={"placement": "bottom", "always_visible": False}
                            ),
                        ], className="mb-3"),

                        html.Hr(),
                        html.H6("Morphology Parameters", className="mb-2 text-muted"),

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
                                tooltip={"placement": "bottom", "always_visible": False}
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
                                tooltip={"placement": "bottom", "always_visible": False}
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
                                tooltip={"placement": "bottom", "always_visible": False}
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
                                tooltip={"placement": "bottom", "always_visible": False}
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
                                tooltip={"placement": "bottom", "always_visible": False}
                            ),
                        ], className="mb-3"),
                        
                        # Threshold slider
                        html.Div([
                            html.Label("Threshold (0 = Auto OTSU)", className="form-label"),
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
                                tooltip={"placement": "bottom", "always_visible": False}
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
                                tooltip={"placement": "bottom", "always_visible": False}
                            ),
                        ], className="mb-3"),

                    ], className="p-3 border rounded bg-light mb-4")
                ], id="advanced-options-container", style={"display": "none"}),
                
                html.H5("Analysis Actions", className="mb-3 mt-2"),
                
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

                dbc.Button("Calculate Grain Size (Line Intercept)", 
                          id="btn-grain-size", 
                          color="info", 
                          className="w-100 mb-2",
                          size="md"),

                dbc.Button("Erase Selection from Mask", 
                          id="btn-erase-mask", 
                          color="danger", 
                          outline=True,
                          className="w-100 mb-2",
                          size="md"),
                
                # Calculating indicator
                html.Div(id="calculating-indicator", className="mb-3"),
            ], className="mb-4"),
            
        ], width=4, style={"height": "100vh", "overflow-y": "auto", "padding-right": "20px"}),
        
        # Right column - Image and Results
        dbc.Col([
            html.H3("Image Analysis", className="mb-3"),
            
            # Toggle buttons above the graph
            html.Div([
                dbc.ButtonGroup([
                    dbc.Button(
                        "👁 Mask", 
                        id="mask-toggle-btn",
                        color="secondary",
                        outline=True,
                        size="sm"
                    ),
                    dbc.Button(
                        "👁 Pores", 
                        id="pores-toggle-btn",
                        color="secondary", 
                        outline=True,
                        size="sm"
                    ),
                    dbc.Button(
                        "👁 Border", 
                        id="border-toggle-btn",
                        color="secondary", 
                        outline=True,
                        size="sm"
                    ),
                    dbc.Button(
                        "👁 Grain Grid", 
                        id="grain-toggle-btn",
                        color="secondary", 
                        outline=True,
                        size="sm"
                    ),
                    dbc.Button(
                        "🧹 Clear Areas", 
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
                dbc.Alert(id="full-image-result", color="primary", style={"display": "none"}),
                dbc.Alert(id="border-result", color="warning", style={"display": "none"}),
                dbc.Alert(id="region-result", color="success", style={"display": "none"}),
                dbc.Alert(id="grain-result", color="info", style={"display": "none"}),
            ], className="mt-3"),
            
        ], width=8, style={"height": "100vh", "overflow-y": "auto", "padding-left": "20px"}),
    ], style={"margin-top": "20px"}),
    
    # Store components for data persistence
    dcc.Store(id='image-store'),
    dcc.Store(id='selection-store'),
    dcc.Store(id='mask-store'),  
    dcc.Store(id='pores-mask-store'), 
    dcc.Store(id='border-mask-store'),
    dcc.Store(id='grain-mask-store'),  # Store for Grain visualization layer 
    dcc.Store(id='binary-img-store'),  
    dcc.Store(id='mask-toggle-store', data=False),  
    dcc.Store(id='pores-toggle-store', data=False),  
    dcc.Store(id='border-toggle-store', data=False),
    dcc.Store(id='grain-toggle-store', data=False),  
    
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
    _, buffer = cv2.imencode('.png', mask)
    mask_b64 = base64.b64encode(buffer).decode()
    return mask_b64

def decode_mask_from_base64(mask_b64, is_color=False):
    """Convert base64 string back to numpy array mask"""
    if mask_b64 is None:
        return None
    mask_bytes = base64.b64decode(mask_b64)
    mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
    if is_color:
        mask = cv2.imdecode(mask_array, cv2.IMREAD_UNCHANGED)
    else:
        mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
    return mask

def parse_contents(contents):
    """Parse uploaded file contents"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(decoded))
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img, img_cv

def create_image_figure(img_pil, title="Uploaded Image"):
    """Create a plotly figure from PIL image"""
    fig = go.Figure()
    img_width, img_height = img_pil.width, img_pil.height
    
    fig.add_layout_image(
        dict(
            source=img_pil, xref="x", yref="y", x=0, y=0,
            sizex=img_width, sizey=img_height, sizing="stretch",
            opacity=1, layer="below"
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis=dict(
            range=[0, img_width], showgrid=True, zeroline=False,
            scaleanchor="y", scaleratio=1, constrain="domain",
            rangemode="normal", fixedrange=False, constraintoward="center"
        ),
        yaxis=dict(
            range=[img_height, 0], showgrid=True, zeroline=False,
            constrain="domain", rangemode="normal", fixedrange=False,
            constraintoward="center"
        ),
        showlegend=False, dragmode="pan",
        newshape=dict(fillcolor="rgba(255,0,0,0.3)", fillrule="evenodd", line=dict(color="red", width=2), opacity=0.7),
        margin=dict(l=40, r=40, t=40, b=40), autosize=True
    )
    return fig

def crop_image_from_selection(img_cv, selection_data):
    """Crop image based on selection coordinates"""
    if not selection_data or 'shapes' not in selection_data or len(selection_data['shapes']) == 0:
        return img_cv
    shape = selection_data['shapes'][-1]
    x0, y0 = int(shape['x0']), int(shape['y0'])
    x1, y1 = int(shape['x1']), int(shape['y1'])
    
    height, width = img_cv.shape[:2]
    x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
    y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
    return img_cv[y0:y1, x0:x1]

@app.callback(
    [Output('image-store', 'data'),
     Output('image-graph', 'figure'),
     Output('upload-status', 'children')],
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
    prevent_initial_call=True
)
def update_image(contents, filename):
    """Update image display when new file is uploaded"""
    if contents is None:
        return None, {}, ""
    try:
        img_pil, img_cv = parse_contents(contents)
        img_buffer = io.BytesIO()
        img_pil.save(img_buffer, format='PNG')
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        fig = create_image_figure(img_pil, f"Uploaded Image: {filename}")
        status = dbc.Alert(f"Successfully loaded {filename}", color="success")
        return img_b64, fig, status
    except Exception as e:
        error_msg = dbc.Alert(f"Error loading image: {str(e)}", color="danger")
        return None, {}, error_msg

@app.callback(
    Output('selection-store', 'data'),
    Input('image-graph', 'relayoutData'),
    State('selection-store', 'data')
)
def store_selection(relayoutData, current_selection):
    """Store selection data when user draws rectangle"""
    if relayoutData is None:
        return current_selection
    if relayoutData and 'shapes' in relayoutData:
        has_axis_changes = any(key.startswith(('xaxis.', 'yaxis.')) for key in relayoutData.keys())
        if has_axis_changes:
            return current_selection
        return relayoutData
    return current_selection

@app.callback(
    [Output('full-image-result', 'children'),
     Output('full-image-result', 'style'),
     Output('border-result', 'children'),
     Output('border-result', 'style'),
     Output('calculating-indicator', 'children', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True),
     Output('notification', 'style', allow_duplicate=True),
     Output('notification-interval', 'disabled', allow_duplicate=True),
     Output('mask-toggle-store', 'data', allow_duplicate=True),
     Output('pores-toggle-store', 'data', allow_duplicate=True),
     Output('border-toggle-store', 'data', allow_duplicate=True),
     Output('grain-toggle-store', 'data', allow_duplicate=True),
     Output('mask-store', 'data'),
     Output('pores-mask-store', 'data'),
     Output('border-mask-store', 'data'),
     Output('binary-img-store', 'data')],
    [Input('btn-full-porosity', 'n_clicks'),
     Input('image-store', 'data')],
    [State('crop-legend-checkbox', 'value'),
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
    """Calculate and display full image porosity (triggers manually or instantly on image load)"""
    ctx = dash.callback_context
    if not ctx.triggered or image_data is None:
        return (dash.no_update,) * 16
    
    crop_legend_enabled = 'crop_legend' in (crop_legend_value or [])
    fast_mask_enabled = 'fast_mask' in (fast_mask_value or [])
    
    if processing_size is None or processing_size < 128:
        processing_size = 512
    
    kernel_ratios = [1/400, 1/200, 1/100, 1/50, 1/25, 1/10]
    open_kernel_ratio = kernel_ratios[open_kernel_ratio_idx] if 0 <= open_kernel_ratio_idx < len(kernel_ratios) else 1/200
    close_kernel_ratio = kernel_ratios[close_kernel_ratio_idx] if 0 <= close_kernel_ratio_idx < len(kernel_ratios) else 1/200
    
    border_ratios = [0.05, 0.10, 0.15, 0.20]
    border_ratio = border_ratios[border_ratio_idx] if 0 <= border_ratio_idx < len(border_ratios) else 0.10
    
    try:
        img_bytes = base64.b64decode(image_data)
        img_pil = Image.open(io.BytesIO(img_bytes))
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        porosity, binary_img, mask, combined_mask, border_porosity, border_combined_mask, border_mask = analyze_porosity(
            img_cv, 
            crop_legend_enabled=crop_legend_enabled,
            crop_height=crop_height,
            open_kernel_ratio=open_kernel_ratio,
            close_kernel_ratio=close_kernel_ratio,
            manual_threshold=threshold_value,
            mask_threshold=mask_threshold_value,
            use_area_of_interest=True,
            open_iterations=open_iterations,
            close_iterations=close_iterations,
            border_pixels=border_pixels,
            border_ratio=border_ratio,
            fast_mask_enabled=fast_mask_enabled,
            processing_size=processing_size
        )
        
        mask_b64 = encode_mask_to_base64(mask)
        pores_mask_b64 = encode_mask_to_base64(combined_mask)
        border_mask_b64 = encode_mask_to_base64(border_mask)
        binary_b64 = encode_mask_to_base64(binary_img)
        
        result_full = f"Full Image Porosity: {porosity:.2f}%"
        result_border = f"Border Porosity: {border_porosity:.2f}%"
        
        success_notification = dbc.Alert(
            [html.I(className="fas fa-check me-2"), "Calculations Complete!"], 
            color="success", className="d-flex align-items-center"
        )
        success_style = {"position": "fixed", "bottom": "20px", "left": "20px", "zIndex": 9999, "display": "block", "minWidth": "150px"}
        
        # Turn mask on by default if triggered by image load
        force_mask_on = (ctx.triggered[0]['prop_id'] == 'image-store.data')
        mask_state = True if force_mask_on else current_mask_state
        pores_state = False if force_mask_on else current_pores_state
        border_state = False if force_mask_on else current_border_state
        grain_state = False # Force grain state off on new run
        
        return result_full, {"display": "block"}, result_border, {"display": "block"}, "", success_notification, success_style, False, mask_state, pores_state, border_state, grain_state, mask_b64, pores_mask_b64, border_mask_b64, binary_b64
        
    except Exception as e:
        error_result = f"Error: {str(e)}"
        error_notification = dbc.Alert([html.I(className="fas fa-times me-2"), "Error!"], color="danger", className="d-flex align-items-center")
        error_style = {"position": "fixed", "bottom": "20px", "left": "20px", "zIndex": 9999, "display": "block", "minWidth": "150px"}
        return error_result, {"display": "block"}, dash.no_update, {"display": "none"}, "", error_notification, error_style, False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


@app.callback(
    [Output('grain-result', 'children'),
     Output('grain-result', 'style'),
     Output('grain-mask-store', 'data'),
     Output('mask-toggle-store', 'data', allow_duplicate=True),
     Output('pores-toggle-store', 'data', allow_duplicate=True),
     Output('border-toggle-store', 'data', allow_duplicate=True),
     Output('grain-toggle-store', 'data', allow_duplicate=True),
     Output('image-graph', 'figure', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True),
     Output('notification', 'style', allow_duplicate=True)],
    Input('btn-grain-size', 'n_clicks'),
    [State('image-store', 'data'),
     State('mask-store', 'data'),
     State('zoom-factor-input', 'value'),
     State('grain-lines-slider', 'value'),
     State('threshold-slider', 'value'),
     State('image-graph', 'figure'),
     State('image-graph', 'relayoutData')],
    prevent_initial_call=True
)
def calculate_grain_size_callback(n_clicks, image_data, mask_b64, zoom_factor, num_lines, threshold_val, current_figure, relayout_data):
    """Calculates Grain Size using Line Intercept Method, constrained by Area of Interest mask"""
    if n_clicks is None or not image_data: 
        return (dash.no_update,) * 10
        
    try:
        img_bytes = base64.b64decode(image_data)
        img_pil = Image.open(io.BytesIO(img_bytes))
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Pull red Area of Interest mask to restrict grain lines.
        # Fallback to a full white mask if it hasn't been run yet.
        mask = decode_mask_from_base64(mask_b64) if mask_b64 else None
        
        intersections, mean_intercept_px, overlay = calculate_grain_size(img_cv, mask, num_lines, threshold_val)
        
        # Calculate real-world scale (µm per pixel)
        width_pixels = img_cv.shape[1]
        zoom_factor = zoom_factor if zoom_factor is not None else 5000
        pixel_size_um = zoom_factor / width_pixels
        mean_intercept_um = mean_intercept_px * pixel_size_um
        
        result_text = html.Span([
            html.B(f"Grain Boundary Intersections: {intersections}"), html.Br(),
            f"Mean Intercept: {mean_intercept_px:.2f} px (",
            html.B(f"{mean_intercept_um:.2f} µm", style={"color": "#0dcaf0"}),
            ")"
        ])
        
        # Encode overlay to base64 for storage
        _, buffer = cv2.imencode('.png', overlay)
        overlay_b64 = base64.b64encode(buffer).decode()

        # Build figure directly to immediately display the layer
        fig = create_image_figure(img_pil, "Grain Size Visualization (Green=Boundaries, Red=Test Grid)")
        fig.add_layout_image(
            dict(
                source=Image.fromarray(overlay, 'RGBA'),
                xref="x", yref="y", x=0, y=0,
                sizex=img_pil.width, sizey=img_pil.height,
                sizing="stretch", opacity=1.0, layer="above"
            )
        )
        
        # Retain view state
        if current_figure and 'layout' in current_figure:
            if 'xaxis' in current_figure['layout'] and 'range' in current_figure['layout']['xaxis']:
                fig.update_layout(xaxis=dict(range=current_figure['layout']['xaxis']['range']))
            if 'yaxis' in current_figure['layout'] and 'range' in current_figure['layout']['yaxis']:
                fig.update_layout(yaxis=dict(range=current_figure['layout']['yaxis']['range']))
        if relayout_data:
            if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
                fig.update_layout(xaxis=dict(range=[relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]))
            if 'yaxis.range[0]' in relayout_data and 'yaxis.range[1]' in relayout_data:
                fig.update_layout(yaxis=dict(range=[relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]))

        success_note = dbc.Alert([html.I(className="fas fa-check me-2"), "Grain Size Calculated!"], color="success", className="d-flex align-items-center")
        style = {"position": "fixed", "bottom": "20px", "left": "20px", "zIndex": 9999, "display": "block"}

        # Activate grain toggle, deactivate others
        return result_text, {"display": "block"}, overlay_b64, False, False, False, True, fig, success_note, style
        
    except Exception as e:
        error = dbc.Alert([html.I(className="fas fa-times me-2"), f"Error: {str(e)}"], color="danger", className="d-flex align-items-center")
        style = {"position": "fixed", "bottom": "20px", "left": "20px", "zIndex": 9999, "display": "block"}
        return (dash.no_update,) * 8 + (error, style)


@app.callback(
    [Output('mask-store', 'data', allow_duplicate=True),
     Output('pores-mask-store', 'data', allow_duplicate=True),
     Output('border-mask-store', 'data', allow_duplicate=True),
     Output('full-image-result', 'children', allow_duplicate=True),
     Output('full-image-result', 'style', allow_duplicate=True),
     Output('border-result', 'children', allow_duplicate=True),
     Output('border-result', 'style', allow_duplicate=True),
     Output('image-graph', 'figure', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True),
     Output('notification', 'style', allow_duplicate=True)],
    Input('btn-erase-mask', 'n_clicks'),
    [State('selection-store', 'data'),
     State('mask-store', 'data'),
     State('binary-img-store', 'data'),
     State('border-pixels-input', 'value'),
     State('border-ratio-slider', 'value'),
     State('image-store', 'data'),
     State('mask-toggle-store', 'data'),
     State('pores-toggle-store', 'data'),
     State('border-toggle-store', 'data'),
     State('image-graph', 'relayoutData')],
    prevent_initial_call=True
)
def erase_from_mask(erase_clicks, selection_data, mask_b64, binary_b64, border_pixels, border_ratio_idx, image_data, show_mask, show_pores, show_border, relayout_data):
    """Erases the drawn rectangle from the Area of Interest mask and recalculates porosity."""
    if erase_clicks is None or not selection_data or 'shapes' not in selection_data or not mask_b64 or not binary_b64:
        error_notification = dbc.Alert(
            [html.I(className="fas fa-exclamation-triangle me-2"), "Draw a rectangle to erase!"],
            color="warning", className="d-flex align-items-center"
        )
        style = {"position": "fixed", "bottom": "20px", "left": "20px", "zIndex": 9999, "display": "block"}
        return (dash.no_update,) * 10
    
    try:
        mask = decode_mask_from_base64(mask_b64)
        binary_img = decode_mask_from_base64(binary_b64)
        
        shape = selection_data['shapes'][-1]
        x0, y0 = int(shape['x0']), int(shape['y0'])
        x1, y1 = int(shape['x1']), int(shape['y1'])
        
        height, width = mask.shape
        x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
        y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
        
        new_mask = mask.copy()
        new_mask[y0:y1, x0:x1] = 0
            
        inverse_binary = cv2.bitwise_not(binary_img)
        new_combined_mask = cv2.bitwise_and(inverse_binary, new_mask)
        
        border_ratios = [0.05, 0.10, 0.15, 0.20]
        border_ratio = border_ratios[border_ratio_idx] if border_ratio_idx is not None else 0.10
        new_border_mask = select_border_of_interest(new_mask, border_pixels, border_ratio)
        new_border_combined_mask = cv2.bitwise_and(inverse_binary, new_border_mask)
        
        whole_area = np.count_nonzero(new_mask == 255)
        pore_area = np.count_nonzero(new_combined_mask == 255)
        porosity = (pore_area * 100 / whole_area) if whole_area > 0 else 0
        
        border_area = np.count_nonzero(new_border_mask == 255)
        border_pore_area = np.count_nonzero(new_border_combined_mask == 255)
        border_porosity = (border_pore_area * 100 / border_area) if border_area > 0 else 0
        
        result_full = f"Full Image Porosity: {porosity:.2f}%"
        result_border = f"Border Porosity: {border_porosity:.2f}%"
        
        new_mask_b64 = encode_mask_to_base64(new_mask)
        new_pores_mask_b64 = encode_mask_to_base64(new_combined_mask)
        new_border_mask_b64 = encode_mask_to_base64(new_border_mask)

        img_bytes = base64.b64decode(image_data)
        img_pil = Image.open(io.BytesIO(img_bytes))
        fig = create_image_figure(img_pil, "Uploaded Image")
        
        if show_mask and new_mask_b64:
            red_mask = np.zeros((new_mask.shape[0], new_mask.shape[1], 4), dtype=np.uint8)
            red_mask[new_mask == 255] = [255, 0, 0, 127]
            fig.add_layout_image(dict(source=Image.fromarray(red_mask, 'RGBA'), xref="x", yref="y", x=0, y=0, sizex=img_pil.width, sizey=img_pil.height, sizing="stretch", opacity=1.0, layer="above"))
            fig.update_layout(title="Area of Interest Mask (Red = Analysis Area)")
            
        elif show_pores and new_pores_mask_b64:
            blue_mask = np.zeros((new_combined_mask.shape[0], new_combined_mask.shape[1], 4), dtype=np.uint8)
            blue_mask[new_combined_mask == 255] = [0, 150, 255, 200]
            fig.add_layout_image(dict(source=Image.fromarray(blue_mask, 'RGBA'), xref="x", yref="y", x=0, y=0, sizex=img_pil.width, sizey=img_pil.height, sizing="stretch", opacity=1.0, layer="above"))
            fig.update_layout(title="Pores Mask (Bright Blue = Pores)")
            
        elif show_border and new_border_mask_b64:
            yellow_mask = np.zeros((new_border_mask.shape[0], new_border_mask.shape[1], 4), dtype=np.uint8)
            yellow_mask[new_border_mask == 255] = [255, 255, 0, 160]
            fig.add_layout_image(dict(source=Image.fromarray(yellow_mask, 'RGBA'), xref="x", yref="y", x=0, y=0, sizex=img_pil.width, sizey=img_pil.height, sizing="stretch", opacity=1.0, layer="above"))
            fig.update_layout(title="Border Mask (Yellow = Border Ring)")
            
        if relayout_data:
            if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
                fig.update_layout(xaxis=dict(range=[relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]),
                                  yaxis=dict(range=[relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]))
        if selection_data and 'shapes' in selection_data:
            fig.update_layout(shapes=selection_data['shapes'])
        
        success_notification = dbc.Alert([html.I(className="fas fa-magic me-2"), "Area erased successfully!"], color="info", className="d-flex align-items-center")
        style = {"position": "fixed", "bottom": "20px", "left": "20px", "zIndex": 9999, "display": "block"}

        return new_mask_b64, new_pores_mask_b64, new_border_mask_b64, result_full, {"display": "block"}, result_border, {"display": "block"}, fig, success_notification, style
        
    except Exception as e:
        error = dbc.Alert([html.I(className="fas fa-times me-2"), f"Error editing mask: {str(e)}"], color="danger", className="d-flex align-items-center")
        style = {"position": "fixed", "bottom": "20px", "left": "20px", "zIndex": 9999, "display": "block"}
        return (dash.no_update,) * 8 + (error, style)


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
    if btn_clicks is None or not selection_data or not image_data or 'shapes' not in selection_data:
        return "", {"display": "none"}, dash.no_update, dash.no_update
    
    fast_mask_enabled = 'fast_mask' in (fast_mask_value or [])
    if processing_size is None or processing_size < 128: processing_size = 512
    
    kernel_ratios = [1/400, 1/200, 1/100, 1/50, 1/25, 1/10]
    open_kernel_ratio = kernel_ratios[open_kernel_ratio_idx] if 0 <= open_kernel_ratio_idx < len(kernel_ratios) else 1/200
    close_kernel_ratio = kernel_ratios[close_kernel_ratio_idx] if 0 <= close_kernel_ratio_idx < len(kernel_ratios) else 1/200
    
    try:
        img_bytes = base64.b64decode(image_data)
        img_pil = Image.open(io.BytesIO(img_bytes))
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cropped_img = crop_image_from_selection(img_cv, selection_data)
        
        if cropped_img.size > 0:
            porosity, binary_img, mask, combined_mask, border_porosity, border_combined_mask, border_mask = analyze_porosity(
                cropped_img, 
                crop_legend_enabled=False, crop_height=None, open_kernel_ratio=open_kernel_ratio,
                close_kernel_ratio=close_kernel_ratio, manual_threshold=threshold_value, mask_threshold=0,
                use_area_of_interest=False, open_iterations=open_iterations, close_iterations=close_iterations,
                border_pixels=None, border_ratio=0.10, fast_mask_enabled=fast_mask_enabled, processing_size=processing_size
            )
            
            if current_pores_state and selection_data and 'shapes' in selection_data:
                shape = selection_data['shapes'][-1]
                x0, y0 = int(shape['x0']), int(shape['y0'])
                x1, y1 = int(shape['x1']), int(shape['y1'])
                
                height, width = img_cv.shape[:2]
                x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
                y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
                
                full_size_pores_mask = np.zeros((height, width), dtype=np.uint8)
                region_height, region_width = y1 - y0, x1 - x0
                if region_height > 0 and region_width > 0:
                    resized_pores_mask = cv2.resize(combined_mask, (region_width, region_height), interpolation=cv2.INTER_NEAREST)
                    full_size_pores_mask[y0:y1, x0:x1] = resized_pores_mask
                
                fig = create_image_figure(img_pil, "Pores Mask (Bright Blue = Pores)")
                blue_mask = np.zeros((full_size_pores_mask.shape[0], full_size_pores_mask.shape[1], 4), dtype=np.uint8)
                blue_mask[full_size_pores_mask == 255] = [0, 150, 255, 200]
                fig.add_layout_image(dict(source=Image.fromarray(blue_mask, 'RGBA'), xref="x", yref="y", x=0, y=0, sizex=img_pil.width, sizey=img_pil.height, sizing="stretch", opacity=1.0, layer="above"))
                
                if current_figure and 'layout' in current_figure:
                    if 'xaxis' in current_figure['layout'] and 'range' in current_figure['layout']['xaxis']:
                        fig.update_layout(xaxis=dict(range=current_figure['layout']['xaxis']['range']))
                    if 'yaxis' in current_figure['layout'] and 'range' in current_figure['layout']['yaxis']:
                        fig.update_layout(yaxis=dict(range=current_figure['layout']['yaxis']['range']))
                
                if relayout_data:
                    if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
                        fig.update_layout(xaxis=dict(range=[relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]))
                    if 'yaxis.range[0]' in relayout_data and 'yaxis.range[1]' in relayout_data:
                        fig.update_layout(yaxis=dict(range=[relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]))
                
                if selection_data and 'shapes' in selection_data:
                    fig.update_layout(shapes=selection_data['shapes'])
                
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
        return f"Error: {str(e)}", {"display": "block"}, dash.no_update, dash.no_update


@app.callback(
    Output('image-graph', 'figure', allow_duplicate=True),
    [Input('mask-toggle-store', 'data'),
     Input('pores-toggle-store', 'data'),
     Input('border-toggle-store', 'data'),
     Input('grain-toggle-store', 'data'),
     Input('image-store', 'data')],
    [State('selection-store', 'data'),
     State('image-graph', 'relayoutData'),
     State('mask-store', 'data'),
     State('pores-mask-store', 'data'),
     State('border-mask-store', 'data'),
     State('grain-mask-store', 'data')],
    prevent_initial_call=True
)
def toggle_mask_view(show_mask, show_pores_enabled, show_border_enabled, show_grain_enabled, image_data, selection_data, relayout_data, mask_b64, pores_mask_b64, border_mask_b64, grain_mask_b64):
    """Fast toggle between image layers using cached masks while preserving drawn rectangles"""
    if image_data is None:
        return dash.no_update
    
    try:
        img_bytes = base64.b64decode(image_data)
        img_pil = Image.open(io.BytesIO(img_bytes))
        title = "Uploaded Image"
        fig = create_image_figure(img_pil, title)
        
        if show_mask and not show_pores_enabled and not show_border_enabled and not show_grain_enabled:
            if mask_b64:
                try:
                    mask = decode_mask_from_base64(mask_b64)
                    if mask is not None:
                        red_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                        red_mask[mask == 255] = [255, 0, 0, 127]
                        fig.add_layout_image(dict(source=Image.fromarray(red_mask, 'RGBA'), xref="x", yref="y", x=0, y=0, sizex=img_pil.width, sizey=img_pil.height, sizing="stretch", opacity=1.0, layer="above"))
                        title = "Area of Interest Mask (Red = Analysis Area)"
                except Exception: pass
        
        elif show_pores_enabled and not show_mask and not show_border_enabled and not show_grain_enabled:
            if pores_mask_b64:
                try:
                    pores_mask = decode_mask_from_base64(pores_mask_b64)
                    if pores_mask is not None:
                        blue_mask = np.zeros((pores_mask.shape[0], pores_mask.shape[1], 4), dtype=np.uint8)
                        blue_mask[pores_mask == 255] = [0, 150, 255, 200]
                        fig.add_layout_image(dict(source=Image.fromarray(blue_mask, 'RGBA'), xref="x", yref="y", x=0, y=0, sizex=img_pil.width, sizey=img_pil.height, sizing="stretch", opacity=1.0, layer="above"))
                        title = "Pores Mask (Bright Blue = Pores)"
                except Exception: pass
        
        elif show_border_enabled and not show_mask and not show_pores_enabled and not show_grain_enabled:
            if border_mask_b64:
                try:
                    border_mask = decode_mask_from_base64(border_mask_b64)
                    if border_mask is not None:
                        yellow_mask = np.zeros((border_mask.shape[0], border_mask.shape[1], 4), dtype=np.uint8)
                        yellow_mask[border_mask == 255] = [255, 255, 0, 160]
                        fig.add_layout_image(dict(source=Image.fromarray(yellow_mask, 'RGBA'), xref="x", yref="y", x=0, y=0, sizex=img_pil.width, sizey=img_pil.height, sizing="stretch", opacity=1.0, layer="above"))
                        title = "Border Mask (Yellow = Border Ring)"
                except Exception: pass

        elif show_grain_enabled and not show_mask and not show_pores_enabled and not show_border_enabled:
            if grain_mask_b64:
                try:
                    # Grain mask is encoded as full RGBA transparent PNG
                    overlay_mask = decode_mask_from_base64(grain_mask_b64, is_color=True)
                    if overlay_mask is not None:
                        fig.add_layout_image(dict(source=Image.fromarray(overlay_mask, 'RGBA'), xref="x", yref="y", x=0, y=0, sizex=img_pil.width, sizey=img_pil.height, sizing="stretch", opacity=1.0, layer="above"))
                        title = "Grain Size Visualization (Green = Edges, Red = Grid Lines)"
                except Exception: pass
        
        fig.update_layout(title=title)
        
        if relayout_data:
            if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
                fig.update_layout(xaxis=dict(range=[relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]))
            if 'yaxis.range[0]' in relayout_data and 'yaxis.range[1]' in relayout_data:
                fig.update_layout(yaxis=dict(range=[relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]))
        
        if selection_data and 'shapes' in selection_data:
            fig.update_layout(shapes=selection_data['shapes'])
        
        return fig
    except Exception as e:
        return dash.no_update

@app.callback(
    [Output('notification', 'children'),
     Output('notification', 'style')],
    Input('btn-full-porosity', 'n_clicks'),
    State('image-store', 'data'),
    prevent_initial_call=True
)
def show_updating_notification(btn_clicks, image_data):
    """Show updating notification when calculation button is clicked"""
    if btn_clicks is None or image_data is None: return "", {"display": "none"}
    note = dbc.Alert([html.I(className="fas fa-spinner fa-spin me-2"), "Updating..."], color="warning", className="d-flex align-items-center")
    style = {"position": "fixed", "bottom": "20px", "left": "20px", "zIndex": 9999, "display": "block"}
    return note, style

@app.callback(
    [Output('notification', 'style', allow_duplicate=True),
     Output('notification-interval', 'disabled', allow_duplicate=True)],
    Input('notification-interval', 'n_intervals'),
    prevent_initial_call=True
)
def hide_notification(n_intervals):
    return {"display": "none"}, True

@app.callback(
    [Output('mask-toggle-store', 'data'),
     Output('pores-toggle-store', 'data'),
     Output('border-toggle-store', 'data'),
     Output('grain-toggle-store', 'data')],
    [Input('mask-toggle-btn', 'n_clicks'),
     Input('pores-toggle-btn', 'n_clicks'),
     Input('border-toggle-btn', 'n_clicks'),
     Input('grain-toggle-btn', 'n_clicks')],
    [State('mask-toggle-store', 'data'),
     State('pores-toggle-store', 'data'),
     State('border-toggle-store', 'data'),
     State('grain-toggle-store', 'data')],
    prevent_initial_call=True
)
def handle_toggle_button_clicks(mask_clicks, pores_clicks, border_clicks, grain_clicks, current_mask_state, current_pores_state, current_border_state, current_grain_state):
    ctx = dash.callback_context
    if not ctx.triggered: return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'mask-toggle-btn': return not current_mask_state, False, False, False
    elif button_id == 'pores-toggle-btn': return False, not current_pores_state, False, False
    elif button_id == 'border-toggle-btn': return False, False, not current_border_state, False
    elif button_id == 'grain-toggle-btn': return False, False, False, not current_grain_state
    
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    [Output('mask-toggle-btn', 'color'), Output('mask-toggle-btn', 'outline'),
     Output('pores-toggle-btn', 'color'), Output('pores-toggle-btn', 'outline'),
     Output('border-toggle-btn', 'color'), Output('border-toggle-btn', 'outline'),
     Output('grain-toggle-btn', 'color'), Output('grain-toggle-btn', 'outline')],
    [Input('mask-toggle-store', 'data'), Input('pores-toggle-store', 'data'), Input('border-toggle-store', 'data'), Input('grain-toggle-store', 'data')],
    prevent_initial_call=True
)
def update_button_appearance(mask_active, pores_active, border_active, grain_active):
    return ("danger" if mask_active else "secondary", not mask_active,
            "primary" if pores_active else "secondary", not pores_active,
            "warning" if border_active else "secondary", not border_active,
            "info" if grain_active else "secondary", not grain_active)

@app.callback(
    [Output('selection-store', 'data', allow_duplicate=True),
     Output('image-graph', 'figure', allow_duplicate=True)],
    Input('clear-rectangles-btn', 'n_clicks'),
    [State('image-store', 'data'),
     State('mask-toggle-store', 'data'),
     State('pores-toggle-store', 'data'),
     State('border-toggle-store', 'data'),
     State('grain-toggle-store', 'data'),
     State('mask-store', 'data'),
     State('pores-mask-store', 'data'),
     State('border-mask-store', 'data'),
     State('grain-mask-store', 'data')],
    prevent_initial_call=True
)
def clear_rectangles(clear_clicks, image_data, show_mask, show_pores, show_border, show_grain, mask_b64, pores_mask_b64, border_mask_b64, grain_mask_b64):
    """Clear all drawn rectangles"""
    if clear_clicks is None or image_data is None: return dash.no_update, dash.no_update
    
    try:
        img_bytes = base64.b64decode(image_data)
        img_pil = Image.open(io.BytesIO(img_bytes))
        fig = create_image_figure(img_pil, "Uploaded Image")
        title = "Uploaded Image"
        
        if show_mask and mask_b64:
            try:
                mask = decode_mask_from_base64(mask_b64)
                if mask is not None:
                    red_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                    red_mask[mask == 255] = [255, 0, 0, 127]
                    fig.add_layout_image(dict(source=Image.fromarray(red_mask, 'RGBA'), xref="x", yref="y", x=0, y=0, sizex=img_pil.width, sizey=img_pil.height, sizing="stretch", opacity=1.0, layer="above"))
                    title = "Area of Interest Mask (Red = Analysis Area)"
            except Exception: pass
            
        elif show_pores and pores_mask_b64:
            try:
                pores_mask = decode_mask_from_base64(pores_mask_b64)
                if pores_mask is not None:
                    blue_mask = np.zeros((pores_mask.shape[0], pores_mask.shape[1], 4), dtype=np.uint8)
                    blue_mask[pores_mask == 255] = [0, 150, 255, 200]
                    fig.add_layout_image(dict(source=Image.fromarray(blue_mask, 'RGBA'), xref="x", yref="y", x=0, y=0, sizex=img_pil.width, sizey=img_pil.height, sizing="stretch", opacity=1.0, layer="above"))
                    title = "Pores Mask (Bright Blue = Pores)"
            except Exception: pass
            
        elif show_border and border_mask_b64:
            try:
                border_mask = decode_mask_from_base64(border_mask_b64)
                if border_mask is not None:
                    yellow_mask = np.zeros((border_mask.shape[0], border_mask.shape[1], 4), dtype=np.uint8)
                    yellow_mask[border_mask == 255] = [255, 255, 0, 160]
                    fig.add_layout_image(dict(source=Image.fromarray(yellow_mask, 'RGBA'), xref="x", yref="y", x=0, y=0, sizex=img_pil.width, sizey=img_pil.height, sizing="stretch", opacity=1.0, layer="above"))
                    title = "Border Mask (Yellow = Border Ring)"
            except Exception: pass

        elif show_grain and grain_mask_b64:
            try:
                overlay_mask = decode_mask_from_base64(grain_mask_b64, is_color=True)
                if overlay_mask is not None:
                    fig.add_layout_image(dict(source=Image.fromarray(overlay_mask, 'RGBA'), xref="x", yref="y", x=0, y=0, sizex=img_pil.width, sizey=img_pil.height, sizing="stretch", opacity=1.0, layer="above"))
                    title = "Grain Size Visualization (Green = Edges, Red = Grid Lines)"
            except Exception: pass
            
        fig.update_layout(title=title)
        return None, fig
    except Exception as e:
        return dash.no_update, dash.no_update

@app.callback(Output("crop-height-container", "style"), [Input("crop-legend-checkbox", "value")])
def toggle_crop_height_input(crop_legend_value):
    return {"display": "block"} if crop_legend_value and "crop_legend" in crop_legend_value else {"display": "none"}

@app.callback(Output("processing-size-container", "style"), [Input("show-size-checkbox", "value")])
def toggle_processing_size_input(show_size_value):
    return {"display": "block", "marginLeft": "25px"} if show_size_value and "show_size" in show_size_value else {"display": "none", "marginLeft": "25px"}

@app.callback(Output("advanced-options-container", "style"), [Input("advanced-options-checkbox", "value")])
def toggle_advanced_options(advanced_value):
    return {"display": "block"} if advanced_value and "show_advanced" in advanced_value else {"display": "none"}

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)