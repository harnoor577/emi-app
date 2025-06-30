import os
import io
import base64
import zipfile
import tempfile
import shutil
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from matplotlib.path import Path
import geopandas as gpd

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("3D ECa Map Viewer"),

    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div(['📂 Drag & Drop or ', html.A('Upload CSV')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px'
            },
            multiple=False
        ),
        dcc.Upload(
            id='upload-shapefile',
            children=html.Div(['🖉️ Drag & Drop or ', html.A('Upload Shapefile (.zip)')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px'
            },
            multiple=False
        )
    ]),

    html.Div([
        html.Label("Select ECa Layer:"),
        dcc.Dropdown(id='layer-dropdown')
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        html.Label("Z-Axis Aspect Ratio (Vertical exaggeration):"),
        dcc.Slider(
            id='aspect-z-slider',
            min=0.1, max=2.0, step=0.1, value=0.3,
            marks={i / 10: str(i / 10) for i in range(1, 21)}
        )
    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),

    html.Div([
        dcc.Checklist(
            id='lock-pan-toggle',
            options=[{'label': ' 🔒 Lock View', 'value': 'lock'}],
            value=[],
            inputStyle={"margin-right": "5px"},
            labelStyle={"font-weight": "bold"}
        )
    ], style={'margin-top': '10px'}),

    dcc.Store(id='stored-data'),
    dcc.Store(id='shapefile-mask'),
    dcc.Graph(id='3d-emi-volume')
])

@app.callback(
    Output('stored-data', 'data'),
    Output('layer-dropdown', 'options'),
    Output('layer-dropdown', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def parse_csv(contents, filename):
    if contents is None:
        return None, [], None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    emi_layers = [f'layer{i}' for i in range(1, 21) if f'layer{i}' in df.columns]
    options = [{'label': 'All Layers', 'value': 'all'}] + [
        {'label': f'{l} ({i*0.1:.1f} m)', 'value': l} for i, l in enumerate(emi_layers)
    ]
    return df.to_json(date_format='iso', orient='split'), options, 'all'

@app.callback(
    Output('shapefile-mask', 'data'),
    Input('upload-shapefile', 'contents')
)
def process_shapefile(contents):
    if contents is None:
        return None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, 'shapefile.zip')
    with open(zip_path, 'wb') as f:
        f.write(decoded)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
    if not shp_files:
        shutil.rmtree(temp_dir)
        return None
    try:
        gdf = gpd.read_file(os.path.join(temp_dir, shp_files[0]))
        polygons = gdf[gdf.geom_type.isin(['Polygon', 'MultiPolygon'])]
        if polygons.empty:
            return None
        polygon = polygons.geometry.iloc[0]
        if polygon.geom_type == 'MultiPolygon':
            polygon = polygon.geoms[0]
        coords = np.array(polygon.exterior.coords)[:, :2]
        return coords.tolist()
    except Exception as e:
        print("Shapefile read error:", e)
        return None

@app.callback(
    Output('3d-emi-volume', 'figure'),
    Input('stored-data', 'data'),
    Input('layer-dropdown', 'value'),
    Input('aspect-z-slider', 'value'),
    Input('lock-pan-toggle', 'value'),
    Input('shapefile-mask', 'data')
)
def update_figure(json_data, selected_layer, z_aspect, lock_toggle, mask_data):
    if json_data is None or selected_layer is None:
        return go.Figure()

    df = pd.read_json(json_data, orient='split')
    x, y = df['x'].values, df['y'].values
    z_elev = df['Elevation'].values
    z_slope = df['Slope'].values
    z_aspect_arr = df['Aspect'].values

    emi_layers = [f'layer{i}' for i in range(1, 21) if f'layer{i}' in df.columns]
    all_emi_vals = pd.concat([df[l] for l in emi_layers])
    zmin, zmax = all_emi_vals.quantile(0.05), all_emi_vals.quantile(0.95)

    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), 100),
        np.linspace(y.min(), y.max(), 100)
    )
    points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    if mask_data is not None:
        path = Path(np.array(mask_data))
        mask = path.contains_points(points).reshape(grid_x.shape)
    else:
        mask = np.full(grid_x.shape, True)

    grid_elev = griddata((x, y), z_elev, (grid_x, grid_y), method='linear')
    grid_slope = griddata((x, y), z_slope, (grid_x, grid_y), method='linear')
    grid_aspect = griddata((x, y), z_aspect_arr, (grid_x, grid_y), method='linear')

    grid_elev = np.where(mask, grid_elev, np.nan)
    grid_slope = np.where(mask, grid_slope, np.nan)
    grid_aspect = np.where(mask, grid_aspect, np.nan)

    fig = go.Figure()

    def hover(layername):
        return (
            f"<b>X:</b> %{{x}}<br><b>Y:</b> %{{y}}<br><b>Elevation:</b> %{{z:.2f}}<br>"
            f"<b>{layername}:</b> %{{surfacecolor:.2f}}<br>"
            f"<b>Slope:</b> %{{customdata[0]:.1f}}°<br><b>Aspect:</b> %{{customdata[1]:.1f}}°<extra></extra>"
        ), np.stack([grid_slope, grid_aspect], axis=-1)

    if selected_layer == 'all':
        for i, layer in enumerate(emi_layers):
            depth = i * 0.1
            layer_vals = griddata((x, y), df[layer], (grid_x, grid_y), method='linear')
            layer_vals = np.where(mask, layer_vals, np.nan)
            surface_z = np.where(mask, grid_elev - depth, np.nan)
            hovertemplate, customdata = hover(layer)
            fig.add_trace(go.Surface(
                x=grid_x, y=grid_y, z=surface_z,
                surfacecolor=layer_vals,
                colorscale='Viridis', cmin=zmin, cmax=zmax,
                showscale=False,
                opacity=0.98 if i != 0 else 1.0,
                customdata=customdata,
                hovertemplate=hovertemplate,
                name=layer
            ))
    else:
        i = emi_layers.index(selected_layer)
        depth = i * 0.1
        layer_vals = griddata((x, y), df[selected_layer], (grid_x, grid_y), method='linear')
        layer_vals = np.where(mask, layer_vals, np.nan)
        surface_z = np.where(mask, grid_elev - depth, np.nan)
        hovertemplate, customdata = hover(selected_layer)
        fig.add_trace(go.Surface(
            x=grid_x, y=grid_y, z=surface_z,
            surfacecolor=layer_vals,
            colorscale='Viridis', cmin=zmin, cmax=zmax,
            showscale=True,
            colorbar=dict(title=selected_layer, len=0.75),
            opacity=1.0,
            customdata=customdata,
            hovertemplate=hovertemplate,
            name=selected_layer
        ))

    layout = dict(
        title="3D EMI Map from Uploaded Files",
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Elevation',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=z_aspect)
        ),
        height=800
    )

    if 'lock' in lock_toggle:
        layout["scene_camera"] = dict(eye=dict(x=1.25, y=1.25, z=0.8))
        layout["scene_dragmode"] = False

    fig.update_layout(**layout)
    return fig

if __name__ == '__main__':
    server=app.server
    app.run_server(debug=True)
