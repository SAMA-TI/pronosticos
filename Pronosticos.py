#!pip install dash plotly
#!pip install dash-bootstrap-components
#!pip install -U plotly ipywidgets
#!pip install dash flask

import os
import requests
import pandas as pd
from datetime import datetime, timedelta, time as dtime, timezone
import time  # para medir tiempo
import math
import pytz
import warnings
from urllib3.exceptions import InsecureRequestWarning
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import dash.dash_table as dt
from dash import dcc
import sys
import flask

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# Simple in-memory cache for API responses (2 minutes max to ensure fresh data)
_cache = {}
CACHE_DURATION = 120  # 2 minutes in seconds

def get_cached_or_fetch(cache_key, fetch_function, *args, **kwargs):
    """
    Simple cache mechanism with 2-minute expiration.
    Since data expires every 5 minutes, this ensures we get fresh data
    while reducing redundant API calls during deployment.
    """
    now = time.time()
    
    if cache_key in _cache:
        cached_data, cached_time = _cache[cache_key]
        if now - cached_time < CACHE_DURATION:
            return cached_data
    
    # Cache miss or expired, fetch new data
    try:
        data = fetch_function(*args, **kwargs)
        _cache[cache_key] = (data, now)
        return data
    except Exception as e:
        # If fetch fails and we have expired cache, return it anyway
        if cache_key in _cache:
            print(f"Fetch failed, using expired cache for {cache_key}: {e}")
            return _cache[cache_key][0]
        raise

# Estaciones de precipitación (sp)
sp_codes = ["101", "102", "103", "104", "106", "108", "109", "131", "132", "133", "134", "135", 
            "136", "137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147", 
            "149", "150", "151", "152", "154", "155", "156", "157","158","159", "160", "161", "162", "163"]

def obtener_datos_estacion(code, calidad=1, timeout=30, max_retries=3):
    page = 1
    datos = []
    
    while True:
        url = f"https://sigran.antioquia.gov.co/api/v1/estaciones/sp_{code}/precipitacion?calidad={calidad}&page={page}"
        
        # Retry logic for each request
        for attempt in range(max_retries):
            try:
                response = requests.get(url, verify=False, timeout=timeout)
                response.raise_for_status()
                break  # Success, exit retry loop
                
            except requests.Timeout:
                print(f"Station {code}, page {page}: Request timed out (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:  # Last attempt
                    print(f"Station {code}: Max retries reached, skipping")
                    return datos
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.RequestException as e:
                print(f"Station {code}, page {page}: Request error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:  # Last attempt
                    print(f"Station {code}: Max retries reached, skipping")
                    return datos
                time.sleep(2 ** attempt)  # Exponential backoff
        else:
            # If we exit the retry loop without breaking, we failed all attempts
            return datos
            
        if response.status_code != 200:
            print(f"Station {code}, page {page}: HTTP {response.status_code}")
            break
            
        try:
            data = response.json()
        except ValueError as e:
            print(f"Station {code}, page {page}: JSON decode error: {e}")
            break
            
        values = data.get("values", [])
        if not values:
            break
            
        datos.extend(values)
        page += 1
        
        # Paramos si ya tenemos más de 72 horas de datos
        fechas = [pd.to_datetime(d['fecha']) for d in datos]
        if fechas and (max(fechas) - min(fechas)).total_seconds() > 72 * 3600:
            break
            
    return datos

def obtener_metadata_sp(code, timeout=30, max_retries=3):
    url = f"https://sigran.antioquia.gov.co/api/v1/estaciones/sp_{code}/"
    
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, verify=False, timeout=timeout)
            if resp.status_code == 200:
                d = resp.json()
                return {
                    "estacion": code,
                    "codigo": d.get("codigo"),
                    "descripcion": d.get("descripcion"),
                    "nombre_web": d.get("nombre_web"),
                    "latitud": float(d.get("latitud", 0)),
                    "longitud": float(d.get("longitud", 0)),
                    "municipio": d.get("municipio"),
                    "region": d.get("region")
                }
            else:
                print(f"Station {code} metadata: HTTP {resp.status_code}")
                return None
                
        except requests.Timeout:
            print(f"Station {code} metadata: Request timed out (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                
        except requests.RequestException as e:
            print(f"Station {code} metadata: Request error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    
    print(f"Station {code} metadata: Max retries reached, skipping")
    return None

def procesar_datos(datos, ahora=None):
    if not datos:
        return None

    df = pd.DataFrame(datos)
    df["fecha"] = pd.to_datetime(df["fecha"], utc=True)
    df["muestra"] = pd.to_numeric(df["muestra"], errors='coerce')

    ahora = ahora or datetime.utcnow().replace(tzinfo=pytz.UTC)

    acumulados = {
        "acum_6h": df[(df["fecha"] > ahora - timedelta(hours=6)) & (df["fecha"] <= ahora)]["muestra"].sum(),
        "acum_24h": df[(df["fecha"] > ahora - timedelta(hours=24)) & (df["fecha"] <= ahora)]["muestra"].sum(),
        "acum_72h": df[(df["fecha"] > ahora - timedelta(hours=72)) & (df["fecha"] <= ahora)]["muestra"].sum()
    }

    # Serie de 120 horas
    serie_120h = []
    for h in range(1, 121):
        t_ini = ahora - timedelta(hours=h)
        t_fin = ahora - timedelta(hours=h-1)
        val = df[(df["fecha"] > t_ini) & (df["fecha"] <= t_fin)]["muestra"].sum()
        serie_120h.append({"hora": t_ini, "acumulado": val})

    # Día meteorológico: 7am local = 12:00 UTC del día anterior
    ult_utc = ahora.replace(minute=0, second=0, microsecond=0)
    if ult_utc.hour < 12:
        dia_base = ult_utc - timedelta(days=1)
    else:
        dia_base = ult_utc
    inicio_dia = dia_base.replace(hour=12)
    
    #def acum_dias(n):
        #return df[(df["fecha"] > inicio_dia - timedelta(days=n)) & (df["fecha"] <= inicio_dia)]["muestra"].sum()
    
    #meteo = {
        #"ultimo_dia_meteorologico": acum_dias(1),
        #"ultimos_7_dias_meteorologicos": acum_dias(7),
        #"ultimos_30_dias_meteorologicos": acum_dias(30)
    #}

    def acum_dias_meteorologicos(n, df):
        # Momento actual en UTC
        ahora = datetime.now(timezone.utc)

        # 7:00 AM UTC del día actual (inicio del día meteorológico actual)
        inicio_meteo = datetime.combine(ahora.date(), dtime(7, 0, tzinfo=timezone.utc))


        # Rango del día meteorológico: de (hace n días a las 7 AM) hasta (hoy a las 7 AM)
        fecha_inicio = inicio_meteo - timedelta(days=n)
        fecha_fin = inicio_meteo

        # Filtrar y sumar la columna 'muestra' en ese rango
        return df[(df["fecha"] > fecha_inicio) & (df["fecha"] <= fecha_fin)]["muestra"].sum()

    # Diccionario con los acumulados meteorológicos
    meteo = {
        "ultimo_dia_meteorologico": acum_dias_meteorologicos(1, df),
        "ultimos_7_dias_meteorologicos": acum_dias_meteorologicos(7, df),
        "ultimos_30_dias_meteorologicos": acum_dias_meteorologicos(30, df)
    }

    
    fecha_max = df["fecha"].max()
    dias_sin_datos = (ahora - fecha_max).days
    datos_recientes = int((ahora - fecha_max) <= timedelta(days=1))
    

    return {
        **acumulados,
        **meteo,
        "datos_recientes": datos_recientes,
        "dias_sin_datos": dias_sin_datos,
        "fecha_ultimo_dato": fecha_max, 
        "serie_120h": serie_120h
    }

resultados = []
metadata = []

print(f"Starting data collection from {len(sp_codes)} stations...")
start_time = time.time()

for i, code in enumerate(sp_codes, 1):
    print(f"Processing station {code} ({i}/{len(sp_codes)})...")
    
    try:
        # Use cache for both data and metadata
        datos = get_cached_or_fetch(f"data_{code}", obtener_datos_estacion, code)
        resumen = procesar_datos(datos)
        meta = get_cached_or_fetch(f"meta_{code}", obtener_metadata_sp, code)
        
        if resumen and meta:
            resumen["estacion"] = code
            meta.update(resumen)
            resultados.append(resumen)
            metadata.append(meta)
            print(f"✓ Station {code}: Data loaded successfully")
        else:
            print(f"✗ Station {code}: No data or metadata available")
            
    except Exception as e:
        print(f"✗ Station {code}: Unexpected error: {e}")
        continue

elapsed_time = time.time() - start_time
print(f"Data collection completed: {len(resultados)} stations loaded in {elapsed_time:.1f} seconds")

df_meta = pd.DataFrame(metadata)

# Convertir a DataFrame para resumen (sin la serie de 120h)
df_resultado = pd.DataFrame([{k: v for k, v in r.items() if k != "serie_120h"} for r in resultados])

# Ordenar por defecto: estaciones con datos recientes primero, y por fecha más reciente
df_resultado = df_resultado.sort_values(by=["datos_recientes", "fecha_ultimo_dato"], ascending=[False, False])

# Crear copia con etiquetas legibles
df_pie = df_resultado.copy()
df_pie['datos_recientes'] = df_pie['datos_recientes'].map({1: 'Reciente', 0: 'No reciente'})

# Registros con datos recientes (menos de 7 días sin datos)
df_reciente = df_resultado[df_resultado["dias_sin_datos"] < 7].copy()
df_reciente = df_reciente.sort_values(by='estacion', ascending=True)


# Registros sin datos recientes (7 días o más sin datos)
df_no_reciente = df_resultado[df_resultado["dias_sin_datos"] >= 7].copy()

#Cruce de Info API con datos de Municipio y Subregión
# Cargar el archivo Excel (Base de datos estaciones SAMA)
try:
    df_excel = pd.read_excel('Base de datos estaciones SAMA.xlsx', usecols=[
        'GRUPO', 'MUNICIPIO', 'NOM_EST', 'COD_EST', 'TIPO', 'COMUN_PRIORIZ', 'CORRIENTE', 'LAT', 'LONG'
    ])
except FileNotFoundError:
    # Create a fallback empty DataFrame if Excel file is not found
    print("Warning: Excel file not found. Creating empty DataFrame.")
    df_excel = pd.DataFrame(columns=[
        'GRUPO', 'MUNICIPIO', 'NOM_EST', 'COD_EST', 'TIPO', 'COMUN_PRIORIZ', 'CORRIENTE', 'LAT', 'LONG'
    ])

# Reorganizar las columnas como se indicó
df_excel = df_excel[['COD_EST', 'TIPO', 'GRUPO', 'MUNICIPIO', 'NOM_EST', 'COMUN_PRIORIZ', 'CORRIENTE', 'LAT', 'LONG']]

# LIMPIAR columna COD_EST
df_excel['COD_EST'] = df_excel['COD_EST'].astype(str).str.strip().str.lower()

#Correción de regiones erroneas (Fuerza bruta)
# Diccionario con los valores correctos
correcciones = {
    'sp_163': 8,
    'sp_149': 3,
    'sp_151': 6,
    'sp_158': 6
}

# Aplicar las correcciones
for codigo, region_correcta in correcciones.items():
    df_meta.loc[df_meta['codigo'] == codigo, 'region'] = region_correcta

# 1. Renombrar la columna 'municipio' en df_meta
df_meta = df_meta.rename(columns={'municipio': 'municipio_num'})

# 2. Crear un DataFrame auxiliar con solo las columnas necesarias de df_excel
df_municipio = df_excel[['COD_EST', 'MUNICIPIO']].rename(columns={
    'COD_EST': 'codigo',  # para que coincida con df_meta
    'MUNICIPIO': 'municipio'
})

# 3. Hacer el merge con df_meta usando la columna común 'codigo'
df_meta = df_meta.merge(df_municipio, on='codigo', how='left')

df_meta['municipio'] = df_meta['municipio'].str.capitalize()
df_meta.loc[df_meta['codigo'] == 'sp_151', 'municipio'] = 'Sonson'

#Renombrar la columna 'region' a 'subregion_num'
df_meta = df_meta.rename(columns={'region': 'subregion_num'})

# Diccionario de equivalencias de subregiones
mapa_subregiones = {
    1: 'Valle de Aburra',
    2: 'Bajo Cauca',
    3: 'Magdalena Medio',
    4: 'Nordeste',
    5: 'Norte',
    6: 'Oriente',
    7: 'Occidente',
    8: 'Suroeste',
    9: 'Urabá'
}

# Creación de la columna 'subregion' usando el diccionario
df_meta['subregion'] = df_meta['subregion_num'].map(mapa_subregiones)

#Para tablero
app = dash.Dash(__name__)
server = app.server
#app.title = " 🌧️ Tablero de estaciones de precipitación"

# DEBUGGING - Print server information for deployment logs
print("=== DEPLOYMENT DEBUG ===")
print(f"Python version: {sys.version}")
print(f"Flask version: {flask.__version__}")
print(f"Dash version: {dash.__version__}")
print(f"Server object type: {type(server)}")
print(f"Server object id: {id(server)}")
print(f"Module globals contains 'server': {'server' in globals()}")
print(f"Module name: {__name__}")
print("========================")

# Make sure server is available at module level
globals()['server'] = server

# Configure server for deployment with explicit port and host binding
server.config.update(
    PORT=int(os.environ.get('PORT', 8056)),
    HOST=os.environ.get('HOST', '0.0.0.0')
)


app.layout = html.Div([
    html.H1(
    "🌧️ Tablero de estaciones de precipitación",
    style={"textAlign": "center"}
    ),

    # Filtros
    html.Div([
        html.Label("Filtrar por subregión:"),
        dcc.Dropdown(
            id="subregion-dropdown",
            options=[{"label": r, "value": r} for r in sorted(df_meta["subregion"].unique())],
            placeholder="Selecciona una región",
            value=None,
            clearable=True
        ),

        html.Label("Filtrar por municipio:"),
        dcc.Dropdown(
            id="municipio-dropdown",
            options=[],
            placeholder="Selecciona un municipio",
            value=None,
            clearable=True
        ),

        html.Label("Selecciona estación:"),
        dcc.Dropdown(
            id="estacion-dropdown",
            options=[],  # Se llena dinámicamente
            value=None
        ),
    ], style={"marginBottom": "20px"}),

    # Serie de tiempo
    dcc.Graph(id="serie-120h-graph"),

    # Mapa
    dcc.Graph(id="mapa-ubicacion"),

    # Tabla de acumulados horas con botón de descarga arriba
    html.H3("Acumulados recientes por estación"),
    html.Div([
        html.Button("Descargar CSV", id="btn-descargar", n_clicks=0),
        dcc.Download(id="descarga-tabla")
    ], style={"marginBottom": "10px"}),

    dash_table.DataTable(
        id='tabla-acumulados',
        columns=[
            {"name": "Estación", "id": "estacion"},
            {"name": "Acum. 6h", "id": "acum_6h"},
            {"name": "Acum. 24h", "id": "acum_24h"},
            {"name": "Acum. 72h", "id": "acum_72h"}
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center"},
        style_header={"backgroundColor": "#e1e1e1", "fontWeight": "bold"},
        data=[]
    ),

    
    ##Acá prueba
    # Tabla de acumulados meteorológicos con botón de descarga arriba
    html.H3("Acumulados meteorológicos por estación"),
    html.Div([
        html.Button("Descargar CSV (Dias Meteorologicos)", id="btn-descargar-meteo", n_clicks=0),
        dcc.Download(id="descarga-tabla-meteo")
    ], style={"marginBottom": "10px"}),

    dash_table.DataTable(
        id='tabla-meteo',
        columns=[
            {"name": "Estación", "id": "estacion"},
            {"name": "Último día", "id": "ultimo_dia_meteorologico"},
            {"name": "Últimos 7 días", "id": "ultimos_7_dias_meteorologicos"},
            {"name": "Últimos 30 días", "id": "ultimos_30_dias_meteorologicos"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center"},
        style_header={"backgroundColor": "#e1e1e1", "fontWeight": "bold"},
        data=[]
    ),

    # Gráfico de acumulados meteorológicos
    html.H3("Acumulado meteorológico del último día por estación"),
    dcc.Graph(id="grafico-acumulados-meteo")
    ,
    #Hasta acá prueba
    
    html.H3("Estaciones sin datos por más de 7 días"),
    dcc.Graph(
        id="grafico-dias-sin-datos",
        figure=px.bar(
            df_no_reciente.assign(estacion=lambda df: "sp_" + df["estacion"].astype(str))
                     .sort_values("dias_sin_datos", ascending=False),
            #df_no_reciente.sort_values("dias_sin_datos", ascending=False),
            x="estacion", y="dias_sin_datos",
            #title="Días sin datos por estación"
        )
    ),



    html.H3("Disponibilidad de datos recientes"),
    dcc.Graph(
        id="grafico-disponibilidad",
        figure=go.Figure(
            data=[
                go.Bar(
                    name=str(label),
                    y=["Estaciones"],
                    x=[(df_pie["datos_recientes"] == label).mean() * 100],
                    orientation='h',
                    #text=[f"{(df_pie['datos_recientes'] == label).mean() * 100:.1f}%"],
                    text=[f"{(df_pie['datos_recientes'] == label).mean() * 100:.1f}% ({(df_pie['datos_recientes'] == label).sum()})"],
                    textposition='inside'
                )
                for label in df_pie["datos_recientes"].unique()
            ]
        ).update_layout(
            barmode='stack',
            title=dict(
                text="Porcentaje de estaciones con/sin datos recientes",
                x=0.0,  # Alineado a la izquierda (0.5 es centrado)
                font=dict(
                    size=15,
                    family="Arial",
                    color="black"
                )
            ),
            xaxis=dict(title="Porcentaje (%)", tickmode='linear', dtick=20, range=[0, 100]),
            yaxis=dict(showticklabels=False),
            height=200,
            margin=dict(l=30, r=30, t=40, b=30),
            legend=dict(
                orientation='h',
                yanchor="bottom",
                y=1.1,        # Mueve la leyenda hacia arriba
                xanchor="center",
                x=0.5,
                font=dict(
                    size=12    # Tamaño de la fuente de la leyenda
                )
            )
    )

    ),
])  # << aquí se cierra html.Div con toda la lista completa

# Callbacks

@app.callback(
    Output("municipio-dropdown", "options"),
    Input("subregion-dropdown", "value")
)
def actualizar_municipios(subregion):
    if not subregion:
        return []
    municipios = df_meta[df_meta["subregion"] == subregion]["municipio"].dropna().unique()
    #municipios = df_meta[df_meta["region"] == region]["municipio"].fillna("Sin municipio").unique()
    return [{"label": m, "value": m} for m in sorted(municipios)]


@app.callback(
    Output("estacion-dropdown", "options"),
    Output("estacion-dropdown", "value"),
    Input("subregion-dropdown", "value"),
    Input("municipio-dropdown", "value")
)
def actualizar_estaciones(subregion, municipio):
    df_filtrado = df_meta.copy()
    if subregion:
        df_filtrado = df_filtrado[df_filtrado["subregion"] == subregion]
    if municipio:
        df_filtrado = df_filtrado[df_filtrado["municipio"] == municipio]

    opciones = [{"label": f"sp_{e}", "value": e} for e in sorted(df_filtrado["estacion"].unique())]
    valor_default = opciones[0]["value"] if opciones else None
    return opciones, valor_default


@app.callback(
    Output("serie-120h-graph", "figure"),
    Input("estacion-dropdown", "value")
)


def update_serie_120h(estacion_id):
    serie = next((r["serie_120h"] for r in resultados if r["estacion"] == estacion_id), [])
    if not serie:
        return px.line(title="No hay datos disponibles")
    df_serie = pd.DataFrame(serie)
    fig = px.line(df_serie, x="hora", y="acumulado", title=f"Serie 120h - sp_{estacion_id}")
    fig.update_layout(xaxis_title="Hora", yaxis_title="Acumulado")
    return fig


@app.callback(
    Output("mapa-ubicacion", "figure"),
    Input("estacion-dropdown", "value")
)
def update_mapa(estacion_id):
    fila = df_meta[df_meta["estacion"] == estacion_id]
    if fila.empty:
        fig_empty = px.scatter_map(lat=[], lon=[])
        fig_empty.update_layout(
            title="Ubicación no disponible",
            mapbox_style="carto-positron",
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            showlegend=False
        )
        return fig_empty

    # Mapa base con punto azul claro
    fig_map = px.scatter_map(
        fila,
        lat="latitud",
        lon="longitud",
        hover_name="estacion",
        hover_data=["municipio", "subregion"],
        color_discrete_sequence=["#1f77b4"],  # Azul claro
        zoom=10,
        title="📍 Ubicación de la estación seleccionada"
    )

    # Marcador personalizado (más pequeño, visible, sin interferencias)
    fig_map.add_trace(go.Scattermap(
        lat=fila["latitud"],
        lon=fila["longitud"],
        mode='markers',
        marker=go.scattermap.Marker(
            size=20,
            symbol='marker',
            color='red',
            opacity=0.5
        ),
        hoverinfo='skip',
        showlegend=False
    ))

    fig_map.update_layout(
        mapbox_style="carto-positron",
        margin={"r": 10, "t": 40, "l": 10, "b": 10},
        showlegend=False  # 🔴 Oculta la leyenda
    )

    return fig_map




@app.callback(
    Output("tabla-acumulados", "data"),
    Input("subregion-dropdown", "value"),
    Input("municipio-dropdown", "value")
)
def actualizar_tabla(subregion, municipio):
    df = df_reciente.merge(df_meta[["estacion", "subregion", "municipio"]], on="estacion", how="left")
    if subregion:
        df = df[df["subregion"] == subregion]
    if municipio:
        df = df[df["municipio"] == municipio]

    # Redondeamos columnas de acumulados a 3 decimales
    df[["acum_6h", "acum_24h", "acum_72h"]] = df[["acum_6h", "acum_24h", "acum_72h"]].round(3)

    return [
        {
            "estacion": f"sp_{row['estacion']}",
            "acum_6h": row["acum_6h"],
            "acum_24h": row["acum_24h"],
            "acum_72h": row["acum_72h"]
        } for _, row in df.iterrows()
    ]

##Acá prueba
## Callback para la tabla de acumulados meteorologicos
@app.callback(
    Output("tabla-meteo", "data"),
    Input("subregion-dropdown", "value"),
    Input("municipio-dropdown", "value")
)
def actualizar_tabla_meteo(subregion, municipio):
    df = df_reciente.merge(df_meta[["estacion", "subregion", "municipio"]], on="estacion", how="left")
    if subregion:
        df = df[df["subregion"] == subregion]
    if municipio:
        df = df[df["municipio"] == municipio]

    df[["ultimo_dia_meteorologico", "ultimos_7_dias_meteorologicos", "ultimos_30_dias_meteorologicos"]] = df[[
        "ultimo_dia_meteorologico", "ultimos_7_dias_meteorologicos", "ultimos_30_dias_meteorologicos"]].round(3)

    return [
        {
            "estacion": f"sp_{row['estacion']}",
            "ultimo_dia_meteorologico": row["ultimo_dia_meteorologico"],
            "ultimos_7_dias_meteorologicos": row["ultimos_7_dias_meteorologicos"],
            "ultimos_30_dias_meteorologicos": row["ultimos_30_dias_meteorologicos"]
        } for _, row in df.iterrows()
    ]

##Hasta acá prueba
## Callback para descargar CSV de la tabla meteorológica
@app.callback(
    Output("descarga-tabla", "data"),
    Input("btn-descargar", "n_clicks"),
    prevent_initial_call=True
)
def descargar_tabla(n_clicks):
    df_to_download = df_resultado[["estacion", "acum_6h", "acum_24h", "acum_72h"]].copy()
    df_to_download["estacion"] = df_to_download["estacion"].apply(lambda x: f"sp_{x}")
    return dcc.send_data_frame(df_to_download.to_csv, filename="acumulados_estaciones.csv", index=False)

##Acá prueba
@app.callback(
    Output("descarga-tabla-meteo", "data"),
    Input("btn-descargar-meteo", "n_clicks"),
    prevent_initial_call=True
)
def descargar_tabla_meteo(n_clicks):
    df_to_download = df_reciente[["estacion", "ultimo_dia_meteorologico", "ultimos_7_dias_meteorologicos", "ultimos_30_dias_meteorologicos"]].copy()
    df_to_download["estacion"] = df_to_download["estacion"].apply(lambda x: f"sp_{x}")
    return dcc.send_data_frame(df_to_download.to_csv, filename="acumulados_meteorologicos.csv", index=False)



## Callback para gráfica de los acumulados meteorológicos
@app.callback(
    Output("grafico-acumulados-meteo", "figure"),
    Input("subregion-dropdown", "value"),
    Input("municipio-dropdown", "value")
)
def actualizar_grafico_meteo(subregion, municipio):
    df = df_reciente.merge(df_meta[["estacion", "subregion", "municipio"]], on="estacion", how="left")
    
    if subregion:
        df = df[df["subregion"] == subregion]
    if municipio:
        df = df[df["municipio"] == municipio]

    if df.empty:
        return px.bar(title="No hay datos para mostrar")

    df = df.copy()
    df["estacion"] = df["estacion"].apply(lambda x: f"sp_{x}")

    fig = px.bar(
        df,
        x="estacion",
        y="ultimo_dia_meteorologico",
        #title="Acumulado meteorológico del último día por estación"
    )

    fig.update_layout(
        xaxis_title="Estación",
        yaxis_title="Acumulado (mm)",
        showlegend=False,
        xaxis_tickangle=90  # Ángulo de etiqueta datos
    )

    return fig

## Hasta acá prueba

# # Expose the server for Gunicorn (required for Render.com)
# server = app.server

if __name__ == "__main__":
    # Use the same configuration as defined in server.config
    port = server.config.get('PORT', 8056)
    host = server.config.get('HOST', '0.0.0.0')
    
    # Run the app
    app.run(host=host, port=port, debug=False)


