# 🌧️ Tablero de Estaciones de Precipitación

Dashboard interactivo para monitorear estaciones meteorológicas de precipitación en Antioquia, Colombia.

## Características

- Visualización de datos en tiempo real de 38+ estaciones meteorológicas
- Acumulados de precipitación por períodos (6h, 24h, 72h)
- Acumulados meteorológicos (último día, 7 días, 30 días)
- Mapas interactivos de ubicación de estaciones
- Filtros por subregión y municipio
- Descarga de datos en formato CSV
- Gráficos de disponibilidad de datos

## Instalación Local

1. Clona el repositorio:

```bash
git clone <tu-repositorio>
cd pronosticos
```

2. Crea un entorno virtual:

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

4. Ejecuta la aplicación:

```bash
python Pronosticos.py
```

5. Abre tu navegador en: http://127.0.0.1:8056

## Despliegue en Render.com

### Preparación

1. Asegúrate de que todos los archivos estén en tu repositorio de GitHub:
   - `Pronosticos.py` (aplicación principal)
   - `requirements.txt` (dependencias)
   - `Procfile` (comando de inicio)
   - `runtime.txt` (versión de Python)
   - `Base de datos estaciones SAMA.xlsx` (base de datos)

### Pasos en Render.com

1. **Conecta tu repositorio:**

   - Ve a [render.com](https://render.com)
   - Crea una cuenta o inicia sesión
   - Conecta tu cuenta de GitHub

2. **Crea un nuevo Web Service:**

   - Click en "New +" → "Web Service"
   - Selecciona tu repositorio
   - Configura los siguientes parámetros:

3. **Configuración del servicio:**

   ```
   Name: tablero-precipitacion (o el nombre que prefieras)
   Environment: Python 3
   Region: Oregon (US West) o la más cercana
   Branch: main (o tu rama principal)
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn Pronosticos:app.server
   ```

4. **Variables de entorno (opcional):**

   ```
   PORT: (se asigna automáticamente)
   HOST: 0.0.0.0
   ```

5. **Deploy:**
   - Click en "Create Web Service"
   - Espera a que termine el despliegue (puede tomar varios minutos)

### Consideraciones importantes

- **Tiempo de carga:** La aplicación puede tardar varios minutos en cargar inicialmente porque consulta 38+ APIs de estaciones meteorológicas.
- **Excel file:** Asegúrate de incluir el archivo `Base de datos estaciones SAMA.xlsx` en tu repositorio.
- **Limitaciones de memoria:** Si encuentras problemas de memoria, considera reducir el número de estaciones o implementar caché.

## Estructura del Proyecto

```
pronosticos/
├── Pronosticos.py              # Aplicación principal Dash
├── requirements.txt            # Dependencias Python
├── Procfile                   # Comando de inicio para Render
├── runtime.txt                # Versión de Python
├── Base de datos estaciones SAMA.xlsx  # Base de datos estaciones
├── .gitignore                 # Archivos a ignorar en Git
└── README.md                  # Este archivo
```

## Tecnologías Utilizadas

- **Python 3.12**
- **Dash/Plotly** - Framework web y visualizaciones
- **Pandas** - Manipulación de datos
- **Requests** - Llamadas a APIs
- **Gunicorn** - Servidor web para producción

## API Data Source

Los datos se obtienen de: `https://sigran.antioquia.gov.co/api/v1/estaciones/`

## Troubleshooting

### Error de archivo Excel

Si el archivo Excel no se encuentra, la aplicación continuará funcionando con funcionalidad limitada.

### Timeout en APIs

Si las APIs externas no responden, algunas estaciones pueden no mostrar datos.

### Problemas de memoria en Render

- Reduce el número de estaciones en `sp_codes`
- Implementa caché para las llamadas a API
- Considera usar un plan de Render con más memoria
