#!/usr/bin/env python3
"""
Test rápido para verificar que el manejo de errores funciona correctamente
"""

import pandas as pd

# Simular listas vacías como las que podrían ocurrir en Render
resultados = []
metadata = []

print("Testing error handling with empty results...")

# Crear DataFrames como en el código principal
df_meta = pd.DataFrame(metadata)

# Convertir a DataFrame para resumen (sin la serie de 120h)
if resultados:
    df_resultado = pd.DataFrame([{k: v for k, v in r.items() if k != "serie_120h"} for r in resultados])
    
    # Verificar que las columnas necesarias existen antes de ordenar
    required_columns = ["datos_recientes", "fecha_ultimo_dato"]
    missing_columns = [col for col in required_columns if col not in df_resultado.columns]
    
    if missing_columns:
        print(f"⚠️  Warning: Missing columns in df_resultado: {missing_columns}")
        print(f"Available columns: {list(df_resultado.columns)}")
        # Agregar columnas faltantes con valores por defecto
        for col in missing_columns:
            if col == "datos_recientes":
                df_resultado[col] = 0
            elif col == "fecha_ultimo_dato":
                df_resultado[col] = pd.Timestamp.now()
    
    # Ordenar por defecto: estaciones con datos recientes primero, y por fecha más reciente
    df_resultado = df_resultado.sort_values(by=["datos_recientes", "fecha_ultimo_dato"], ascending=[False, False])
    
    # Crear copia con etiquetas legibles
    df_pie = df_resultado.copy()
    df_pie['datos_recientes'] = df_pie['datos_recientes'].map({1: 'Reciente', 0: 'No reciente'})
else:
    print("⚠️  Warning: No data loaded from stations. Creating empty DataFrames.")
    # Crear DataFrames vacíos con las columnas esperadas
    df_resultado = pd.DataFrame(columns=[
        "estacion", "acum_6h", "acum_24h", "acum_72h", 
        "ultimo_dia_meteorologico", "ultimos_7_dias_meteorologicos", "ultimos_30_dias_meteorologicos",
        "datos_recientes", "dias_sin_datos", "fecha_ultimo_dato"
    ])
    df_pie = df_resultado.copy()
    df_pie['datos_recientes'] = df_pie['datos_recientes'].map({1: 'Reciente', 0: 'No reciente'})

# Registros con datos recientes (menos de 7 días sin datos)
if not df_resultado.empty and "dias_sin_datos" in df_resultado.columns:
    df_reciente = df_resultado[df_resultado["dias_sin_datos"] < 7].copy()
    df_reciente = df_reciente.sort_values(by='estacion', ascending=True)
else:
    df_reciente = pd.DataFrame(columns=df_resultado.columns)

# Registros sin datos recientes (7 días o más sin datos)
if not df_resultado.empty and "dias_sin_datos" in df_resultado.columns:
    df_no_reciente = df_resultado[df_resultado["dias_sin_datos"] >= 7].copy()
else:
    df_no_reciente = pd.DataFrame(columns=df_resultado.columns)

print("✅ Error handling test completed successfully!")
print(f"df_resultado shape: {df_resultado.shape}")
print(f"df_reciente shape: {df_reciente.shape}")
print(f"df_no_reciente shape: {df_no_reciente.shape}")
print(f"Columns in df_resultado: {list(df_resultado.columns)}")
