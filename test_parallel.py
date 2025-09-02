#!/usr/bin/env python3
"""
Test script para verificar que la paralelización funciona correctamente
con un subconjunto pequeño de estaciones.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta, time as dtime, timezone
import time
import pytz
import warnings
from urllib3.exceptions import InsecureRequestWarning
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# Cache simple
_cache = {}
CACHE_DURATION = 120

def get_cached_or_fetch(cache_key, fetch_function, *args, **kwargs):
    now = time.time()
    
    if cache_key in _cache:
        cached_data, cached_time = _cache[cache_key]
        if now - cached_time < CACHE_DURATION:
            return cached_data
    
    try:
        data = fetch_function(*args, **kwargs)
        _cache[cache_key] = (data, now)
        return data
    except Exception as e:
        if cache_key in _cache:
            print(f"Fetch failed, using expired cache for {cache_key}: {e}")
            return _cache[cache_key][0]
        raise

def obtener_datos_estacion(code, calidad=1, timeout=30, max_retries=3):
    page = 1
    datos = []
    
    while True:
        url = f"https://sigran.antioquia.gov.co/api/v1/estaciones/sp_{code}/precipitacion?calidad={calidad}&page={page}"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, verify=False, timeout=timeout)
                response.raise_for_status()
                break
                
            except requests.Timeout:
                print(f"Station {code}, page {page}: Request timed out (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    print(f"Station {code}: Max retries reached, skipping")
                    return datos
                time.sleep(2 ** attempt)
                
            except requests.RequestException as e:
                print(f"Station {code}, page {page}: Request error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print(f"Station {code}: Max retries reached, skipping")
                    return datos
                time.sleep(2 ** attempt)
        else:
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
                time.sleep(2 ** attempt)
                
        except requests.RequestException as e:
            print(f"Station {code} metadata: Request error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    print(f"Station {code} metadata: Max retries reached, skipping")
    return None

def procesar_datos_simple(datos):
    """Versión simplificada para el test"""
    if not datos:
        return None
    
    return {"total_registros": len(datos)}

def procesar_estacion_completa(code):
    """Procesa una estación completa"""
    try:
        datos = get_cached_or_fetch(f"data_{code}", obtener_datos_estacion, code)
        resumen = procesar_datos_simple(datos)
        meta = get_cached_or_fetch(f"meta_{code}", obtener_metadata_sp, code)
        
        if resumen and meta:
            resumen["estacion"] = code
            meta.update(resumen)
            return {
                'success': True,
                'code': code,
                'resumen': resumen,
                'meta': meta
            }
        else:
            return {
                'success': False,
                'code': code,
                'error': 'No data or metadata available'
            }
            
    except Exception as e:
        return {
            'success': False,
            'code': code,
            'error': str(e)
        }

if __name__ == "__main__":
    # Test con solo 3 estaciones
    test_codes = ["101", "102", "103"]
    
    print("=== TEST SECUENCIAL ===")
    start_time = time.time()
    resultados_seq = []
    
    for code in test_codes:
        print(f"Processing station {code}...")
        resultado = procesar_estacion_completa(code)
        resultados_seq.append(resultado)
    
    tiempo_secuencial = time.time() - start_time
    print(f"Tiempo secuencial: {tiempo_secuencial:.1f} segundos")
    
    print("\n=== TEST PARALELO ===")
    start_time = time.time()
    resultados_par = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_code = {executor.submit(procesar_estacion_completa, code): code for code in test_codes}
        
        for future in as_completed(future_to_code):
            code = future_to_code[future]
            resultado = future.result()
            resultados_par.append(resultado)
            if resultado['success']:
                print(f"✓ Station {code}: Success")
            else:
                print(f"✗ Station {code}: {resultado['error']}")
    
    tiempo_paralelo = time.time() - start_time
    print(f"Tiempo paralelo: {tiempo_paralelo:.1f} segundos")
    
    if tiempo_secuencial > 0:
        mejora = ((tiempo_secuencial - tiempo_paralelo) / tiempo_secuencial) * 100
        print(f"\n🚀 Mejora: {mejora:.1f}% más rápido")
        print(f"Speedup: {tiempo_secuencial/tiempo_paralelo:.1f}x")
