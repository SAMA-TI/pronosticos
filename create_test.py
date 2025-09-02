#!/usr/bin/env python3
"""
Script de prueba para validar la implementación de ThreadPoolExecutor
con el código real pero con un subconjunto de estaciones.
"""

# Usar solo las primeras 5 estaciones para la prueba
import sys
import os

# Modifica temporalmente la lista de estaciones para la prueba
test_codes = ["101", "102", "103", "104", "106"]

# Lee el archivo original
with open('Pronosticos.py', 'r') as f:
    content = f.read()

# Reemplaza la lista de códigos para la prueba
original_codes = 'sp_codes = ["101", "102", "103", "104", "106", "108", "109", "131", "132", "133", "134", "135", \n            "136", "137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147", \n            "149", "150", "151", "152", "154", "155", "156", "157","158","159", "160", "161", "162", "163"]'

test_codes_str = f'sp_codes = {test_codes}'

# Crea el contenido de prueba
test_content = content.replace(original_codes, test_codes_str)

# También modificar para que no ejecute la app de Dash
test_content = test_content.replace('if __name__ == "__main__":', 'if False:  # __name__ == "__main__":')

# Escribir el archivo de prueba
with open('test_pronosticos_parallel.py', 'w') as f:
    f.write(test_content)

print("✅ Archivo de prueba creado: test_pronosticos_parallel.py")
print("🧪 Puedes ejecutarlo con: python3 test_pronosticos_parallel.py")
print(f"📊 Probará con {len(test_codes)} estaciones: {test_codes}")
