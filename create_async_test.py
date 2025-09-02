#!/usr/bin/env python3
"""
Test del archivo principal con implementación asíncrona
usando un subconjunto de estaciones.
"""

# Leer el archivo principal y modificarlo para el test
with open('Pronosticos.py', 'r') as f:
    content = f.read()

# Modificar para usar solo 5 estaciones en el test
original_codes = 'sp_codes = ["101", "102", "103", "104", "106", "108", "109", "131", "132", "133", "134", "135", \n            "136", "137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147", \n            "149", "150", "151", "152", "154", "155", "156", "157","158","159", "160", "161", "162", "163"]'

test_codes = 'sp_codes = ["101", "102", "103", "104", "106"]'

# Crear contenido de prueba
test_content = content.replace(original_codes, test_codes)

# Modificar para que no ejecute la app de Dash
test_content = test_content.replace('if __name__ == "__main__":', 'if False:  # __name__ == "__main__":')

# Escribir archivo de prueba
with open('test_pronosticos_async.py', 'w') as f:
    f.write(test_content)

print("✅ Archivo de prueba asíncrono creado: test_pronosticos_async.py")
print("🧪 Se probará con 5 estaciones")
print("🚀 Debería usar la implementación asíncrona automáticamente")
