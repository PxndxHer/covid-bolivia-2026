"""Ejecuta el pipeline completo: datos → EDA → modelos → figuras → web.

Uso (desde cualquier carpeta):
    python run_all.py

Imprime el tiempo de cada paso y termina con código de error distinto de cero
si alguno falla, para que la integración continua lo detecte.
"""

import sys
import time
import traceback
from pathlib import Path

# Permite ejecutar el script desde otra carpeta: la raíz del proyecto es la de este archivo.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import datos, eda, figuras, modelo_perfiles, modelo_riesgo, modelo_series, web  # noqa: E402
from src.config import asegurar_directorios  # noqa: E402

PASOS = [
    ("Datos y controles de calidad", datos.ejecutar),
    ("Análisis exploratorio", eda.ejecutar),
    ("Modelo 1 · perfiles de países", modelo_perfiles.ejecutar),
    ("Modelo 2 · pronóstico e IPS", modelo_series.ejecutar),
    ("Modelo 3 · riesgo municipal", modelo_riesgo.ejecutar),
    ("Figuras", figuras.generar_todas),
    ("Página web", web.ejecutar),
]


def ejecutar_paso(nombre: str, funcion) -> float:
    """Ejecuta un paso, imprime su duración y la devuelve en segundos."""
    print(f"> {nombre}...", flush=True)
    inicio = time.perf_counter()
    funcion()
    duracion = time.perf_counter() - inicio
    print(f"  listo en {duracion:.1f} s", flush=True)
    return duracion


def main() -> int:
    """Corre todos los pasos en orden; se detiene en el primero que falle."""
    # Las consolas de Windows pueden usar una codificación sin tildes completas; se reemplaza
    # cualquier carácter no representable en vez de interrumpir el pipeline.
    for flujo in (sys.stdout, sys.stderr):
        flujo.reconfigure(errors="replace")
    asegurar_directorios()
    tiempos = []
    for nombre, funcion in PASOS:
        try:
            tiempos.append((nombre, ejecutar_paso(nombre, funcion)))
        except Exception:  # noqa: BLE001 — se informa cualquier error y se corta el pipeline
            traceback.print_exc()
            print(f"ERROR: falló el paso «{nombre}»", file=sys.stderr)
            return 1
    print("\nResumen de tiempos")
    for nombre, duracion in tiempos:
        print(f"  {nombre:<32} {duracion:7.1f} s")
    total = sum(duracion for _, duracion in tiempos)
    print(f"  {'Total':<32} {total:7.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
