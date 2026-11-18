"""Carga de la configuración del estudio y rutas absolutas del proyecto.

Todas las rutas se construyen a partir de la raíz del proyecto (la carpeta que
contiene `config.yaml`), de modo que los scripts funcionan igual sin importar
desde qué carpeta se ejecuten.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

# La raíz es la carpeta padre de `src/`. Se usa `resolve()` para obtener una
# ruta absoluta aunque el módulo se importe mediante una ruta relativa.
RAIZ = Path(__file__).resolve().parent.parent

ARCHIVO_CONFIG = RAIZ / "config.yaml"


@dataclass(frozen=True)
class Rutas:
    """Rutas absolutas de las carpetas del proyecto.

    Se agrupan en una clase inmutable para que ningún módulo construya rutas
    por su cuenta y todas apunten siempre a la misma estructura.
    """

    raiz: Path
    crudos: Path
    procesados: Path
    resultados: Path
    figuras: Path
    plantillas: Path
    docs: Path
    docs_figuras: Path
    tests: Path


RUTAS = Rutas(
    raiz=RAIZ,
    crudos=RAIZ / "datos" / "crudos",
    procesados=RAIZ / "datos" / "procesados",
    resultados=RAIZ / "resultados",
    figuras=RAIZ / "figuras",
    plantillas=RAIZ / "plantillas",
    docs=RAIZ / "docs",
    docs_figuras=RAIZ / "docs" / "figuras",
    tests=RAIZ / "tests",
)


@lru_cache(maxsize=1)
def cargar_config(archivo: Path = ARCHIVO_CONFIG) -> dict:
    """Lee `config.yaml` y devuelve su contenido como diccionario.

    Se guarda en caché porque varios módulos la consultan durante una misma
    ejecución y el archivo no cambia mientras corre el pipeline.
    """
    if not archivo.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {archivo}")
    with archivo.open(encoding="utf-8") as manejador:
        config = yaml.safe_load(manejador)
    return config


def ruta_proyecto(relativa: str) -> Path:
    """Convierte una ruta relativa a la raíz (como las de `config.yaml`) en absoluta."""
    return RAIZ / relativa


def asegurar_directorios() -> None:
    """Crea las carpetas de salida si no existen.

    Las carpetas generadas pueden borrarse sin riesgo: este paso las recrea al
    inicio de cada ejecución.
    """
    carpetas = [
        RUTAS.procesados,
        RUTAS.resultados,
        RUTAS.figuras,
        RUTAS.docs,
        RUTAS.docs_figuras,
    ]
    for carpeta in carpetas:
        carpeta.mkdir(parents=True, exist_ok=True)


def guardar_json(datos: dict, nombre: str) -> Path:
    """Guarda un diccionario de resultados en `resultados/<nombre>.json`.

    Se usa UTF-8 sin escapar caracteres para que los textos en español sean
    legibles en el archivo, y sangría para que las diferencias en git sean claras.
    """
    RUTAS.resultados.mkdir(parents=True, exist_ok=True)
    destino = RUTAS.resultados / f"{nombre}.json"
    # newline="\n": el mismo archivo byte a byte en Windows y en Linux (CI).
    with destino.open("w", encoding="utf-8", newline="\n") as manejador:
        json.dump(datos, manejador, ensure_ascii=False, indent=2)
        manejador.write("\n")
    return destino


def leer_json(nombre: str) -> dict:
    """Lee `resultados/<nombre>.json`; falla con un mensaje claro si aún no existe."""
    origen = RUTAS.resultados / f"{nombre}.json"
    if not origen.exists():
        raise FileNotFoundError(
            f"No existe {origen}. Ejecute primero el módulo que lo genera (o run_all.py)."
        )
    with origen.open(encoding="utf-8") as manejador:
        return json.load(manejador)


CONFIG = cargar_config()
