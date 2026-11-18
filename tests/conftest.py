"""Configuración común de las pruebas.

Agrega la raíz del proyecto al path (para importar `src`) y ofrece los JSON de
resultados como fixtures. Las pruebas de resultados requieren haber ejecutado
antes `python run_all.py`.
"""

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import CONFIG, RUTAS, leer_json  # noqa: E402

# Los valores de aceptación del proyecto original solo valen con la configuración por defecto.
CONFIGURACION_POR_DEFECTO = (
    CONFIG["pais"]["nombre"] == "Bolivia"
    and CONFIG["fecha_corte"] == "2020-09-19"
    and CONFIG["municipios"]["semana_fin"] == 30
    and CONFIG["municipios"]["semana_referencia"] == 27
    and CONFIG["municipios"]["umbral_letalidad"] == 3.7
    and CONFIG["semilla"] == 42
)

solo_configuracion_por_defecto = pytest.mark.skipif(
    not CONFIGURACION_POR_DEFECTO,
    reason="Los valores de aceptación corresponden a la configuración por defecto.",
)


def cargar_resultado(nombre: str) -> dict:
    """Lee un JSON de resultados o omite la prueba si todavía no se generó."""
    if not (RUTAS.resultados / f"{nombre}.json").exists():
        pytest.skip(f"Falta resultados/{nombre}.json: ejecute python run_all.py")
    return leer_json(nombre)


@pytest.fixture(scope="session")
def eda():
    return cargar_resultado("eda")


@pytest.fixture(scope="session")
def perfiles():
    return cargar_resultado("perfiles")


@pytest.fixture(scope="session")
def series():
    return cargar_resultado("series")


@pytest.fixture(scope="session")
def riesgo():
    return cargar_resultado("riesgo")


@pytest.fixture(scope="session")
def pagina():
    archivo = RUTAS.docs / "index.html"
    if not archivo.exists():
        pytest.skip("Falta docs/index.html: ejecute python run_all.py")
    return archivo.read_text(encoding="utf-8")
