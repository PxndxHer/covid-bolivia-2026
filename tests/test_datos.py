"""Pruebas de los controles de calidad de datos."""

import pandas as pd
import pytest

from src import datos
from tests.conftest import solo_configuracion_por_defecto


def tabla_valida() -> pd.DataFrame:
    """Dos filas municipales correctas, base para provocar cada falla."""
    return pd.DataFrame(
        {
            "cod_departamento": [1, 1],
            "departamento": ["A", "A"],
            "cod_municipio": [101, 102],
            "municipio": ["X", "Y"],
            "semana": [20, 20],
            "confirmados": [10, 5],
            "activos": [6, 5],
            "fallecidos": [1, 0],
            "recuperados": [3, 0],
        }
    )


def test_tabla_valida_pasa_todos_los_controles():
    tabla = tabla_valida()
    datos.verificar_nulos(tabla)
    datos.verificar_duplicados(tabla)
    datos.verificar_no_negativos(tabla)
    datos.verificar_identidad(tabla)


def test_detecta_duplicados_municipio_semana():
    tabla = tabla_valida()
    tabla.loc[1, "cod_municipio"] = 101
    with pytest.raises(datos.ErrorCalidadDatos, match="duplicadas"):
        datos.verificar_duplicados(tabla)


def test_detecta_nulos():
    tabla = tabla_valida()
    tabla.loc[0, "fallecidos"] = None
    with pytest.raises(datos.ErrorCalidadDatos, match="nulos"):
        datos.verificar_nulos(tabla)


def test_detecta_falla_de_identidad_contable():
    tabla = tabla_valida()
    tabla.loc[0, "confirmados"] = 11
    with pytest.raises(datos.ErrorCalidadDatos, match="confirmados = activos"):
        datos.verificar_identidad(tabla)


def test_detecta_conteos_negativos():
    tabla = tabla_valida()
    tabla.loc[1, "activos"] = -1
    with pytest.raises(datos.ErrorCalidadDatos, match="negativos"):
        datos.verificar_no_negativos(tabla)


def test_letalidad_indefinida_sin_casos():
    tabla = tabla_valida()
    tabla.loc[1, ["confirmados", "activos"]] = 0
    resultado = datos.agregar_letalidad(tabla)
    assert resultado.loc[0, "letalidad"] == pytest.approx(10.0)
    assert pd.isna(resultado.loc[1, "letalidad"])


def test_datos_reales_pasan_los_controles():
    municipios, informe = datos.limpiar_municipios()
    assert informe["duplicados"] == 0
    assert informe["fallas_identidad"] == 0
    assert not municipios.duplicated(["cod_municipio", "semana"]).any()
    suma = municipios["activos"] + municipios["fallecidos"] + municipios["recuperados"]
    assert (municipios["confirmados"] == suma).all()


def test_owid_excluye_agregados_y_respeta_fecha_de_corte():
    owid, _informe = datos.limpiar_owid()
    prefijo = datos.CONFIG["perfiles"]["prefijo_agregados"]
    assert not owid["iso_code"].str.startswith(prefijo).any()
    assert owid["date"].max() <= pd.Timestamp(datos.CONFIG["fecha_corte"])


@solo_configuracion_por_defecto
def test_totales_semana_final_coinciden_con_el_original():
    municipios, informe = datos.limpiar_municipios()
    final = municipios[municipios["semana"] == 30]
    assert final["confirmados"].sum() == 68281
    assert final["fallecidos"].sum() == 2535
    assert informe["filas"] == 1639
    assert informe["municipios"] == 256
