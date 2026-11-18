"""Carga, limpieza y controles de calidad de los datos crudos.

Los archivos de `datos/crudos/` nunca se modifican. Este módulo los lee,
verifica su consistencia y escribe versiones limpias en `datos/procesados/`.
Si un control crítico falla, el pipeline se detiene con un mensaje que dice
exactamente qué filas fallaron, en vez de producir resultados silenciosamente
incorrectos.
"""

import warnings

import numpy as np
import pandas as pd

from src.config import CONFIG, RUTAS, asegurar_directorios, guardar_json, ruta_proyecto

ARCHIVO_MUNICIPIOS_LIMPIO = RUTAS.procesados / "municipios.csv"
ARCHIVO_OWID_LIMPIO = RUTAS.procesados / "owid_paises.csv"

COLUMNAS_CONTEO = ["confirmados", "activos", "fallecidos", "recuperados"]

# Número máximo de filas problemáticas que se muestran en un mensaje de error.
MAX_FILAS_MENSAJE = 10


class ErrorCalidadDatos(Exception):
    """Error que se lanza cuando un control de calidad crítico no se cumple."""


# -----------------------------------------------------------------------------
# Datos municipales
# -----------------------------------------------------------------------------
def leer_municipios_crudos() -> pd.DataFrame:
    """Lee el CSV municipal y renombra sus columnas por posición.

    Se renombra por posición (según `columnas_municipios` en config.yaml) porque
    los encabezados originales usan mayúsculas, abreviaturas y símbolos como
    "RECUPERADOS(*)", que pueden cambiar entre versiones del archivo.
    """
    archivo = ruta_proyecto(CONFIG["archivos"]["municipios"])
    if not archivo.exists():
        raise FileNotFoundError(f"No se encontró el archivo municipal: {archivo}")
    datos = pd.read_csv(archivo, encoding="utf-8")
    nombres = CONFIG["columnas_municipios"]
    if datos.shape[1] != len(nombres):
        raise ErrorCalidadDatos(
            f"El archivo municipal tiene {datos.shape[1]} columnas; se esperaban {len(nombres)}."
        )
    datos.columns = nombres
    return datos


def normalizar_textos(datos: pd.DataFrame) -> pd.DataFrame:
    """Quita espacios sobrantes en los nombres de departamento y municipio."""
    datos = datos.copy()
    for columna in ["departamento", "municipio"]:
        datos[columna] = datos[columna].astype(str).str.strip()
    return datos


def unificar_nombres_municipio(datos: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Asigna a cada código de municipio un único nombre: el de su semana más reciente.

    Algunos municipios aparecen escritos de forma distinta según la semana. El
    código es la clave confiable; el nombre más reciente se considera el
    corregido. Devuelve también cuántos códigos tenían más de un nombre.
    """
    nombres_por_codigo = datos.groupby("cod_municipio")["municipio"].nunique()
    codigos_variantes = int((nombres_por_codigo > 1).sum())
    ultimo = datos.sort_values("semana").groupby("cod_municipio")["municipio"].last()
    datos = datos.copy()
    datos["municipio"] = datos["cod_municipio"].map(ultimo)
    return datos, codigos_variantes


def formatear_filas(filas: pd.DataFrame) -> str:
    """Convierte las primeras filas problemáticas en texto para un mensaje de error."""
    return filas.head(MAX_FILAS_MENSAJE).to_string(index=False)


def verificar_nulos(datos: pd.DataFrame) -> None:
    """Falla si hay valores faltantes: no se imputan conteos epidemiológicos."""
    nulos = datos.isna().sum()
    nulos = nulos[nulos > 0]
    if not nulos.empty:
        raise ErrorCalidadDatos(f"Hay valores nulos por columna:\n{nulos.to_string()}")


def verificar_duplicados(datos: pd.DataFrame) -> None:
    """Falla si un municipio aparece más de una vez en la misma semana.

    Un duplicado duplicaría los casos al sumar por departamento o por país.
    """
    clave = ["cod_municipio", "semana"]
    duplicados = datos[datos.duplicated(clave, keep=False)]
    if not duplicados.empty:
        raise ErrorCalidadDatos(
            f"Hay {len(duplicados)} filas duplicadas por (municipio, semana):\n"
            f"{formatear_filas(duplicados)}"
        )


def verificar_no_negativos(datos: pd.DataFrame) -> None:
    """Falla si algún conteo es negativo, lo que es imposible en datos acumulados."""
    negativos = datos[(datos[COLUMNAS_CONTEO] < 0).any(axis=1)]
    if not negativos.empty:
        raise ErrorCalidadDatos(
            f"Hay {len(negativos)} filas con conteos negativos:\n{formatear_filas(negativos)}"
        )


def verificar_identidad(datos: pd.DataFrame) -> None:
    """Falla si confirmados ≠ activos + fallecidos + recuperados.

    Esta identidad contable es la mejor señal disponible de que una fila se
    transcribió correctamente; si no se cumple, la letalidad no es confiable.
    """
    suma = datos["activos"] + datos["fallecidos"] + datos["recuperados"]
    fallas = datos[datos["confirmados"] != suma]
    if not fallas.empty:
        raise ErrorCalidadDatos(
            f"{len(fallas)} filas no cumplen confirmados = activos + fallecidos + recuperados:\n"
            f"{formatear_filas(fallas)}"
        )


def verificar_rango_semanas(datos: pd.DataFrame) -> list[int]:
    """Falla si hay semanas fuera del rango configurado; devuelve las semanas faltantes.

    Que falte una semana dentro del rango no detiene el análisis (los modelos
    usan semanas puntuales), pero se informa para que las figuras no unan con
    una línea dos puntos no consecutivos.
    """
    inicio = CONFIG["municipios"]["semana_inicio"]
    fin = CONFIG["municipios"]["semana_fin"]
    presentes = set(datos["semana"].unique())
    fuera = sorted(semana for semana in presentes if semana < inicio or semana > fin)
    if fuera:
        raise ErrorCalidadDatos(f"Hay semanas fuera del rango {inicio}–{fin}: {fuera}")
    faltantes = [semana for semana in range(inicio, fin + 1) if semana not in presentes]
    return faltantes


def contar_descensos_acumulados(datos: pd.DataFrame) -> dict:
    """Cuenta las veces que un acumulado baja de una semana a la siguiente.

    En datos acumulados un descenso indica una corrección retroactiva de la
    fuente. No se corrige ni se descarta (no hay criterio objetivo para elegir
    cuál semana es la correcta); se informa como advertencia.
    """
    ordenados = datos.sort_values(["cod_municipio", "semana"])
    diferencias = ordenados.groupby("cod_municipio")[["confirmados", "fallecidos"]].diff()
    descensos = {
        "confirmados": int((diferencias["confirmados"] < 0).sum()),
        "fallecidos": int((diferencias["fallecidos"] < 0).sum()),
    }
    return descensos


def agregar_letalidad(datos: pd.DataFrame) -> pd.DataFrame:
    """Agrega la letalidad (%) = fallecidos / confirmados × 100.

    Con cero confirmados la letalidad queda indefinida (NaN) en vez de 0, para
    no presentar como "sin muertes" a un municipio sin casos.
    """
    datos = datos.copy()
    confirmados = datos["confirmados"].replace(0, np.nan)
    datos["letalidad"] = datos["fallecidos"] / confirmados * 100
    return datos


def limpiar_municipios() -> tuple[pd.DataFrame, dict]:
    """Ejecuta lectura, normalización y todos los controles sobre los datos municipales."""
    datos = leer_municipios_crudos()
    filas_crudas = len(datos)
    datos = normalizar_textos(datos)
    verificar_nulos(datos)
    verificar_duplicados(datos)
    verificar_no_negativos(datos)
    verificar_identidad(datos)
    semanas_faltantes = verificar_rango_semanas(datos)
    datos, codigos_variantes = unificar_nombres_municipio(datos)
    descensos = contar_descensos_acumulados(datos)
    datos = agregar_letalidad(datos)
    datos = datos.sort_values(["semana", "cod_departamento", "cod_municipio"])
    datos = datos.reset_index(drop=True)
    informe = {
        "filas": filas_crudas,
        "municipios": int(datos["cod_municipio"].nunique()),
        "departamentos": int(datos["departamento"].nunique()),
        "semanas_presentes": sorted(int(s) for s in datos["semana"].unique()),
        "semanas_faltantes": semanas_faltantes,
        "duplicados": 0,
        "nulos": 0,
        "fallas_identidad": 0,
        "codigos_con_nombres_distintos": codigos_variantes,
        "descensos_acumulados": descensos,
    }
    return datos, informe


# -----------------------------------------------------------------------------
# Our World in Data
# -----------------------------------------------------------------------------
def leer_owid_crudo() -> pd.DataFrame:
    """Lee la base de OWID conservando solo las columnas que usa el estudio."""
    archivo = ruta_proyecto(CONFIG["archivos"]["owid"])
    if not archivo.exists():
        raise FileNotFoundError(f"No se encontró el archivo de OWID: {archivo}")
    columnas = CONFIG["columnas_owid"]
    datos = pd.read_csv(archivo, usecols=columnas, parse_dates=["date"])
    return datos


def filtrar_owid(datos: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Aplica la fecha de corte y excluye agregados regionales y filas sin país.

    Los agregados (`OWID_WRL`, continentes, etc.) sumarían a los países que ya
    contienen; las filas sin `iso_code` (como "International") no son países.
    Devuelve también cuántas filas se descartaron por estos motivos.
    """
    corte = pd.Timestamp(CONFIG["fecha_corte"])
    prefijo = CONFIG["perfiles"]["prefijo_agregados"]
    en_fecha = datos[datos["date"] <= corte]
    tiene_iso = en_fecha["iso_code"].notna()
    es_agregado = en_fecha["iso_code"].fillna("").str.startswith(prefijo)
    filtrado = en_fecha[tiene_iso & ~es_agregado]
    descartadas = len(en_fecha) - len(filtrado)
    return filtrado.reset_index(drop=True), descartadas


def verificar_owid(datos: pd.DataFrame) -> None:
    """Falla si faltan el país foco o algún país de comparación, o si hay fechas duplicadas."""
    paises = [CONFIG["pais"]["nombre"]] + CONFIG["pais"]["comparacion"]
    ausentes = [pais for pais in paises if pais not in set(datos["location"])]
    if ausentes:
        raise ErrorCalidadDatos(f"Países ausentes en OWID: {ausentes}")
    duplicados = datos[datos.duplicated(["iso_code", "date"], keep=False)]
    if not duplicados.empty:
        raise ErrorCalidadDatos(
            f"Hay {len(duplicados)} filas duplicadas por (país, fecha) en OWID:\n"
            f"{formatear_filas(duplicados)}"
        )


def limpiar_owid() -> tuple[pd.DataFrame, dict]:
    """Lee, filtra y verifica la base de OWID."""
    crudo = leer_owid_crudo()
    datos, descartadas = filtrar_owid(crudo)
    verificar_owid(datos)
    informe = {
        "filas_crudas": int(len(crudo)),
        "filas_descartadas_agregados_o_sin_iso": int(descartadas),
        "filas": int(len(datos)),
        "paises": int(datos["iso_code"].nunique()),
        "fecha_min": datos["date"].min().strftime("%Y-%m-%d"),
        "fecha_max": datos["date"].max().strftime("%Y-%m-%d"),
    }
    return datos, informe


# -----------------------------------------------------------------------------
# Lectura de datos procesados (para los demás módulos)
# -----------------------------------------------------------------------------
def cargar_municipios() -> pd.DataFrame:
    """Lee los datos municipales limpios; los genera si aún no existen."""
    if not ARCHIVO_MUNICIPIOS_LIMPIO.exists():
        ejecutar()
    return pd.read_csv(ARCHIVO_MUNICIPIOS_LIMPIO, encoding="utf-8")


def cargar_owid() -> pd.DataFrame:
    """Lee los datos de OWID limpios; los genera si aún no existen."""
    if not ARCHIVO_OWID_LIMPIO.exists():
        ejecutar()
    return pd.read_csv(ARCHIVO_OWID_LIMPIO, parse_dates=["date"])


def ejecutar() -> dict:
    """Limpia ambas fuentes, guarda los CSV procesados y el informe de calidad."""
    asegurar_directorios()
    municipios, informe_municipios = limpiar_municipios()
    owid, informe_owid = limpiar_owid()
    municipios.to_csv(ARCHIVO_MUNICIPIOS_LIMPIO, index=False, encoding="utf-8")
    owid.to_csv(ARCHIVO_OWID_LIMPIO, index=False, encoding="utf-8")
    if informe_municipios["semanas_faltantes"]:
        warnings.warn(
            f"Semanas sin datos municipales: {informe_municipios['semanas_faltantes']}",
            stacklevel=2,
        )
    informe = {"municipios": informe_municipios, "owid": informe_owid}
    guardar_json(informe, "calidad")
    return informe


if __name__ == "__main__":
    resultado = ejecutar()
    print(resultado)
