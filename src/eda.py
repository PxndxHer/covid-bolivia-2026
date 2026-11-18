"""Análisis exploratorio de los datos municipales y de la comparación regional.

Produce `resultados/eda.json` con todos los números que luego usan las figuras,
la página web y las pruebas. También guarda en `datos/procesados/` la serie
diaria de los países comparados, que es demasiado larga para el JSON.
"""

import numpy as np
import pandas as pd
from scipy import stats

from src import datos as modulo_datos
from src.config import CONFIG, RUTAS, guardar_json

ARCHIVO_SERIES_COMPARACION = RUTAS.procesados / "comparacion_series.csv"


def redondear(valor: float, decimales: int = 4) -> float | None:
    """Redondea para el JSON y convierte NaN en None (JSON no admite NaN)."""
    if valor is None or pd.isna(valor):
        return None
    return round(float(valor), decimales)


# -----------------------------------------------------------------------------
# Datos municipales
# -----------------------------------------------------------------------------
def evolucion_nacional(municipios: pd.DataFrame) -> list[dict]:
    """Suma los acumulados de todos los municipios por semana.

    La letalidad nacional se calcula con los totales (fallecidos totales /
    confirmados totales) y no como promedio de letalidades municipales, para
    que cada caso pese lo mismo.
    """
    totales = municipios.groupby("semana")[modulo_datos.COLUMNAS_CONTEO].sum()
    totales["municipios_con_casos"] = municipios.groupby("semana")["cod_municipio"].nunique()
    totales["letalidad"] = totales["fallecidos"] / totales["confirmados"] * 100
    filas = []
    for semana, fila in totales.iterrows():
        filas.append(
            {
                "semana": int(semana),
                "confirmados": int(fila["confirmados"]),
                "activos": int(fila["activos"]),
                "fallecidos": int(fila["fallecidos"]),
                "recuperados": int(fila["recuperados"]),
                "municipios_con_casos": int(fila["municipios_con_casos"]),
                "letalidad": redondear(fila["letalidad"]),
            }
        )
    return filas


def semana_final(municipios: pd.DataFrame) -> pd.DataFrame:
    """Devuelve las filas de la última semana configurada (cierre del estudio)."""
    semana = CONFIG["municipios"]["semana_fin"]
    return municipios[municipios["semana"] == semana].copy()


def resumen_distribucion(valores: pd.Series) -> dict:
    """Resume una distribución asimétrica con mediana y cuartiles además de la media."""
    return {
        "n": int(valores.count()),
        "media": redondear(valores.mean()),
        "mediana": redondear(valores.median()),
        "p25": redondear(valores.quantile(0.25)),
        "p75": redondear(valores.quantile(0.75)),
        "maximo": redondear(valores.max()),
        "minimo": redondear(valores.min()),
    }


def distribucion_municipal(ultima: pd.DataFrame) -> dict:
    """Describe confirmados y letalidad por municipio en la semana final.

    La letalidad se resume solo con municipios que tienen casos (donde está
    definida). Se informa también la concentración en los 10 municipios con
    más casos, porque la distribución es muy desigual.
    """
    confirmados = ultima["confirmados"]
    letalidad = ultima["letalidad"].dropna()
    top10 = confirmados.nlargest(10).sum()
    return {
        "confirmados": resumen_distribucion(confirmados),
        "letalidad": resumen_distribucion(letalidad),
        "fallecidos": resumen_distribucion(ultima["fallecidos"]),
        "municipios_sin_fallecidos": int((ultima["fallecidos"] == 0).sum()),
        "porcentaje_casos_top10": redondear(top10 / confirmados.sum() * 100),
        "valores_confirmados": [int(v) for v in confirmados],
        "valores_letalidad": [redondear(v) for v in letalidad],
    }


def totales_departamento(ultima: pd.DataFrame) -> list[dict]:
    """Suma casos y fallecidos por departamento y calcula su letalidad agregada."""
    agrupado = ultima.groupby("departamento").agg(
        confirmados=("confirmados", "sum"),
        fallecidos=("fallecidos", "sum"),
        recuperados=("recuperados", "sum"),
        activos=("activos", "sum"),
        municipios=("cod_municipio", "nunique"),
    )
    agrupado["letalidad"] = agrupado["fallecidos"] / agrupado["confirmados"] * 100
    agrupado["porcentaje_casos"] = agrupado["confirmados"] / agrupado["confirmados"].sum() * 100
    agrupado = agrupado.sort_values("confirmados", ascending=False)
    filas = []
    for departamento, fila in agrupado.iterrows():
        filas.append(
            {
                "departamento": departamento,
                "confirmados": int(fila["confirmados"]),
                "fallecidos": int(fila["fallecidos"]),
                "recuperados": int(fila["recuperados"]),
                "activos": int(fila["activos"]),
                "municipios": int(fila["municipios"]),
                "letalidad": redondear(fila["letalidad"]),
                "porcentaje_casos": redondear(fila["porcentaje_casos"]),
            }
        )
    return filas


def prueba_kruskal(ultima: pd.DataFrame) -> dict:
    """Compara la letalidad municipal entre departamentos con Kruskal-Wallis.

    Decisiones:
    - Solo municipios con al menos `min_casos_boxplot` casos: con pocos casos
      una sola muerte cambia la letalidad en varios puntos porcentuales.
    - Solo departamentos con al menos `min_municipios_grupo` municipios: un
      grupo de un valor no aporta información sobre la dispersión interna.
    - Prueba no paramétrica porque la letalidad es asimétrica y los grupos son
      pequeños y de tamaño muy distinto.
    - Se informa épsilon² como tamaño del efecto, para no depender solo del p.
    """
    minimo_casos = CONFIG["municipios"]["min_casos_boxplot"]
    minimo_grupo = CONFIG["municipios"]["min_municipios_grupo"]
    alfa = CONFIG["eda"]["alfa"]
    muestra = ultima[ultima["confirmados"] >= minimo_casos]
    grupos = muestra.groupby("departamento")["letalidad"]
    incluidos = {nombre: serie for nombre, serie in grupos if len(serie) >= minimo_grupo}
    excluidos = {nombre: int(len(serie)) for nombre, serie in grupos if len(serie) < minimo_grupo}
    resultado = stats.kruskal(*incluidos.values())
    n_total = sum(len(serie) for serie in incluidos.values())
    k = len(incluidos)
    epsilon2 = (resultado.statistic - k + 1) / (n_total - k)
    detalle = []
    for nombre, serie in incluidos.items():
        detalle.append(
            {
                "departamento": nombre,
                "n": int(len(serie)),
                "mediana": redondear(serie.median()),
                "valores": [redondear(v) for v in serie],
            }
        )
    detalle = sorted(detalle, key=lambda fila: fila["mediana"], reverse=True)
    return {
        "minimo_casos": minimo_casos,
        "municipios": int(len(muestra)),
        "municipios_en_prueba": int(n_total),
        "grupos": k,
        "gl": k - 1,
        "H": redondear(resultado.statistic),
        "p": float(resultado.pvalue),
        "epsilon2": redondear(epsilon2),
        "significativo": bool(resultado.pvalue < alfa),
        "alfa": alfa,
        "excluidos": excluidos,
        "detalle": detalle,
    }


def correlaciones_spearman(ultima: pd.DataFrame) -> dict:
    """Calcula correlaciones de Spearman entre las variables municipales.

    Se usa Spearman (rangos) porque los conteos tienen colas muy largas: unos
    pocos municipios grandes dominarían un coeficiente de Pearson. Solo entran
    municipios con casos, donde la letalidad está definida.
    """
    variables = CONFIG["eda"]["variables_spearman"]
    muestra = ultima[ultima["confirmados"] > 0][variables]
    rho, valores_p = stats.spearmanr(muestra)
    matriz_rho = [[redondear(v) for v in fila] for fila in rho]
    matriz_p = [[float(v) for v in fila] for fila in valores_p]
    return {
        "variables": variables,
        "n": int(len(muestra)),
        "rho": matriz_rho,
        "p": matriz_p,
    }


# -----------------------------------------------------------------------------
# Comparación internacional
# -----------------------------------------------------------------------------
def paises_estudio() -> list[str]:
    """Devuelve el país foco seguido de los países de comparación."""
    return [CONFIG["pais"]["nombre"]] + CONFIG["pais"]["comparacion"]


def series_comparacion(owid: pd.DataFrame) -> pd.DataFrame:
    """Prepara las series diarias de los países comparados.

    La media móvil de 7 días elimina el patrón semanal de notificación (menos
    reportes en fines de semana) sin desplazar la tendencia más de 3 días.
    """
    ventana = CONFIG["eda"]["ventana_media_movil"]
    seleccion = owid[owid["location"].isin(paises_estudio())].copy()
    seleccion = seleccion.sort_values(["location", "date"])
    agrupado = seleccion.groupby("location")["new_cases_per_million"]
    seleccion["casos_millon_mm"] = agrupado.transform(
        lambda serie: serie.rolling(ventana, min_periods=ventana).mean()
    )
    columnas = [
        "location",
        "date",
        "new_cases_per_million",
        "casos_millon_mm",
        "total_deaths_per_million",
    ]
    return seleccion[columnas].reset_index(drop=True)


def ultimo_valor(serie: pd.Series) -> float:
    """Devuelve el último valor no nulo de una serie ordenada por fecha."""
    validos = serie.dropna()
    if validos.empty:
        return np.nan
    return validos.iloc[-1]


def indicadores_corte(owid: pd.DataFrame) -> list[dict]:
    """Calcula casos y muertes por millón y letalidad a la fecha de corte por país."""
    nombres = CONFIG["pais"]["nombres_es"]
    filas = []
    for pais in paises_estudio():
        datos_pais = owid[owid["location"] == pais].sort_values("date")
        casos = ultimo_valor(datos_pais["total_cases"])
        muertes = ultimo_valor(datos_pais["total_deaths"])
        filas.append(
            {
                "pais": pais,
                "nombre": nombres.get(pais, pais),
                "es_foco": pais == CONFIG["pais"]["nombre"],
                "casos_por_millon": redondear(ultimo_valor(datos_pais["total_cases_per_million"]), 1),
                "muertes_por_millon": redondear(ultimo_valor(datos_pais["total_deaths_per_million"]), 1),
                "letalidad": redondear(muertes / casos * 100),
                "casos": int(casos),
                "muertes": int(muertes),
            }
        )
    return filas


def posicion_foco(indicadores: list[dict], clave: str) -> int:
    """Devuelve el puesto del país foco (1 = valor más alto) en un indicador."""
    ordenados = sorted(indicadores, key=lambda fila: fila[clave], reverse=True)
    for puesto, fila in enumerate(ordenados, start=1):
        if fila["es_foco"]:
            return puesto
    raise ValueError("El país foco no está en los indicadores.")


def contar_dias_negativos(owid: pd.DataFrame) -> dict:
    """Cuenta, por país comparado, los días con casos nuevos negativos.

    Un valor negativo es una corrección retroactiva de la fuente. Se informa
    en vez de corregirse para no alterar los datos de OWID.
    """
    seleccion = owid[owid["location"].isin(paises_estudio())]
    negativos = seleccion[seleccion["new_cases"] < 0].groupby("location").size()
    return {pais: int(cantidad) for pais, cantidad in negativos.items()}


def comparacion_internacional(owid: pd.DataFrame) -> dict:
    """Reúne los indicadores al corte, los puestos del país foco y el pico de casos."""
    series = series_comparacion(owid)
    series.to_csv(ARCHIVO_SERIES_COMPARACION, index=False, encoding="utf-8")
    indicadores = indicadores_corte(owid)
    foco = series[series["location"] == CONFIG["pais"]["nombre"]]
    fila_pico = foco.loc[foco["casos_millon_mm"].idxmax()]
    return {
        "fecha_corte": CONFIG["fecha_corte"],
        "ventana_media_movil": CONFIG["eda"]["ventana_media_movil"],
        "indicadores": indicadores,
        "dias_negativos": contar_dias_negativos(owid),
        "puesto_foco": {
            "casos_por_millon": posicion_foco(indicadores, "casos_por_millon"),
            "muertes_por_millon": posicion_foco(indicadores, "muertes_por_millon"),
            "letalidad": posicion_foco(indicadores, "letalidad"),
            "total_paises": len(indicadores),
        },
        "pico_foco_mm": {
            "fecha": fila_pico["date"].strftime("%Y-%m-%d"),
            "casos_millon_mm": redondear(fila_pico["casos_millon_mm"], 1),
        },
    }


def ejecutar() -> dict:
    """Ejecuta todo el análisis exploratorio y guarda `resultados/eda.json`."""
    municipios = modulo_datos.cargar_municipios()
    owid = modulo_datos.cargar_owid()
    ultima = semana_final(municipios)
    resultados = {
        "semana_fin": CONFIG["municipios"]["semana_fin"],
        "nacional": evolucion_nacional(municipios),
        "distribucion": distribucion_municipal(ultima),
        "departamentos": totales_departamento(ultima),
        "kruskal": prueba_kruskal(ultima),
        "spearman": correlaciones_spearman(ultima),
        "internacional": comparacion_internacional(owid),
    }
    guardar_json(resultados, "eda")
    return resultados


if __name__ == "__main__":
    salida = ejecutar()
    print(f"Kruskal-Wallis: H = {salida['kruskal']['H']}, p = {salida['kruskal']['p']:.5f}")
