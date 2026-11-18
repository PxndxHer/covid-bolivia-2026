"""Modelo 2 · Pronóstico de casos semanales e índice de presión sanitaria (IPS).

1. Compara seis modelos de pronóstico a 4 semanas con validación de origen
   móvil y elige el mejor según la métrica de config.yaml.
2. Proyecta las 4 semanas siguientes al corte con intervalos empíricos.
3. Calcula el IPS (casos de 14 días por cama hospitalaria) para los países
   comparados y escenarios de camas adicionales para el país foco.
El resultado es `resultados/series.json`.
"""

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import Holt

from src import datos as modulo_datos
from src.config import CONFIG, guardar_json

PARAMETROS = CONFIG["series"]
ARCHIVO_IPS = modulo_datos.RUTAS.procesados / "ips_series.csv"


def redondear(valor: float, decimales: int = 4) -> float | None:
    """Redondea para el JSON; NaN e infinitos se guardan como None."""
    if valor is None or not np.isfinite(valor):
        return None
    return round(float(valor), decimales)


def fecha_texto(fecha: pd.Timestamp) -> str:
    """Fecha en formato ISO para el JSON."""
    return pd.Timestamp(fecha).strftime("%Y-%m-%d")


# -----------------------------------------------------------------------------
# Serie semanal
# -----------------------------------------------------------------------------
def serie_semanal(owid: pd.DataFrame, pais: str) -> pd.Series:
    """Suma los casos nuevos diarios por semana (semanas que cierran el día de config).

    Se trabaja por semana porque los datos diarios tienen un fuerte patrón de
    notificación (menos reportes en fines de semana) que no es epidemiológico.
    """
    diaria = owid[owid["location"] == pais].set_index("date")["new_cases"]
    regla = f"W-{PARAMETROS['dia_cierre_semana']}"
    semanal = diaria.resample(regla).sum()
    inicio = pd.Timestamp(PARAMETROS["fecha_inicio"])
    corte = pd.Timestamp(CONFIG["fecha_corte"])
    return semanal.loc[inicio:corte].astype(float)


# -----------------------------------------------------------------------------
# Modelos de pronóstico: cada uno recibe la serie de entrenamiento y el horizonte
# -----------------------------------------------------------------------------
def pronostico_ingenuo(entrenamiento: np.ndarray, horizonte: int) -> np.ndarray:
    """Repite el último valor observado. Es la referencia mínima que todo modelo debe superar."""
    return np.repeat(entrenamiento[-1], horizonte)


def pronostico_media_movil(entrenamiento: np.ndarray, horizonte: int) -> np.ndarray:
    """Repite la media de las últimas semanas; suaviza el ruido pero reacciona con retraso."""
    ventana = PARAMETROS["ventana_media_movil"]
    return np.repeat(entrenamiento[-ventana:].mean(), horizonte)


def extrapolar_polinomio(entrenamiento: np.ndarray, horizonte: int, grado: int, ventana: int) -> np.ndarray:
    """Ajusta un polinomio a las últimas `ventana` semanas y lo extiende `horizonte` semanas."""
    tiempo = np.arange(len(entrenamiento))
    coeficientes = np.polyfit(tiempo[-ventana:], entrenamiento[-ventana:], grado)
    futuro = np.arange(len(entrenamiento), len(entrenamiento) + horizonte)
    return np.polyval(coeficientes, futuro)


def pronostico_lineal(entrenamiento: np.ndarray, horizonte: int) -> np.ndarray:
    """Tendencia lineal de las últimas semanas: capta la dirección reciente de la curva."""
    return extrapolar_polinomio(entrenamiento, horizonte, 1, PARAMETROS["ventana_lineal"])


def pronostico_polinomico(entrenamiento: np.ndarray, horizonte: int) -> np.ndarray:
    """Polinomio de grado 2 sobre toda la serie: capta la curvatura de una ola, pero
    extrapola mal fuera del rango observado (se incluye como contraste)."""
    return extrapolar_polinomio(entrenamiento, horizonte, PARAMETROS["grado_polinomio"], len(entrenamiento))


def pronostico_holt(entrenamiento: np.ndarray, horizonte: int) -> np.ndarray:
    """Holt con tendencia amortiguada.

    La amortiguación hace que la tendencia se aplane con el horizonte, lo que
    evita proyectar crecimientos o caídas indefinidas cerca de un pico.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        modelo = Holt(entrenamiento, damped_trend=True, initialization_method=PARAMETROS["holt_inicializacion"])
        ajuste = modelo.fit()
    return ajuste.forecast(horizonte)


def ajustar_arima(entrenamiento: np.ndarray):
    """Elige ARIMA(p,1,q) sobre log1p(casos) por AIC y devuelve el mejor ajuste.

    El logaritmo estabiliza la varianza (los errores crecen con el nivel de la
    serie) y garantiza pronósticos positivos al deshacer la transformación.
    Las combinaciones que no convergen se descartan. El resultado se guarda en
    caché porque la validación y el registro de órdenes ajustan la misma serie.
    """
    return ajustar_arima_cache(tuple(entrenamiento))


@lru_cache(maxsize=None)
def ajustar_arima_cache(entrenamiento: tuple):
    """Implementación de `ajustar_arima` con caché (requiere una tupla, que es inmutable)."""
    transformada = np.log1p(np.array(entrenamiento))
    mejor = None
    for p in range(PARAMETROS["arima_p_max"] + 1):
        for q in range(PARAMETROS["arima_q_max"] + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    ajuste = ARIMA(transformada, order=(p, PARAMETROS["arima_d"], q)).fit()
                except (ValueError, np.linalg.LinAlgError):
                    continue
            if mejor is None or ajuste.aic < mejor.aic:
                mejor = ajuste
    return mejor


def pronostico_arima(entrenamiento: np.ndarray, horizonte: int) -> np.ndarray:
    """Pronóstico puntual de ARIMA devuelto a la escala original."""
    ajuste = ajustar_arima(entrenamiento)
    return np.expm1(ajuste.forecast(horizonte))


MODELOS = {
    "ingenuo": pronostico_ingenuo,
    "media_movil": pronostico_media_movil,
    "lineal": pronostico_lineal,
    "polinomica": pronostico_polinomico,
    "holt": pronostico_holt,
    "arima": pronostico_arima,
}


# -----------------------------------------------------------------------------
# Validación con origen móvil
# -----------------------------------------------------------------------------
def validacion_origen_movil(serie: pd.Series) -> pd.DataFrame:
    """Pronostica desde cada origen posible y registra cada error.

    En cada corte el modelo solo ve las semanas anteriores (como en la vida
    real) y pronostica las `horizonte` siguientes. El primer corte usa
    `primer_origen` semanas; el último deja exactamente `horizonte` semanas
    para comparar.
    """
    valores = serie.to_numpy()
    fechas = serie.index
    horizonte = PARAMETROS["horizonte"]
    registros = []
    for origen in range(PARAMETROS["primer_origen"], len(valores) - horizonte + 1):
        entrenamiento = valores[:origen]
        escala_mase = np.mean(np.abs(np.diff(entrenamiento)))
        for nombre, funcion in MODELOS.items():
            pronostico = funcion(entrenamiento, horizonte)
            for paso in range(horizonte):
                registros.append(
                    {
                        "modelo": nombre,
                        "origen": origen,
                        "h": paso + 1,
                        "fecha": fechas[origen + paso],
                        "real": valores[origen + paso],
                        "pronostico": float(pronostico[paso]),
                        "escala_mase": escala_mase,
                    }
                )
    tabla = pd.DataFrame(registros)
    tabla["error"] = tabla["real"] - tabla["pronostico"]
    return tabla


def calcular_metricas(grupo: pd.DataFrame, mae_ingenuo: float) -> dict:
    """MAE, RMSE, MAPE, R², MASE y MAE relativo al modelo ingenuo.

    - R² compara con predecir la media de los valores reales; puede ser
      negativo si el modelo es peor que esa constante.
    - MASE (Hyndman y Koehler, 2006) escala cada error por el error medio del
      pronóstico ingenuo de un paso dentro del entrenamiento; < 1 indica
      un error menor que el de un cambio semanal típico.
    - El MAE relativo divide directamente por el MAE del modelo ingenuo en las
      mismas semanas; < 1 significa que supera al ingenuo.
    """
    error = grupo["error"]
    real = grupo["real"]
    sse = np.sum(error**2)
    sst = np.sum((real - real.mean()) ** 2)
    mae = np.mean(np.abs(error))
    return {
        "MAE": redondear(mae, 1),
        "RMSE": redondear(np.sqrt(np.mean(error**2)), 1),
        "MAPE": redondear(np.mean(np.abs(error / real)) * 100, 2),
        "R2": redondear(1 - sse / sst, 3),
        "MASE": redondear(np.mean(np.abs(error) / grupo["escala_mase"]), 3),
        "MAE_relativo_ingenuo": redondear(mae / mae_ingenuo, 3),
        "n": int(len(grupo)),
    }


def tabla_metricas(validacion: pd.DataFrame) -> list[dict]:
    """Métricas de cada modelo, ordenadas por la métrica de selección."""
    referencia = PARAMETROS["modelo_referencia_mase"]
    mae_ingenuo = validacion.loc[validacion["modelo"] == referencia, "error"].abs().mean()
    nombres = PARAMETROS["nombres_modelos"]
    filas = []
    for modelo, grupo in validacion.groupby("modelo"):
        fila = {"modelo": modelo, "nombre": nombres[modelo]}
        fila.update(calcular_metricas(grupo, mae_ingenuo))
        filas.append(fila)
    metrica = PARAMETROS["metrica_seleccion"].upper()
    return sorted(filas, key=lambda fila: fila[metrica])


def metricas_por_horizonte(validacion: pd.DataFrame) -> dict:
    """RMSE de cada modelo para cada paso del horizonte (el error crece con h)."""
    resultado = {}
    for modelo, grupo in validacion.groupby("modelo"):
        rmse = grupo.groupby("h")["error"].apply(lambda e: np.sqrt(np.mean(e**2)))
        resultado[modelo] = [redondear(valor, 1) for valor in rmse]
    return resultado


# -----------------------------------------------------------------------------
# Proyección final e intervalos
# -----------------------------------------------------------------------------
def cuantiles_error(validacion: pd.DataFrame, modelo: str) -> dict:
    """Cuantiles empíricos de los errores de validación por horizonte.

    Los intervalos no suponen normalidad: se construyen con los errores que
    el modelo cometió realmente en la validación para cada paso h. Con 13
    cortes por paso son aproximados, y así se informa.
    """
    errores = validacion[validacion["modelo"] == modelo]
    cuantiles = {}
    for nivel in PARAMETROS["niveles_intervalo"]:
        alfa = (100 - nivel) / 100
        inferior = errores.groupby("h")["error"].quantile(alfa / 2)
        superior = errores.groupby("h")["error"].quantile(1 - alfa / 2)
        cuantiles[nivel] = (inferior.to_numpy(), superior.to_numpy())
    return cuantiles


def fechas_futuras(serie: pd.Series) -> pd.DatetimeIndex:
    """Fechas de cierre de las semanas a pronosticar después del corte."""
    regla = f"W-{PARAMETROS['dia_cierre_semana']}"
    inicio = serie.index[-1] + pd.Timedelta(days=1)
    return pd.date_range(inicio, periods=PARAMETROS["horizonte"], freq=regla)


def proyeccion_mejor_modelo(serie: pd.Series, validacion: pd.DataFrame, modelo: str) -> dict:
    """Ajusta el mejor modelo con toda la serie y añade intervalos empíricos.

    Los límites inferiores se truncan en cero porque no puede haber casos
    negativos.
    """
    horizonte = PARAMETROS["horizonte"]
    puntual = MODELOS[modelo](serie.to_numpy(), horizonte)
    cuantiles = cuantiles_error(validacion, modelo)
    fechas = fechas_futuras(serie)
    filas = []
    for paso in range(horizonte):
        fila = {"fecha": fecha_texto(fechas[paso]), "h": paso + 1, "pronostico": redondear(puntual[paso], 0)}
        for nivel, (inferior, superior) in cuantiles.items():
            fila[f"inferior_{nivel}"] = redondear(max(puntual[paso] + inferior[paso], 0), 0)
            fila[f"superior_{nivel}"] = redondear(puntual[paso] + superior[paso], 0)
        filas.append(fila)
    return {"modelo": modelo, "nombre": PARAMETROS["nombres_modelos"][modelo], "semanas": filas}


def proyeccion_arima(serie: pd.Series) -> dict:
    """Proyección ARIMA con sus intervalos teóricos, como comparación.

    Los intervalos se calculan en escala log1p y se transforman de vuelta,
    por lo que resultan asimétricos.
    """
    ajuste = ajustar_arima(serie.to_numpy())
    horizonte = PARAMETROS["horizonte"]
    prediccion = ajuste.get_forecast(horizonte)
    fechas = fechas_futuras(serie)
    media = np.expm1(prediccion.predicted_mean)
    filas = [{"fecha": fecha_texto(fechas[p]), "h": p + 1, "pronostico": redondear(media[p], 0)} for p in range(horizonte)]
    for nivel in PARAMETROS["niveles_intervalo"]:
        limites = np.expm1(prediccion.conf_int(alpha=(100 - nivel) / 100))
        for paso in range(horizonte):
            filas[paso][f"inferior_{nivel}"] = redondear(max(limites[paso, 0], 0), 0)
            filas[paso][f"superior_{nivel}"] = redondear(limites[paso, 1], 0)
    orden = ajuste.model.order
    return {"orden": [int(v) for v in orden], "aic": redondear(ajuste.aic, 2), "semanas": filas}


def ordenes_arima_validacion(serie: pd.Series) -> list[dict]:
    """Orden ARIMA elegido en cada corte de la validación (muestra la inestabilidad del AIC)."""
    valores = serie.to_numpy()
    horizonte = PARAMETROS["horizonte"]
    ordenes = []
    for origen in range(PARAMETROS["primer_origen"], len(valores) - horizonte + 1):
        ajuste = ajustar_arima(valores[:origen])
        ordenes.append({"origen": origen, "orden": [int(v) for v in ajuste.model.order]})
    return ordenes


# -----------------------------------------------------------------------------
# Índice de presión sanitaria
# -----------------------------------------------------------------------------
def camas_totales(owid_pais: pd.DataFrame) -> float:
    """Camas hospitalarias totales = camas por mil × población / 1000.

    Son camas totales (no de terapia intensiva): OWID no publica camas UTI
    para la mayoría de los países de la región.
    """
    camas_por_mil = owid_pais["hospital_beds_per_thousand"].dropna().iloc[-1]
    poblacion = owid_pais["population"].dropna().iloc[-1]
    return camas_por_mil * poblacion / 1000


def serie_ips(owid: pd.DataFrame, pais: str) -> tuple[pd.DataFrame, float]:
    """Calcula el IPS diario de un país: casos nuevos de 14 días / camas totales.

    Se usan 14 días porque es el periodo típico de enfermedad y potencial
    hospitalización; así el numerador aproxima los casos activos recientes.
    """
    datos_pais = owid[owid["location"] == pais].sort_values("date")
    camas = camas_totales(datos_pais)
    ventana = PARAMETROS["ips_ventana_dias"]
    casos_ventana = datos_pais["new_cases"].fillna(0).rolling(ventana, min_periods=ventana).sum()
    tabla = pd.DataFrame(
        {
            "location": pais,
            "date": datos_pais["date"].to_numpy(),
            "casos_14d": casos_ventana.to_numpy(),
            "ips": (casos_ventana / camas).to_numpy(),
        }
    )
    return tabla, camas


def resumen_ips(owid: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """IPS pico (valor y fecha) e IPS al corte de cada país comparado."""
    paises = [CONFIG["pais"]["nombre"]] + CONFIG["pais"]["comparacion"]
    nombres = CONFIG["pais"]["nombres_es"]
    filas = []
    series = []
    for pais in paises:
        tabla, camas = serie_ips(owid, pais)
        series.append(tabla)
        pico = tabla.loc[tabla["ips"].idxmax()]
        filas.append(
            {
                "pais": pais,
                "nombre": nombres.get(pais, pais),
                "es_foco": pais == CONFIG["pais"]["nombre"],
                "camas": int(round(camas)),
                "ips_pico": redondear(pico["ips"], 3),
                "fecha_pico": fecha_texto(pico["date"]),
                "casos_14d_pico": int(pico["casos_14d"]),
                "ips_corte": redondear(tabla["ips"].iloc[-1], 3),
            }
        )
    return filas, pd.concat(series, ignore_index=True)


def escenarios_camas(resumen: list[dict]) -> dict:
    """IPS pico del país foco con más camas y camas necesarias para igualar la mediana regional.

    La mediana (y no la media) de los otros países evita que un solo país con
    un pico extremo fije la meta.
    """
    foco = next(fila for fila in resumen if fila["es_foco"])
    otros = [fila["ips_pico"] for fila in resumen if not fila["es_foco"]]
    mediana_otros = float(np.median(otros))
    escenarios = []
    for incremento in PARAMETROS["escenarios_camas"]:
        camas = foco["camas"] * (1 + incremento / 100)
        escenarios.append(
            {
                "incremento_pct": incremento,
                "camas": int(round(camas)),
                "ips_pico": redondear(foco["casos_14d_pico"] / camas, 3),
            }
        )
    camas_necesarias = foco["casos_14d_pico"] / mediana_otros
    return {
        "escenarios": escenarios,
        "mediana_ips_pico_otros": redondear(mediana_otros, 3),
        "camas_necesarias": int(round(camas_necesarias)),
        "camas_adicionales": int(round(camas_necesarias - foco["camas"])),
        "incremento_necesario_pct": redondear((camas_necesarias / foco["camas"] - 1) * 100, 1),
    }


# -----------------------------------------------------------------------------
# Ejecución
# -----------------------------------------------------------------------------
def describir_serie(serie: pd.Series) -> dict:
    """Valores de la serie semanal y su pico, para figuras y página web."""
    return {
        "fechas": [fecha_texto(f) for f in serie.index],
        "casos": [int(v) for v in serie],
        "n_semanas": int(len(serie)),
        "fecha_pico": fecha_texto(serie.idxmax()),
        "casos_pico": int(serie.max()),
        "total": int(serie.sum()),
    }


def ejecutar() -> dict:
    """Ejecuta el Modelo 2 completo y guarda `resultados/series.json`."""
    owid = modulo_datos.cargar_owid()
    serie = serie_semanal(owid, CONFIG["pais"]["nombre"])
    validacion = validacion_origen_movil(serie)
    metricas = tabla_metricas(validacion)
    mejor = metricas[0]["modelo"]
    resumen, series_ips = resumen_ips(owid)
    series_ips.to_csv(ARCHIVO_IPS, index=False, encoding="utf-8")
    validacion_mejor = validacion[validacion["modelo"].isin([mejor, PARAMETROS["modelo_referencia_mase"]])]
    resultados = {
        "serie": describir_serie(serie),
        "horizonte": PARAMETROS["horizonte"],
        "cortes": int(validacion["origen"].nunique()),
        "primer_origen": PARAMETROS["primer_origen"],
        "metrica_seleccion": PARAMETROS["metrica_seleccion"],
        "metricas": metricas,
        "rmse_por_horizonte": metricas_por_horizonte(validacion),
        "mejor_modelo": mejor,
        "validacion_mejor": [
            {"modelo": f["modelo"], "origen": int(f["origen"]), "h": int(f["h"]), "fecha": fecha_texto(f["fecha"]),
             "real": int(f["real"]), "pronostico": redondear(f["pronostico"], 0)}
            for f in validacion_mejor.to_dict("records")
        ],
        "proyeccion": proyeccion_mejor_modelo(serie, validacion, mejor),
        "proyeccion_arima": proyeccion_arima(serie),
        "ordenes_arima": ordenes_arima_validacion(serie),
        "ips": {
            "ventana_dias": PARAMETROS["ips_ventana_dias"],
            "tipo_camas": "totales",
            "paises": resumen,
            **escenarios_camas(resumen),
        },
    }
    guardar_json(resultados, "series")
    return resultados


if __name__ == "__main__":
    salida = ejecutar()
    for fila in salida["metricas"]:
        print(fila)
    foco = next(f for f in salida["ips"]["paises"] if f["es_foco"])
    print("IPS foco:", foco)
