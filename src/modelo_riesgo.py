"""Modelo 3 · Riesgo de letalidad alta por municipio.

Pregunta: con lo que se sabía en la semana de referencia (27), ¿se puede
anticipar qué municipios tendrán letalidad alta en la semana final (30)?
Se comparan tres clasificadores con dos líneas base: la clase mayoritaria y la
persistencia (suponer que la letalidad de la semana 27 se mantiene). Si la
persistencia iguala o supera a los modelos, se presenta como resultado
principal. El resultado es `resultados/riesgo.json`.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import datos as modulo_datos
from src.config import CONFIG, guardar_json

PARAMETROS = CONFIG["riesgo"]
MUNICIPIOS = CONFIG["municipios"]
METRICAS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
COLUMNA_DEPARTAMENTO = "departamento"


def redondear(valor: float, decimales: int = 4) -> float | None:
    """Redondea para el JSON; NaN se guarda como None."""
    if valor is None or not np.isfinite(valor):
        return None
    return round(float(valor), decimales)


# -----------------------------------------------------------------------------
# Muestra y variables
# -----------------------------------------------------------------------------
def construir_muestra(municipios: pd.DataFrame) -> pd.DataFrame:
    """Municipios con al menos `min_casos_modelo` casos en la semana final.

    El mínimo de casos evita que la letalidad objetivo sea puro ruido: con
    5 casos, una muerte equivale a 20 % de letalidad. Se exige además que el
    municipio exista en la semana de referencia (si no, no habría variables).
    """
    fin = municipios[municipios["semana"] == MUNICIPIOS["semana_fin"]].set_index("cod_municipio")
    referencia = municipios[municipios["semana"] == MUNICIPIOS["semana_referencia"]].set_index("cod_municipio")
    seleccion = fin[fin["confirmados"] >= MUNICIPIOS["min_casos_modelo"]]
    seleccion = seleccion[seleccion.index.isin(referencia.index)]
    muestra = pd.DataFrame(
        {
            "municipio": seleccion["municipio"],
            "departamento": seleccion["departamento"],
            "confirmados_fin": seleccion["confirmados"],
            "letalidad_fin": seleccion["letalidad"],
            "confirmados_ref": referencia.loc[seleccion.index, "confirmados"],
            "fallecidos_ref": referencia.loc[seleccion.index, "fallecidos"],
            "letalidad_ref": referencia.loc[seleccion.index, "letalidad"],
        }
    )
    muestra["objetivo"] = (muestra["letalidad_fin"] >= MUNICIPIOS["umbral_letalidad"]).astype(int)
    return muestra


def variables_conjunto(muestra: pd.DataFrame, conjunto: str) -> pd.DataFrame:
    """Matriz de variables de un conjunto.

    - A: tamaño del brote (log10 de confirmados, porque los conteos abarcan
      tres órdenes de magnitud) y departamento como variables indicadoras.
    - B: A más la letalidad y los fallecidos de la semana de referencia, para
      medir cuánto aporta conocer la letalidad previa.
    """
    tamano = np.log10(muestra["confirmados_ref"]).rename("log10_confirmados")
    departamentos = pd.get_dummies(muestra["departamento"], prefix=COLUMNA_DEPARTAMENTO, dtype=float)
    matriz = pd.concat([tamano, departamentos], axis=1)
    if conjunto == "B":
        matriz["letalidad_ref"] = muestra["letalidad_ref"]
        matriz["fallecidos_ref"] = muestra["fallecidos_ref"]
    return matriz


def crear_modelos() -> dict:
    """Clasificadores con los hiperparámetros de config.yaml.

    Los árboles se limitan en profundidad y tamaño de hoja porque con 94
    municipios un árbol sin límites memorizaría la muestra. La regresión
    logística se estandariza dentro del pipeline, para que la media y la DE se
    calculen solo con los datos de entrenamiento de cada pliegue.
    """
    semilla = CONFIG["semilla"]
    return {
        "arbol": DecisionTreeClassifier(random_state=semilla, **PARAMETROS["arbol"]),
        "bosque": RandomForestClassifier(random_state=semilla, **PARAMETROS["bosque"]),
        "logistica": make_pipeline(StandardScaler(), LogisticRegression(random_state=semilla, **PARAMETROS["logistica"])),
        "mayoritaria": DummyClassifier(strategy="most_frequent"),
    }


def crear_validacion(repeticiones: int) -> RepeatedStratifiedKFold:
    """Validación cruzada estratificada repetida.

    La estratificación mantiene la proporción de municipios con letalidad alta
    en cada pliegue; repetir con particiones distintas reduce la dependencia
    del resultado de una partición afortunada.
    """
    return RepeatedStratifiedKFold(
        n_splits=PARAMETROS["validacion"]["n_pliegues"],
        n_repeats=repeticiones,
        random_state=CONFIG["semilla"],
    )


# -----------------------------------------------------------------------------
# Evaluación
# -----------------------------------------------------------------------------
def evaluar_predicciones(real: np.ndarray, clase: np.ndarray, puntaje: np.ndarray) -> dict:
    """Calcula las cinco métricas de un pliegue.

    `zero_division=0` hace que un modelo que nunca predice la clase alta tenga
    precisión 0 (en vez de un error), como corresponde a la clase mayoritaria.
    """
    return {
        "accuracy": accuracy_score(real, clase),
        "precision": precision_score(real, clase, zero_division=0),
        "recall": recall_score(real, clase, zero_division=0),
        "f1": f1_score(real, clase, zero_division=0),
        "roc_auc": roc_auc_score(real, puntaje),
    }


def evaluar_pliegues(muestra: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Evalúa todos los modelos, conjuntos y la persistencia en los mismos pliegues.

    Usar exactamente las mismas particiones para todos permite compararlos
    pliegue a pliegue. La persistencia no se entrena: clasifica como alto todo
    municipio cuya letalidad en la semana de referencia ya superaba el umbral,
    y usa esa letalidad como puntaje para el AUC.
    """
    objetivo = muestra["objetivo"].to_numpy()
    umbral = MUNICIPIOS["umbral_letalidad"]
    conjuntos = {nombre: variables_conjunto(muestra, nombre) for nombre in ["A", "B"]}
    modelos = crear_modelos()
    validacion = crear_validacion(PARAMETROS["validacion"]["n_repeticiones"])
    n_pliegues = PARAMETROS["validacion"]["n_pliegues"]
    registros = []
    matrices = {}
    for indice, (entrenamiento, prueba) in enumerate(validacion.split(conjuntos["A"], objetivo)):
        repeticion = indice // n_pliegues
        real = objetivo[prueba]
        for conjunto, matriz in conjuntos.items():
            for nombre, modelo in modelos.items():
                ajustado = clone(modelo).fit(matriz.iloc[entrenamiento], objetivo[entrenamiento])
                clase = ajustado.predict(matriz.iloc[prueba])
                puntaje = ajustado.predict_proba(matriz.iloc[prueba])[:, 1]
                registrar(registros, matrices, nombre, conjunto, indice, repeticion, real, clase, puntaje)
        letalidad = muestra["letalidad_ref"].to_numpy()[prueba]
        clase_persistencia = (letalidad >= umbral).astype(int)
        registrar(registros, matrices, "persistencia", "-", indice, repeticion, real, clase_persistencia, letalidad)
    return pd.DataFrame(registros), matrices


def registrar(registros: list, matrices: dict, modelo: str, conjunto: str, pliegue: int,
              repeticion: int, real: np.ndarray, clase: np.ndarray, puntaje: np.ndarray) -> None:
    """Guarda las métricas del pliegue y suma su matriz de confusión a la de su repetición."""
    fila = {"modelo": modelo, "conjunto": conjunto, "pliegue": pliegue, "repeticion": repeticion}
    fila.update(evaluar_predicciones(real, clase, puntaje))
    registros.append(fila)
    clave = (modelo, conjunto, repeticion)
    matriz = confusion_matrix(real, clase, labels=[0, 1])
    matrices[clave] = matrices.get(clave, np.zeros((2, 2))) + matriz


def intervalo_nadeau_bengio(valores: np.ndarray, proporcion_prueba: float) -> tuple[float, float]:
    """Intervalo de confianza con la corrección de Nadeau y Bengio (2003).

    En la validación cruzada repetida los pliegues comparten datos de
    entrenamiento, así que sus resultados no son independientes y un intervalo
    t ingenuo sería demasiado estrecho. La corrección infla la varianza por
    el factor (1/J + n_prueba/n_entrenamiento), con J = número de pliegues.
    """
    nivel = PARAMETROS["validacion"]["nivel_confianza"]
    cantidad = len(valores)
    media = np.mean(valores)
    varianza = np.var(valores, ddof=1)
    proporcion = proporcion_prueba / (1 - proporcion_prueba)
    error_estandar = np.sqrt((1 / cantidad + proporcion) * varianza)
    critico = stats.t.ppf((1 + nivel) / 2, df=cantidad - 1)
    return media - critico * error_estandar, media + critico * error_estandar


def resumir_metricas(pliegues: pd.DataFrame) -> list[dict]:
    """Media, desviación e IC 95 % de cada métrica para cada modelo y conjunto."""
    proporcion_prueba = 1 / PARAMETROS["validacion"]["n_pliegues"]
    nombres = PARAMETROS["nombres_modelos"]
    filas = []
    for (modelo, conjunto), grupo in pliegues.groupby(["modelo", "conjunto"], sort=False):
        fila = {"modelo": modelo, "conjunto": conjunto, "nombre": nombres[modelo]}
        for metrica in METRICAS:
            valores = grupo[metrica].to_numpy()
            inferior, superior = intervalo_nadeau_bengio(valores, proporcion_prueba)
            fila[metrica] = {
                "media": redondear(np.mean(valores)),
                "de": redondear(np.std(valores, ddof=1)),
                "ic_inf": redondear(max(inferior, 0)),
                "ic_sup": redondear(min(superior, 1)),
            }
        filas.append(fila)
    return filas


def diferencia_con_persistencia(pliegues: pd.DataFrame, modelo: str, conjunto: str) -> dict:
    """Diferencia pliegue a pliegue entre un modelo y la persistencia, con IC corregido.

    Comparar en los mismos pliegues elimina la variación debida a la partición.
    Si el intervalo incluye el cero, el modelo no supera a la persistencia.
    """
    proporcion_prueba = 1 / PARAMETROS["validacion"]["n_pliegues"]
    elegido = pliegues[(pliegues["modelo"] == modelo) & (pliegues["conjunto"] == conjunto)].set_index("pliegue")
    persistencia = pliegues[pliegues["modelo"] == "persistencia"].set_index("pliegue")
    resultado = {}
    for metrica in METRICAS:
        diferencias = (elegido[metrica] - persistencia[metrica]).to_numpy()
        inferior, superior = intervalo_nadeau_bengio(diferencias, proporcion_prueba)
        resultado[metrica] = {
            "media": redondear(np.mean(diferencias)),
            "ic_inf": redondear(inferior),
            "ic_sup": redondear(superior),
            "proporcion_pliegues_modelo_mejor": redondear(np.mean(diferencias > 0), 3),
        }
    return resultado


def matrices_confusion_medias(matrices: dict) -> dict:
    """Promedia, entre repeticiones, las matrices de confusión sumadas por repetición.

    En cada repetición cada municipio está en prueba exactamente una vez, así
    que la suma de sus 5 pliegues cubre toda la muestra (94 municipios) y la
    media entre repeticiones se lee en número de municipios.
    """
    agrupadas = {}
    for (modelo, conjunto, _repeticion), matriz in matrices.items():
        agrupadas.setdefault((modelo, conjunto), []).append(matriz)
    resultado = {}
    for (modelo, conjunto), lista in agrupadas.items():
        media = np.mean(lista, axis=0)
        clave = f"{modelo}_{conjunto}" if conjunto != "-" else modelo
        resultado[clave] = {
            "verdaderos_negativos": redondear(media[0, 0], 2),
            "falsos_positivos": redondear(media[0, 1], 2),
            "falsos_negativos": redondear(media[1, 0], 2),
            "verdaderos_positivos": redondear(media[1, 1], 2),
            "total": redondear(media.sum(), 2),
        }
    return resultado


# -----------------------------------------------------------------------------
# Importancia por permutación
# -----------------------------------------------------------------------------
def grupos_variables(matriz: pd.DataFrame) -> dict:
    """Agrupa las columnas por variable original.

    Las columnas indicadoras de departamento se permutan juntas: permutar una
    sola dejaría filas con dos departamentos o ninguno, una combinación
    imposible que exageraría su importancia.
    """
    grupos = {}
    for columna in matriz.columns:
        nombre = COLUMNA_DEPARTAMENTO if columna.startswith(f"{COLUMNA_DEPARTAMENTO}_") else columna
        grupos.setdefault(nombre, []).append(columna)
    return grupos


def caida_auc(modelo, prueba_x: pd.DataFrame, prueba_y: np.ndarray, columnas: list, base: float,
              generador: np.random.Generator) -> list[float]:
    """Caída del AUC al permutar conjuntamente un grupo de columnas en el pliegue de prueba."""
    caidas = []
    for _ in range(PARAMETROS["permutacion"]["n_repeticiones"]):
        permutada = prueba_x.copy()
        orden = generador.permutation(len(permutada))
        permutada[columnas] = prueba_x[columnas].to_numpy()[orden]
        puntaje = modelo.predict_proba(permutada)[:, 1]
        caidas.append(base - roc_auc_score(prueba_y, puntaje))
    return caidas


def importancia_permutacion(muestra: pd.DataFrame) -> list[dict]:
    """Importancia por permutación (caída del AUC) calculada en pliegues de prueba.

    Medirla en datos no vistos evita premiar variables que el modelo solo
    usó para memorizar el entrenamiento. Se usan menos repeticiones que en la
    evaluación porque cada pliegue requiere cientos de predicciones.
    """
    configuracion = PARAMETROS["permutacion"]
    matriz = variables_conjunto(muestra, configuracion["conjunto"])
    objetivo = muestra["objetivo"].to_numpy()
    modelo_base = crear_modelos()[configuracion["modelo"]]
    grupos = grupos_variables(matriz)
    generador = np.random.default_rng(CONFIG["semilla"])
    caidas = {nombre: [] for nombre in grupos}
    validacion = crear_validacion(configuracion["n_repeticiones_cv"])
    for entrenamiento, prueba in validacion.split(matriz, objetivo):
        ajustado = clone(modelo_base).fit(matriz.iloc[entrenamiento], objetivo[entrenamiento])
        prueba_x = matriz.iloc[prueba]
        prueba_y = objetivo[prueba]
        base = roc_auc_score(prueba_y, ajustado.predict_proba(prueba_x)[:, 1])
        for nombre, columnas in grupos.items():
            caidas[nombre].extend(caida_auc(ajustado, prueba_x, prueba_y, columnas, base, generador))
    nombres = PARAMETROS["nombres_variables"]
    filas = []
    for nombre, valores in caidas.items():
        valores = np.array(valores)
        filas.append(
            {
                "variable": nombre,
                "nombre": nombres.get(nombre, nombre),
                "media": redondear(valores.mean()),
                "de": redondear(valores.std(ddof=1)),
                "p05": redondear(np.percentile(valores, 5)),
                "p95": redondear(np.percentile(valores, 95)),
            }
        )
    return sorted(filas, key=lambda fila: fila["media"], reverse=True)


# -----------------------------------------------------------------------------
# Ejecución
# -----------------------------------------------------------------------------
def mejor_modelo(resumen: list[dict]) -> dict:
    """Modelo entrenado (no línea base) con mayor valor medio en la métrica principal."""
    metrica = PARAMETROS["metrica_principal"]
    candidatos = [fila for fila in resumen if fila["modelo"] not in ("persistencia", "mayoritaria")]
    return max(candidatos, key=lambda fila: fila[metrica]["media"])


def decidir_resultado_principal(resumen: list[dict], diferencia: dict) -> dict:
    """Decide si la persistencia es el resultado principal.

    Regla: la persistencia es principal si su media en la métrica principal es
    mayor o igual que la del mejor modelo, o si el IC de la diferencia incluye
    el cero (el modelo no la supera de forma demostrable).
    """
    metrica = PARAMETROS["metrica_principal"]
    persistencia = next(fila for fila in resumen if fila["modelo"] == "persistencia")
    mejor = mejor_modelo(resumen)
    supera_o_iguala = persistencia[metrica]["media"] >= mejor[metrica]["media"]
    ic_incluye_cero = diferencia[metrica]["ic_inf"] <= 0 <= diferencia[metrica]["ic_sup"]
    return {
        "metrica": metrica,
        "persistencia_principal": bool(supera_o_iguala or ic_incluye_cero),
        "persistencia_supera_o_iguala": bool(supera_o_iguala),
        "ic_diferencia_incluye_cero": bool(ic_incluye_cero),
    }


def describir_muestra(muestra: pd.DataFrame) -> dict:
    """Tamaño de la muestra, clases y casos de persistencia observados."""
    umbral = MUNICIPIOS["umbral_letalidad"]
    alto_ref = muestra["letalidad_ref"] >= umbral
    alto_fin = muestra["objetivo"] == 1
    return {
        "municipios": int(len(muestra)),
        "altos": int(alto_fin.sum()),
        "bajos": int((~alto_fin).sum()),
        "umbral_letalidad": umbral,
        "semana_referencia": MUNICIPIOS["semana_referencia"],
        "semana_fin": MUNICIPIOS["semana_fin"],
        "min_casos": MUNICIPIOS["min_casos_modelo"],
        "alto_en_ambas": int((alto_ref & alto_fin).sum()),
        "pasan_a_alto": int((~alto_ref & alto_fin).sum()),
        "pasan_a_bajo": int((alto_ref & ~alto_fin).sum()),
        "bajo_en_ambas": int((~alto_ref & ~alto_fin).sum()),
        "por_departamento": {
            departamento: {"municipios": int(len(g)), "altos": int(g["objetivo"].sum())}
            for departamento, g in muestra.groupby("departamento")
        },
    }


def ejecutar() -> dict:
    """Ejecuta el Modelo 3 completo y guarda `resultados/riesgo.json`."""
    municipios = modulo_datos.cargar_municipios()
    muestra = construir_muestra(municipios)
    pliegues, matrices = evaluar_pliegues(muestra)
    resumen = resumir_metricas(pliegues)
    mejor = mejor_modelo(resumen)
    diferencia = diferencia_con_persistencia(pliegues, mejor["modelo"], mejor["conjunto"])
    validacion = PARAMETROS["validacion"]
    resultados = {
        "muestra": describir_muestra(muestra),
        "validacion": {
            "pliegues": validacion["n_pliegues"],
            "repeticiones": validacion["n_repeticiones"],
            "nivel_confianza": validacion["nivel_confianza"],
            "metodo_ic": "t con corrección de Nadeau y Bengio (2003)",
        },
        "metricas": resumen,
        "mejor_modelo": {"modelo": mejor["modelo"], "conjunto": mejor["conjunto"], "nombre": mejor["nombre"]},
        "diferencia_mejor_vs_persistencia": diferencia,
        "decision": decidir_resultado_principal(resumen, diferencia),
        "matrices_confusion": matrices_confusion_medias(matrices),
        "importancia": importancia_permutacion(muestra),
        "importancia_modelo": {
            "modelo": PARAMETROS["permutacion"]["modelo"],
            "conjunto": PARAMETROS["permutacion"]["conjunto"],
            "repeticiones_cv": PARAMETROS["permutacion"]["n_repeticiones_cv"],
            "permutaciones": PARAMETROS["permutacion"]["n_repeticiones"],
        },
    }
    guardar_json(resultados, "riesgo")
    return resultados


if __name__ == "__main__":
    salida = ejecutar()
    for fila in salida["metricas"]:
        print(fila["modelo"], fila["conjunto"], "F1", fila["f1"]["media"], "AUC", fila["roc_auc"]["media"])
    print(salida["decision"])
