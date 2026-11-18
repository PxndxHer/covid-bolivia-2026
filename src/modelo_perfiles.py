"""Modelo 1 · Perfiles de países con k-means.

Agrupa países según 8 indicadores de impacto (mortalidad, casos, letalidad),
respuesta (pruebas, rigurosidad) y capacidad estructural (camas, IDH, edad).
Se trabaja con dos muestras: la región del país foco (8 países) y todos los
países con datos completos. El resultado es `resultados/perfiles.json`.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src import datos as modulo_datos
from src.config import CONFIG, guardar_json

PARAMETROS = CONFIG["perfiles"]


def redondear(valor: float, decimales: int = 4) -> float:
    """Redondea un número para el JSON."""
    return round(float(valor), decimales)


# -----------------------------------------------------------------------------
# Construcción de la matriz de variables
# -----------------------------------------------------------------------------
def ultimo_no_nulo(serie: pd.Series) -> float:
    """Último valor no nulo de una serie ordenada por fecha (NaN si no hay ninguno)."""
    validos = serie.dropna()
    if validos.empty:
        return np.nan
    return validos.iloc[-1]


def valores_fecha_corte(owid: pd.DataFrame) -> pd.DataFrame:
    """Filas de cada país en la fecha de corte o, si no se exige, su último dato disponible.

    Exigir la fecha exacta garantiza que todos los países se comparen el mismo
    día; la alternativa admite más países a costa de mezclar fechas distintas.
    """
    if PARAMETROS["exigir_dato_en_fecha_corte"]:
        corte = pd.Timestamp(CONFIG["fecha_corte"])
        return owid[owid["date"] == corte].set_index("location")
    agrupado = owid.sort_values("date").groupby("location")
    return agrupado.agg(ultimo_no_nulo)


def agregar_variable(owid: pd.DataFrame, en_corte: pd.DataFrame, definicion: dict) -> pd.Series:
    """Resume una variable de OWID por país según su tipo de agregación.

    - `ultimo`: valor a la fecha de corte (acumulados y datos estructurales).
    - `ultimo_disponible`: último dato no nulo; las pruebas se informan con
      días o semanas de rezago, y exigir la fecha exacta dejaría fuera a la
      mayoría de los países.
    - `media`: promedio del periodo; la rigurosidad cambia mucho en el tiempo y
      su promedio resume la intensidad total de la respuesta.
    - `calculada`: letalidad = fallecidos / casos × 100 a la fecha de corte.
    """
    columna = definicion["columna"]
    tipo = definicion["agregacion"]
    ordenado = owid.sort_values("date").groupby("location")
    if tipo == "ultimo":
        return en_corte[columna]
    if tipo == "ultimo_disponible":
        return ordenado[columna].agg(ultimo_no_nulo)
    if tipo == "media":
        return ordenado[columna].mean()
    if tipo == "calculada" and columna == "letalidad":
        return en_corte["total_deaths"] / en_corte["total_cases"] * 100
    raise ValueError(f"Agregación desconocida para {columna}: {tipo}")


def matriz_variables(owid: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla país × variable con todos los países (con o sin datos completos)."""
    en_corte = valores_fecha_corte(owid)
    columnas = {}
    for definicion in PARAMETROS["variables"]:
        columnas[definicion["columna"]] = agregar_variable(owid, en_corte, definicion)
    return pd.DataFrame(columnas)


def muestra_mundial(matriz: pd.DataFrame) -> pd.DataFrame:
    """Países con las 8 variables completas.

    No se imputan faltantes: imputar el IDH o las pruebas de un país con la
    media mundial inventaría justo los rasgos que el modelo intenta comparar.
    """
    return matriz.dropna().sort_index()


def muestra_regional(matriz: pd.DataFrame) -> pd.DataFrame:
    """País foco y países de comparación; falla si alguno tiene datos incompletos."""
    paises = [CONFIG["pais"]["nombre"]] + CONFIG["pais"]["comparacion"]
    regional = matriz.loc[paises]
    incompletos = regional.index[regional.isna().any(axis=1)].tolist()
    if incompletos:
        raise ValueError(f"Países de comparación con variables faltantes: {incompletos}")
    return regional


# -----------------------------------------------------------------------------
# Agrupamiento y validación
# -----------------------------------------------------------------------------
def estandarizar(tabla: pd.DataFrame) -> np.ndarray:
    """Estandariza a media 0 y DE 1.

    Sin este paso, los casos por millón (decenas de miles) dominarían las
    distancias y el IDH (0 a 1) no tendría ningún peso.
    """
    return StandardScaler().fit_transform(tabla)


def ajustar_kmeans(matriz: np.ndarray, k: int, semilla: int, n_inicios: int) -> np.ndarray:
    """Ajusta k-means y devuelve las etiquetas.

    Varios inicios aleatorios reducen el riesgo de quedar en un mínimo local
    de la inercia; se conserva la mejor solución.
    """
    modelo = KMeans(n_clusters=k, n_init=n_inicios, random_state=semilla)
    return modelo.fit_predict(matriz)


def tabla_silueta(matriz: np.ndarray, k_min: int, k_max: int) -> list[dict]:
    """Calcula la silueta media para cada k del rango.

    La silueta compara la cohesión interna de cada grupo con su separación del
    grupo vecino; a diferencia de la inercia, no mejora siempre al subir k.
    """
    filas = []
    for k in range(k_min, k_max + 1):
        etiquetas = ajustar_kmeans(matriz, k, CONFIG["semilla"], PARAMETROS["n_inicios"])
        silueta = silhouette_score(matriz, etiquetas)
        tamanos = np.bincount(etiquetas).tolist()
        filas.append({"k": k, "silueta": redondear(silueta), "tamanos": tamanos})
    return filas


def elegir_k(siluetas: list[dict], k_minimo: int) -> int:
    """Elige el k con mayor silueta entre los que respetan el mínimo aceptado.

    En la muestra mundial se exige k ≥ 3 porque con k = 2 k-means tiende a
    separar solo países ricos de pobres, una partición trivial.
    """
    candidatos = [fila for fila in siluetas if fila["k"] >= k_minimo]
    mejor = max(candidatos, key=lambda fila: fila["silueta"])
    return mejor["k"]


def ari_contra_ward(matriz: np.ndarray, etiquetas: np.ndarray, k: int) -> float:
    """ARI entre k-means y un agrupamiento jerárquico de Ward con el mismo k.

    Ward minimiza la misma varianza intragrupo, pero con un algoritmo
    determinista y aglomerativo; si ambos coinciden, los grupos no dependen
    del método elegido.
    """
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(matriz)
    return adjusted_rand_score(etiquetas, ward)


def ari_semillas(matriz: np.ndarray, etiquetas: np.ndarray, k: int, n_inicios: int) -> dict:
    """Resume el ARI entre la solución elegida y corridas con semillas distintas.

    Las semillas se derivan de la semilla global para que sean reproducibles.
    """
    generador = np.random.default_rng(CONFIG["semilla"])
    semillas = generador.integers(0, 1_000_000, PARAMETROS["n_semillas_estabilidad"])
    valores = []
    for semilla in semillas:
        alternativa = ajustar_kmeans(matriz, k, int(semilla), n_inicios)
        valores.append(adjusted_rand_score(etiquetas, alternativa))
    valores = np.array(valores)
    return {
        "corridas": int(len(valores)),
        "inicios_por_corrida": n_inicios,
        "media": redondear(valores.mean()),
        "minimo": redondear(valores.min()),
        "proporcion_identica": redondear(np.mean(valores > 0.999)),
    }


def estabilidad_semillas(matriz: np.ndarray, etiquetas: np.ndarray, k: int) -> dict:
    """Mide la estabilidad frente a la semilla de dos maneras complementarias.

    - Con los mismos inicios que el modelo (50): ¿otra semilla reproduce la
      misma partición? Es la reproducibilidad del procedimiento completo.
    - Con un solo inicio: ¿qué tan fácil es llegar a esta partición desde un
      punto al azar? Un valor bajo indica que la estructura es frágil y que
      los 50 inicios son necesarios.
    """
    return {
        "procedimiento": ari_semillas(matriz, etiquetas, k, PARAMETROS["n_inicios"]),
        "inicio_unico": ari_semillas(matriz, etiquetas, k, 1),
    }


def coasignacion_foco(matriz: np.ndarray, tabla: pd.DataFrame, k: int) -> dict:
    """Proporción de corridas (semillas distintas, 50 inicios) en que cada país comparte grupo con el foco.

    Permite distinguir el núcleo estable del grupo del país foco de los países
    que entran o salen según la semilla. Solo se guardan los países que lo
    acompañan al menos una vez.
    """
    foco = CONFIG["pais"]["nombre"]
    posicion = tabla.index.get_loc(foco)
    generador = np.random.default_rng(CONFIG["semilla"])
    semillas = generador.integers(0, 1_000_000, PARAMETROS["n_semillas_estabilidad"])
    conteo = np.zeros(len(tabla))
    for semilla in semillas:
        etiquetas = ajustar_kmeans(matriz, k, int(semilla), PARAMETROS["n_inicios"])
        conteo += etiquetas == etiquetas[posicion]
    proporcion = pd.Series(conteo / len(semillas), index=tabla.index).drop(foco)
    proporcion = proporcion[proporcion > 0].sort_values(ascending=False)
    return {pais: redondear(valor, 2) for pais, valor in proporcion.items()}


def ordenar_grupos(etiquetas: np.ndarray, tabla: pd.DataFrame) -> np.ndarray:
    """Renumera los grupos de 1 en adelante según su mortalidad media (de mayor a menor).

    Las etiquetas de k-means son arbitrarias; ordenarlas por una variable fija
    hace que el "Grupo 1" signifique lo mismo en cada ejecución.
    """
    mortalidad = pd.Series(tabla["total_deaths_per_million"].to_numpy()).groupby(etiquetas).mean()
    orden = mortalidad.sort_values(ascending=False).index
    mapa = {anterior: nuevo for nuevo, anterior in enumerate(orden, start=1)}
    return np.array([mapa[etiqueta] for etiqueta in etiquetas])


# -----------------------------------------------------------------------------
# Interpretación
# -----------------------------------------------------------------------------
def nombrar_grupo(perfil_z: pd.Series) -> str:
    """Nombra un grupo con sus rasgos más marcados (mayor |z| medio).

    Solo cuentan los rasgos que se alejan de la media más que `umbral_z_nombre`
    desviaciones estándar; así el nombre describe lo que distingue al grupo y
    se genera igual para cualquier país o periodo.
    """
    umbral = PARAMETROS["umbral_z_nombre"]
    descriptores = PARAMETROS["descriptores"]
    marcados = perfil_z[perfil_z.abs() >= umbral]
    marcados = marcados.reindex(marcados.abs().sort_values(ascending=False).index)
    marcados = marcados.head(PARAMETROS["n_rasgos_nombre"])
    if marcados.empty:
        return PARAMETROS["nombre_sin_rasgos"]
    rasgos = []
    for columna, valor in marcados.items():
        sentido = "alto" if valor > 0 else "bajo"
        rasgos.append(descriptores[columna][sentido])
    texto = " y ".join(rasgos)
    return texto[0].upper() + texto[1:]


def describir_grupos(tabla: pd.DataFrame, matriz: np.ndarray, etiquetas: np.ndarray) -> list[dict]:
    """Perfil medio (en unidades originales y estandarizado), tamaño, nombre y miembros de cada grupo."""
    columnas = tabla.columns
    estandarizada = pd.DataFrame(matriz, index=tabla.index, columns=columnas)
    nombres_es = CONFIG["pais"]["nombres_es"]
    foco = CONFIG["pais"]["nombre"]
    grupos = []
    for grupo in sorted(set(etiquetas)):
        mascara = etiquetas == grupo
        perfil_z = estandarizada[mascara].mean()
        perfil = tabla[mascara].mean()
        miembros = tabla.index[mascara].tolist()
        grupos.append(
            {
                "grupo": int(grupo),
                "nombre": nombrar_grupo(perfil_z),
                "tamano": int(mascara.sum()),
                "contiene_foco": foco in miembros,
                "miembros": miembros,
                "miembros_es": sorted(nombres_es.get(pais, pais) for pais in miembros),
                "perfil": {columna: redondear(perfil[columna]) for columna in columnas},
                "perfil_z": {columna: redondear(perfil_z[columna]) for columna in columnas},
            }
        )
    return grupos


def pca_dos_componentes(matriz: np.ndarray, tabla: pd.DataFrame) -> dict:
    """Proyecta los países en 2 componentes principales e informa las cargas.

    Es solo una ayuda visual: los grupos se forman en las 8 dimensiones, y la
    varianza explicada indica cuánto de esa estructura se ve en el plano.
    """
    pca = PCA(n_components=PARAMETROS["n_componentes_pca"], random_state=CONFIG["semilla"])
    coordenadas = pca.fit_transform(matriz)
    cargas = {}
    for indice, columna in enumerate(tabla.columns):
        cargas[columna] = [redondear(pca.components_[c][indice]) for c in range(pca.n_components_)]
    return {
        "varianza_explicada": [redondear(v) for v in pca.explained_variance_ratio_],
        "cargas": cargas,
        "coordenadas": {
            pais: [redondear(v) for v in fila] for pais, fila in zip(tabla.index, coordenadas)
        },
    }


def vecinos_foco(matriz: np.ndarray, tabla: pd.DataFrame) -> list[dict]:
    """Países más parecidos al foco por distancia euclidiana en el espacio estandarizado."""
    foco = CONFIG["pais"]["nombre"]
    posicion = tabla.index.get_loc(foco)
    distancias = np.linalg.norm(matriz - matriz[posicion], axis=1)
    orden = np.argsort(distancias)
    nombres_es = CONFIG["pais"]["nombres_es"]
    vecinos = []
    for indice in orden[1 : PARAMETROS["n_vecinos"] + 1]:
        pais = tabla.index[indice]
        vecinos.append(
            {"pais": pais, "nombre": nombres_es.get(pais, pais), "distancia": redondear(distancias[indice])}
        )
    return vecinos


def valores_foco(tabla: pd.DataFrame, matriz: np.ndarray) -> dict:
    """Valores originales y estandarizados del país foco dentro de la muestra."""
    foco = CONFIG["pais"]["nombre"]
    posicion = tabla.index.get_loc(foco)
    return {
        "valores": {columna: redondear(tabla.iloc[posicion][columna]) for columna in tabla.columns},
        "z": {columna: redondear(matriz[posicion][i]) for i, columna in enumerate(tabla.columns)},
    }


def analizar_muestra(tabla: pd.DataFrame, clave: str) -> dict:
    """Aplica todo el procedimiento a una muestra: elegir k, ajustar, validar e interpretar."""
    configuracion = PARAMETROS["muestras"][clave]
    matriz = estandarizar(tabla)
    siluetas = tabla_silueta(matriz, configuracion["k_min"], configuracion["k_max"])
    k = elegir_k(siluetas, configuracion["k_minimo_aceptado"])
    etiquetas = ajustar_kmeans(matriz, k, CONFIG["semilla"], PARAMETROS["n_inicios"])
    etiquetas = ordenar_grupos(etiquetas, tabla)
    silueta_k = next(fila["silueta"] for fila in siluetas if fila["k"] == k)
    grupos = describir_grupos(tabla, matriz, etiquetas)
    grupo_foco = next(grupo for grupo in grupos if grupo["contiene_foco"])
    return {
        "etiqueta": configuracion["etiqueta"],
        "n_paises": int(len(tabla)),
        "paises": tabla.index.tolist(),
        "siluetas": siluetas,
        "k": k,
        "k_minimo_aceptado": configuracion["k_minimo_aceptado"],
        "silueta": silueta_k,
        "ari_ward": redondear(ari_contra_ward(matriz, etiquetas, k)),
        "estabilidad": estabilidad_semillas(matriz, etiquetas, k),
        "coasignacion_foco": coasignacion_foco(matriz, tabla, k),
        "asignacion": {pais: int(grupo) for pais, grupo in zip(tabla.index, etiquetas)},
        "grupos": grupos,
        "grupo_foco": grupo_foco["grupo"],
        "pca": pca_dos_componentes(matriz, tabla),
        "vecinos_foco": vecinos_foco(matriz, tabla),
        "foco": valores_foco(tabla, matriz),
    }


def sin_registro_corte(owid: pd.DataFrame, matriz: pd.DataFrame) -> list[str]:
    """Países con las 8 variables disponibles salvo por no tener fila en la fecha de corte.

    Sirve para documentar qué países excluye `exigir_dato_en_fecha_corte`.
    """
    corte = pd.Timestamp(CONFIG["fecha_corte"])
    con_fila = set(owid.loc[owid["date"] == corte, "location"])
    candidatos = matriz.drop(columns=["letalidad"]).copy()
    alternativa = owid.sort_values("date").groupby("location").agg(ultimo_no_nulo)
    for definicion in PARAMETROS["variables"]:
        if definicion["agregacion"] == "ultimo":
            candidatos[definicion["columna"]] = alternativa[definicion["columna"]]
    completos = candidatos.dropna().index
    return sorted(pais for pais in completos if pais not in con_fila)


def paises_excluidos(matriz: pd.DataFrame, mundial: pd.DataFrame, owid: pd.DataFrame) -> dict:
    """Cuenta cuántos países quedaron fuera por cada variable faltante (para documentarlo)."""
    faltantes = matriz.isna().sum()
    sin_fila_corte = sin_registro_corte(owid, matriz) if PARAMETROS["exigir_dato_en_fecha_corte"] else []
    return {
        "total_paises": int(len(matriz)),
        "incluidos": int(len(mundial)),
        "excluidos": int(len(matriz) - len(mundial)),
        "faltantes_por_variable": {columna: int(valor) for columna, valor in faltantes.items()},
        "sin_registro_en_fecha_corte": sin_fila_corte,
    }


def ejecutar() -> dict:
    """Ejecuta el Modelo 1 en ambas muestras y guarda `resultados/perfiles.json`."""
    owid = modulo_datos.cargar_owid()
    matriz = matriz_variables(owid)
    mundial = muestra_mundial(matriz)
    regional = muestra_regional(matriz)
    resultados = {
        "variables": [
            {"columna": v["columna"], "nombre": v["nombre"], "agregacion": v["agregacion"]}
            for v in PARAMETROS["variables"]
        ],
        "exigir_dato_en_fecha_corte": PARAMETROS["exigir_dato_en_fecha_corte"],
        "cobertura": paises_excluidos(matriz, mundial, owid),
        "regional": analizar_muestra(regional, "regional"),
        "mundial": analizar_muestra(mundial, "mundial"),
    }
    guardar_json(resultados, "perfiles")
    return resultados


if __name__ == "__main__":
    salida = ejecutar()
    for muestra in ["regional", "mundial"]:
        r = salida[muestra]
        print(f"{muestra}: n={r['n_paises']} k={r['k']} silueta={r['silueta']} ARI Ward={r['ari_ward']}")
