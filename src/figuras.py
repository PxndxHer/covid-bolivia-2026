"""Estilo común de los gráficos y funciones de dibujo.

Las figuras se dibujan a partir de los JSON de `resultados/` y de los CSV de
`datos/procesados/`; ningún valor mostrado en un gráfico se escribe a mano.
Convenciones (de config.yaml): fuente disponible en Windows, números con coma
decimal, meses en español, país foco siempre en rojo y el resto en gris o azul.
"""

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: permite generar figuras en CI y en consola

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from src.config import CONFIG, RUTAS, leer_json

ESTILO = CONFIG["figuras"]
COLOR_FOCO = ESTILO["color_foco"]
COLOR_AZUL = ESTILO["color_secundario"]
COLOR_GRIS = ESTILO["color_neutro"]
COLOR_TEXTO = "#333333"
COLOR_TEXTO_SUAVE = "#666666"
COLOR_GRILLA = "#E6E6E6"


# -----------------------------------------------------------------------------
# Formato de números y fechas en español
# -----------------------------------------------------------------------------
def formatear_numero(valor: float, decimales: int = 0) -> str:
    """Escribe un número con separador de miles y coma decimal (p. ej. 68 281 o 0,79).

    Se formatea primero al estilo inglés y luego se intercambian los
    separadores, lo que no depende de la configuración regional del sistema.
    """
    valor = round(float(valor), decimales)
    if valor == 0:
        valor = 0.0  # evita "-0" cuando un valor negativo pequeño se redondea a cero
    texto = f"{valor:,.{decimales}f}"
    texto = texto.replace(",", "\u0000")
    texto = texto.replace(".", ESTILO["separador_decimal"])
    texto = texto.replace("\u0000", ESTILO["separador_miles"])
    return texto


def formatear_p(valor_p: float, decimales: int = 4) -> str:
    """Escribe un valor p; si es menor que la precisión mostrada, usa "<"."""
    minimo = 10 ** (-decimales)
    if valor_p < minimo:
        return f"p < {formatear_numero(minimo, decimales)}"
    return f"p = {formatear_numero(valor_p, decimales)}"


def decimales_necesarios(valor: float, maximo: int = 2) -> int:
    """Devuelve los decimales mínimos para escribir un valor sin redondearlo (hasta `maximo`)."""
    for decimales in range(maximo + 1):
        if abs(round(valor, decimales) - valor) < 1e-9:
            return decimales
    return maximo


def formateador_numeros(decimales: int | None = 0) -> FuncFormatter:
    """Devuelve un formateador de ejes con números en español.

    Con `decimales=None` cada marca usa los decimales que necesita (2,5 se
    escribe "2,5" y 5 se escribe "5"), para no redondear marcas intermedias.
    """
    if decimales is None:
        return FuncFormatter(lambda valor, _posicion: formatear_numero(valor, decimales_necesarios(valor)))
    return FuncFormatter(lambda valor, _posicion: formatear_numero(valor, decimales))


def formateador_meses() -> FuncFormatter:
    """Devuelve un formateador de ejes de fecha con los meses en español (de config)."""
    meses = ESTILO["meses"]
    return FuncFormatter(lambda valor, _posicion: meses[mdates.num2date(valor).month - 1])


def nombre_departamento(codigo: str) -> str:
    """Traduce el nombre crudo de un departamento a su forma con tildes."""
    return CONFIG["municipios"]["nombres_departamento"].get(codigo, codigo.title())


def color_pais(pais: str) -> str:
    """Rojo para el país foco y gris para los demás."""
    if pais == CONFIG["pais"]["nombre"]:
        return COLOR_FOCO
    return COLOR_GRIS


# -----------------------------------------------------------------------------
# Estilo y guardado
# -----------------------------------------------------------------------------
def aplicar_estilo() -> None:
    """Configura matplotlib con el estilo común de todas las figuras.

    Ejes y grilla discretos para que los datos sean lo más visible; sin bordes
    superior y derecho; una sola fuente (DejaVu Sans viene con matplotlib, así
    que existe en cualquier Windows, Linux o CI).
    """
    plt.rcParams.update(
        {
            "font.family": ESTILO["fuente"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.labelcolor": COLOR_TEXTO,
            "axes.edgecolor": "#BBBBBB",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": COLOR_GRILLA,
            "grid.linewidth": 0.8,
            "xtick.color": COLOR_TEXTO_SUAVE,
            "ytick.color": COLOR_TEXTO_SUAVE,
            "legend.frameon": False,
            "figure.dpi": ESTILO["dpi"],
            "savefig.dpi": ESTILO["dpi"],
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def nueva_figura(filas: int = 1, columnas: int = 1, escala: tuple = (1, 1)):
    """Crea una figura con el tamaño base de config escalado por `escala`."""
    ancho, alto = ESTILO["tamano_base"]
    return plt.subplots(filas, columnas, figsize=(ancho * escala[0], alto * escala[1]))


def agregar_fuente(fig, texto: str) -> None:
    """Escribe una nota de fuente o aclaración al pie de la figura."""
    fig.text(0.01, -0.02, texto, ha="left", va="top", fontsize=8, color=COLOR_TEXTO_SUAVE)


def guardar_figura(fig, nombre: str) -> str:
    """Guarda la figura en `figuras/<nombre>.png`, la cierra y devuelve el nombre de archivo."""
    RUTAS.figuras.mkdir(parents=True, exist_ok=True)
    archivo = f"{nombre}.{ESTILO['formato']}"
    fig.savefig(RUTAS.figuras / archivo)
    plt.close(fig)
    return archivo


def fuente_municipal() -> str:
    """Texto de la fuente de los datos municipales."""
    return "Fuente: datos municipales de COVID-19, Bolivia (mun_covid_se30.csv)."


def fuente_owid() -> str:
    """Texto de la fuente de OWID con la fecha de corte de config."""
    fecha = pd.Timestamp(CONFIG["fecha_corte"])
    return f"Fuente: Our World in Data (CC BY 4.0), datos al {fecha:%d/%m/%Y}."


# -----------------------------------------------------------------------------
# Figuras del análisis exploratorio
# -----------------------------------------------------------------------------
def serie_semanal_completa(nacional: list[dict]) -> pd.DataFrame:
    """Reindexa la serie nacional a todas las semanas del rango.

    Las semanas sin datos quedan como NaN para que la línea se corte en vez de
    unir dos semanas no consecutivas como si hubiera un dato intermedio.
    """
    tabla = pd.DataFrame(nacional).set_index("semana")
    inicio = CONFIG["municipios"]["semana_inicio"]
    fin = CONFIG["municipios"]["semana_fin"]
    return tabla.reindex(range(inicio, fin + 1))


def nota_semanas_faltantes() -> str:
    """Texto que avisa qué semanas faltan en los datos municipales (vacío si ninguna)."""
    faltantes = leer_json("calidad")["municipios"]["semanas_faltantes"]
    if not faltantes:
        return ""
    lista = ", ".join(str(semana) for semana in faltantes)
    return f" Sin datos en la SE {lista} (la línea se interrumpe)."


def figura_evolucion_nacional(eda: dict) -> str:
    """Tres paneles: casos acumulados, fallecidos acumulados y letalidad nacional.

    Se separan en paneles (en vez de un doble eje) porque los fallecidos y la
    letalidad tienen escalas muy distintas a las de los casos.
    """
    tabla = serie_semanal_completa(eda["nacional"])
    fig, ejes = nueva_figura(1, 3, escala=(1.6, 1))
    series_casos = [
        ("confirmados", "Confirmados", COLOR_FOCO),
        ("activos", "Activos", COLOR_AZUL),
        ("recuperados", "Recuperados", COLOR_GRIS),
    ]
    for columna, etiqueta, color in series_casos:
        ejes[0].plot(tabla.index, tabla[columna], marker="o", markersize=4, lw=2, color=color, label=etiqueta)
    ejes[0].set_title("Casos acumulados")
    ejes[0].legend(loc="upper left")
    ejes[1].plot(tabla.index, tabla["fallecidos"], marker="o", markersize=4, lw=2, color=COLOR_FOCO)
    ejes[1].set_title("Fallecidos acumulados")
    ejes[2].plot(tabla.index, tabla["letalidad"], marker="o", markersize=4, lw=2, color=COLOR_FOCO)
    ejes[2].set_title("Letalidad nacional (%)")
    ejes[2].yaxis.set_major_formatter(formateador_numeros(1))
    for eje in ejes[:2]:
        eje.yaxis.set_major_formatter(formateador_numeros(0))
        eje.set_ylim(bottom=0)
    for eje in ejes:
        eje.set_xlabel("Semana epidemiológica")
        eje.set_xticks(tabla.index[::2])
    ultima = eda["nacional"][-1]
    ejes[0].annotate(
        formatear_numero(ultima["confirmados"]),
        (ultima["semana"], ultima["confirmados"]),
        xytext=(-6, 6), textcoords="offset points", ha="right", fontsize=9, color=COLOR_TEXTO,
    )
    ejes[1].annotate(
        formatear_numero(ultima["fallecidos"]),
        (ultima["semana"], ultima["fallecidos"]),
        xytext=(-6, 6), textcoords="offset points", ha="right", fontsize=9, color=COLOR_TEXTO,
    )
    ejes[2].annotate(
        f"{formatear_numero(ultima['letalidad'], 2)} %",
        (ultima["semana"], ultima["letalidad"]),
        xytext=(-6, 6), textcoords="offset points", ha="right", fontsize=9, color=COLOR_TEXTO,
    )
    fig.tight_layout()
    agregar_fuente(fig, fuente_municipal() + nota_semanas_faltantes())
    return guardar_figura(fig, "eda_evolucion_nacional")


def figura_histogramas(eda: dict) -> str:
    """Distribución por municipio de confirmados (escala log) y de letalidad.

    Los confirmados se dibujan en escala logarítmica porque van de 1 a más de
    20 000: en escala lineal casi todos los municipios caerían en la primera barra.
    """
    distribucion = eda["distribucion"]
    confirmados = np.array(distribucion["valores_confirmados"], dtype=float)
    letalidad = np.array(distribucion["valores_letalidad"], dtype=float)
    fig, ejes = nueva_figura(1, 2, escala=(1.4, 1))
    limite_superior = np.ceil(np.log10(confirmados.max()))
    bordes = np.logspace(0, limite_superior, int(limite_superior * 4) + 1)
    ejes[0].hist(confirmados, bins=bordes, color=COLOR_AZUL, edgecolor="white", linewidth=1)
    ejes[0].set_xscale("log")
    ejes[0].xaxis.set_major_formatter(formateador_numeros(0))
    mediana_confirmados = distribucion["confirmados"]["mediana"]
    ejes[0].axvline(mediana_confirmados, color=COLOR_FOCO, lw=1.5, ls="--")
    ejes[0].annotate(
        f"Mediana: {formatear_numero(mediana_confirmados)}",
        (mediana_confirmados, 0.95), xycoords=("data", "axes fraction"),
        xytext=(5, 0), textcoords="offset points", fontsize=9, color=COLOR_TEXTO,
    )
    ejes[0].set_title("Confirmados por municipio")
    ejes[0].set_xlabel("Confirmados acumulados (escala logarítmica)")
    ejes[0].set_ylabel("Número de municipios")
    ejes[1].hist(letalidad, bins=20, color=COLOR_AZUL, edgecolor="white", linewidth=1)
    mediana_letalidad = distribucion["letalidad"]["mediana"]
    ejes[1].axvline(mediana_letalidad, color=COLOR_FOCO, lw=1.5, ls="--")
    ejes[1].annotate(
        f"Mediana: {formatear_numero(mediana_letalidad, 1)} %",
        (mediana_letalidad, 0.95), xycoords=("data", "axes fraction"),
        xytext=(5, 0), textcoords="offset points", fontsize=9, color=COLOR_TEXTO,
    )
    ejes[1].set_title("Letalidad por municipio")
    ejes[1].set_xlabel("Letalidad (%)")
    ejes[1].set_ylabel("Número de municipios")
    ejes[1].xaxis.set_major_formatter(formateador_numeros(0))
    fig.tight_layout()
    semana = eda["semana_fin"]
    nota = f"{fuente_municipal()} Semana epidemiológica {semana}; letalidad solo de municipios con casos."
    agregar_fuente(fig, nota)
    return guardar_figura(fig, "eda_histogramas")


def figura_departamentos(eda: dict) -> str:
    """Confirmados y letalidad por departamento, con la letalidad nacional de referencia."""
    tabla = pd.DataFrame(eda["departamentos"])
    tabla["nombre"] = tabla["departamento"].map(nombre_departamento)
    nacional = eda["nacional"][-1]["letalidad"]
    fig, ejes = nueva_figura(1, 2, escala=(1.4, 1.1))
    orden_casos = tabla.sort_values("confirmados")
    ejes[0].barh(orden_casos["nombre"], orden_casos["confirmados"], color=COLOR_AZUL, height=0.7)
    for posicion, valor in enumerate(orden_casos["confirmados"]):
        ejes[0].text(valor, posicion, f" {formatear_numero(valor)}", va="center", fontsize=8, color=COLOR_TEXTO)
    ejes[0].set_title("Confirmados acumulados")
    ejes[0].xaxis.set_major_formatter(formateador_numeros(0))
    ejes[0].set_xlim(0, orden_casos["confirmados"].max() * 1.22)
    ejes[0].grid(axis="y", visible=False)
    orden_letalidad = tabla.sort_values("letalidad")
    colores = [COLOR_FOCO if valor >= nacional else COLOR_GRIS for valor in orden_letalidad["letalidad"]]
    ejes[1].barh(orden_letalidad["nombre"], orden_letalidad["letalidad"], color=colores, height=0.7)
    for posicion, valor in enumerate(orden_letalidad["letalidad"]):
        ejes[1].text(valor, posicion, f" {formatear_numero(valor, 1)}", va="center", fontsize=8, color=COLOR_TEXTO)
    ejes[1].axvline(nacional, color=COLOR_TEXTO, lw=1, ls="--")
    ejes[1].annotate(
        f"Nacional: {formatear_numero(nacional, 2)} %",
        (nacional, 1.0), xycoords=("data", "axes fraction"),
        xytext=(4, -2), textcoords="offset points", va="top", fontsize=8, color=COLOR_TEXTO,
    )
    ejes[1].set_title("Letalidad (%)")
    ejes[1].xaxis.set_major_formatter(formateador_numeros(0))
    ejes[1].set_xlim(0, orden_letalidad["letalidad"].max() * 1.18)
    ejes[1].grid(axis="y", visible=False)
    fig.tight_layout()
    nota = f"{fuente_municipal()} Semana {eda['semana_fin']}. En rojo, departamentos con letalidad igual o mayor a la nacional."
    agregar_fuente(fig, nota)
    return guardar_figura(fig, "eda_departamentos")


def figura_boxplot_letalidad(eda: dict) -> str:
    """Boxplot de letalidad municipal por departamento con los puntos individuales.

    Se superponen los puntos porque varios grupos tienen menos de 10
    municipios y una caja sola ocultaría cuántos datos la sostienen.
    """
    prueba = eda["kruskal"]
    detalle = prueba["detalle"]
    etiquetas = [f"{nombre_departamento(g['departamento'])}\n(n = {g['n']})" for g in detalle]
    valores = [g["valores"] for g in detalle]
    fig, eje = nueva_figura(1, 1, escala=(1.2, 1.1))
    eje.boxplot(
        valores,
        tick_labels=etiquetas,
        widths=0.55,
        showfliers=False,
        medianprops={"color": COLOR_FOCO, "lw": 2},
        boxprops={"color": COLOR_TEXTO_SUAVE},
        whiskerprops={"color": COLOR_TEXTO_SUAVE},
        capprops={"color": COLOR_TEXTO_SUAVE},
    )
    generador = np.random.default_rng(CONFIG["semilla"])
    for posicion, grupo in enumerate(valores, start=1):
        desplazamiento = generador.uniform(-0.15, 0.15, len(grupo))
        eje.scatter(posicion + desplazamiento, grupo, s=18, color=COLOR_AZUL, alpha=0.7, zorder=3, edgecolor="white", linewidth=0.5)
    nacional = eda["nacional"][-1]["letalidad"]
    eje.axhline(nacional, color=COLOR_TEXTO, lw=1, ls="--", label=f"Letalidad nacional: {formatear_numero(nacional, 2)} %")
    eje.legend(loc="upper right", fontsize=9)
    texto_prueba = (
        f"Kruskal-Wallis: H = {formatear_numero(prueba['H'], 1)}, "
        f"gl = {prueba['gl']}, {formatear_p(prueba['p'])}; "
        f"ε² = {formatear_numero(prueba['epsilon2'], 2)}"
    )
    eje.set_ylabel("Letalidad municipal (%)")
    eje.yaxis.set_major_formatter(formateador_numeros(None))
    eje.set_ylim(bottom=0)
    eje.grid(axis="x", visible=False)
    titulo = f"Letalidad de municipios con {prueba['minimo_casos']} casos o más, por departamento"
    eje.set_title(f"{titulo}\n", loc="left")
    eje.text(0, 1.015, texto_prueba, transform=eje.transAxes, fontsize=9, color=COLOR_TEXTO_SUAVE)
    fig.tight_layout()
    excluidos = ", ".join(f"{nombre_departamento(d)} ({n})" for d, n in prueba["excluidos"].items())
    nota = f"{fuente_municipal()} Semana {eda['semana_fin']}."
    if excluidos:
        nota += f" Excluidos de la prueba por tener un solo municipio: {excluidos}."
    agregar_fuente(fig, nota)
    return guardar_figura(fig, "eda_boxplot_letalidad")


def figura_spearman(eda: dict) -> str:
    """Matriz de correlaciones de Spearman con escala divergente centrada en cero."""
    spearman = eda["spearman"]
    etiquetas = [CONFIG["eda"]["etiquetas_variables"][v] for v in spearman["variables"]]
    matriz = np.array(spearman["rho"], dtype=float)
    fig, eje = nueva_figura(1, 1, escala=(0.85, 1.25))
    imagen = eje.imshow(matriz, cmap="RdBu_r", vmin=-1, vmax=1)
    for fila in range(matriz.shape[0]):
        for columna in range(matriz.shape[1]):
            valor = matriz[fila, columna]
            color = "white" if abs(valor) > 0.6 else COLOR_TEXTO
            eje.text(columna, fila, formatear_numero(valor, 2), ha="center", va="center", fontsize=9, color=color)
    eje.set_xticks(range(len(etiquetas)), labels=etiquetas, rotation=35, ha="right")
    eje.set_yticks(range(len(etiquetas)), labels=etiquetas)
    eje.grid(False)
    for borde in eje.spines.values():
        borde.set_visible(False)
    barra = fig.colorbar(imagen, ax=eje, fraction=0.046, pad=0.04)
    barra.ax.yaxis.set_major_formatter(formateador_numeros(None))
    barra.set_label("ρ de Spearman")
    eje.set_title("Correlaciones de Spearman entre variables municipales")
    fig.tight_layout()
    agregar_fuente(fig, f"{fuente_municipal()} Semana {eda['semana_fin']}, {spearman['n']} municipios con casos.")
    return guardar_figura(fig, "eda_spearman")


def figura_comparacion_casos(eda: dict) -> str:
    """Casos diarios por millón (media móvil) de los países comparados.

    El país foco va en rojo y grueso; los demás en gris fino con su nombre al
    final de la línea, para identificar cada país sin una leyenda de 8 colores.
    """
    series = pd.read_csv(RUTAS.procesados / "comparacion_series.csv", parse_dates=["date"])
    nombres = CONFIG["pais"]["nombres_es"]
    foco = CONFIG["pais"]["nombre"]
    fig, eje = nueva_figura(1, 1, escala=(1.3, 1.15))
    paises = [p for p in series["location"].unique() if p != foco] + [foco]
    posiciones_finales = []
    for pais in paises:
        datos_pais = series[series["location"] == pais].dropna(subset=["casos_millon_mm"])
        es_foco = pais == foco
        eje.plot(
            datos_pais["date"], datos_pais["casos_millon_mm"],
            color=color_pais(pais), lw=2.4 if es_foco else 1.2, zorder=3 if es_foco else 2,
        )
        final = datos_pais.iloc[-1]
        posiciones_finales.append((final["casos_millon_mm"], nombres.get(pais, pais), es_foco, final["date"]))
    eje.set_ylim(bottom=0)
    posiciones_finales = separar_etiquetas(posiciones_finales, eje)
    for valor, nombre, es_foco, fecha in posiciones_finales:
        eje.annotate(
            nombre, (fecha, valor), xytext=(5, 0), textcoords="offset points", va="center",
            fontsize=9, color=COLOR_FOCO if es_foco else COLOR_TEXTO_SUAVE,
            fontweight="bold" if es_foco else "normal",
        )
    inicio = pd.Timestamp(CONFIG["series"]["fecha_inicio"])
    corte = pd.Timestamp(CONFIG["fecha_corte"])
    eje.set_xticks(pd.date_range(inicio, corte, freq="MS"))
    eje.xaxis.set_major_formatter(formateador_meses())
    eje.yaxis.set_major_formatter(formateador_numeros(0))
    eje.set_xlim(inicio, corte + pd.Timedelta(days=22))
    eje.spines["bottom"].set_bounds(mdates.date2num(inicio), mdates.date2num(corte))
    ventana = eda["internacional"]["ventana_media_movil"]
    eje.set_title(f"Casos nuevos diarios por millón de habitantes (media móvil de {ventana} días)")
    eje.set_ylabel("Casos por millón")
    fig.tight_layout()
    agregar_fuente(fig, fuente_owid() + nota_correcciones(eda))
    return guardar_figura(fig, "eda_comparacion_casos")


def nota_correcciones(eda: dict) -> str:
    """Explica los saltos bruscos de las curvas y nombra los países con casos diarios negativos."""
    nota = "\nLos saltos bruscos reflejan reportes acumulados o correcciones de la fuente, no cambios reales de la transmisión."
    correcciones = eda["internacional"]["dias_negativos"]
    if correcciones:
        nombres = CONFIG["pais"]["nombres_es"]
        lista = ", ".join(f"{nombres.get(p, p)} ({n} días)" for p, n in correcciones.items())
        nota += f" Países con casos diarios negativos (recortados a cero en el eje): {lista}."
    return nota


def separar_etiquetas(posiciones: list, eje) -> list:
    """Desplaza verticalmente las etiquetas finales para que no se encimen.

    Ordena las etiquetas por valor y garantiza una distancia mínima entre
    ellas, proporcional al rango visible del eje (equivale a una línea de texto).
    """
    inferior, superior = eje.get_ylim()
    distancia = (superior - inferior) * 0.036
    ordenadas = sorted(posiciones, key=lambda fila: fila[0])
    ajustadas = []
    anterior = -np.inf
    for valor, nombre, es_foco, fecha in ordenadas:
        nuevo = max(valor, anterior + distancia)
        ajustadas.append((nuevo, nombre, es_foco, fecha))
        anterior = nuevo
    return ajustadas


def barras_paises(eje, tabla: pd.DataFrame, columna: str, decimales: int, titulo: str) -> None:
    """Dibuja barras horizontales ordenadas con el país foco en rojo y su valor al lado."""
    ordenada = tabla.sort_values(columna)
    colores = [COLOR_FOCO if foco else COLOR_GRIS for foco in ordenada["es_foco"]]
    eje.barh(ordenada["nombre"], ordenada[columna], color=colores, height=0.7)
    for posicion, valor in enumerate(ordenada[columna]):
        eje.text(valor, posicion, f" {formatear_numero(valor, decimales)}", va="center", fontsize=8, color=COLOR_TEXTO)
    eje.set_xlim(0, ordenada[columna].max() * 1.2)
    eje.xaxis.set_major_formatter(formateador_numeros(0))
    eje.grid(axis="y", visible=False)
    eje.set_title(titulo)


def figura_comparacion_indicadores(eda: dict) -> str:
    """Casos por millón, muertes por millón y letalidad a la fecha de corte."""
    tabla = pd.DataFrame(eda["internacional"]["indicadores"])
    fig, ejes = nueva_figura(1, 3, escala=(1.6, 1))
    barras_paises(ejes[0], tabla, "casos_por_millon", 0, "Casos por millón")
    barras_paises(ejes[1], tabla, "muertes_por_millon", 0, "Fallecidos por millón")
    barras_paises(ejes[2], tabla, "letalidad", 1, "Letalidad (%)")
    fig.tight_layout()
    agregar_fuente(fig, fuente_owid())
    return guardar_figura(fig, "eda_comparacion_indicadores")


def generar_figuras_eda() -> list[str]:
    """Dibuja todas las figuras del análisis exploratorio."""
    eda = leer_json("eda")
    archivos = [
        figura_evolucion_nacional(eda),
        figura_histogramas(eda),
        figura_departamentos(eda),
        figura_boxplot_letalidad(eda),
        figura_spearman(eda),
        figura_comparacion_casos(eda),
        figura_comparacion_indicadores(eda),
    ]
    return archivos


# -----------------------------------------------------------------------------
# Figuras del Modelo 1 · Perfiles de países
# -----------------------------------------------------------------------------
MARCADORES_GRUPO = ["o", "s", "^", "D", "v", "P", "X", "*"]


def nombres_variables_perfiles() -> dict:
    """Nombre en español de cada variable del Modelo 1 (de config)."""
    return {v["columna"]: v["nombre"] for v in CONFIG["perfiles"]["variables"]}


def nombre_pais(pais: str) -> str:
    """Nombre del país en español si está en config; si no, el de OWID."""
    return CONFIG["pais"]["nombres_es"].get(pais, pais)


def panel_silueta(eje, muestra: dict) -> None:
    """Silueta media por k; el k elegido en rojo y los k no admitidos en gris claro."""
    siluetas = pd.DataFrame(muestra["siluetas"])
    minimo = muestra["k_minimo_aceptado"]
    eje.plot(siluetas["k"], siluetas["silueta"], color=COLOR_AZUL, lw=2, marker="o", markersize=6, zorder=2)
    no_admitidos = siluetas[siluetas["k"] < minimo]
    if not no_admitidos.empty:
        eje.scatter(no_admitidos["k"], no_admitidos["silueta"], color="white", edgecolor=COLOR_GRIS, s=50, zorder=3, linewidth=1.5)
        eje.axvspan(siluetas["k"].min() - 0.5, minimo - 0.5, color=COLOR_GRILLA, alpha=0.6, lw=0)
        eje.text(siluetas["k"].min(), siluetas["silueta"].min(), f"k < {minimo}\nno admitido", fontsize=8, color=COLOR_TEXTO_SUAVE, va="bottom")
    elegido = siluetas[siluetas["k"] == muestra["k"]].iloc[0]
    eje.scatter([elegido["k"]], [elegido["silueta"]], color=COLOR_FOCO, s=90, zorder=4)
    eje.annotate(
        f"k = {muestra['k']}\nsilueta = {formatear_numero(elegido['silueta'], 3)}",
        (elegido["k"], elegido["silueta"]), xytext=(12, 0), textcoords="offset points",
        va="center", fontsize=9, color=COLOR_TEXTO,
    )
    eje.set_xticks(siluetas["k"])
    eje.set_xlabel("Número de grupos (k)")
    eje.set_ylabel("Silueta media")
    eje.yaxis.set_major_formatter(formateador_numeros(2))
    eje.set_xlim(siluetas["k"].min() - 0.5, siluetas["k"].max() + 0.5)
    eje.set_title(f"{muestra['etiqueta']} ({muestra['n_paises']} países)")


def figura_silueta(perfiles: dict) -> str:
    """Silueta por k en ambas muestras."""
    fig, ejes = nueva_figura(1, 2, escala=(1.4, 1))
    panel_silueta(ejes[0], perfiles["regional"])
    panel_silueta(ejes[1], perfiles["mundial"])
    fig.tight_layout()
    agregar_fuente(fig, f"{fuente_owid()} k-means con {CONFIG['perfiles']['n_inicios']} inicios sobre 8 variables estandarizadas.")
    return guardar_figura(fig, "perfiles_silueta")


def etiqueta_grupo(grupo: dict) -> str:
    """Texto de leyenda de un grupo: número, nombre y tamaño."""
    return f"G{grupo['grupo']} · {grupo['nombre']} ({grupo['tamano']})"


def figura_pca_mundial(perfiles: dict) -> str:
    """Plano PCA de la muestra mundial y cargas de las variables.

    Los grupos se distinguen por la forma del marcador; el grupo del país foco
    va en azul y el país foco en rojo, para respetar la convención de color.
    """
    muestra = perfiles["mundial"]
    pca = muestra["pca"]
    foco = CONFIG["pais"]["nombre"]
    coordenadas = pd.DataFrame(pca["coordenadas"], index=["pc1", "pc2"]).T
    coordenadas["grupo"] = pd.Series(muestra["asignacion"])
    fig, ejes = plt.subplots(1, 2, figsize=(ESTILO["tamano_base"][0] * 1.7, ESTILO["tamano_base"][1] * 1.35), gridspec_kw={"width_ratios": [1.6, 1]})
    eje = ejes[0]
    for grupo in muestra["grupos"]:
        puntos = coordenadas[coordenadas["grupo"] == grupo["grupo"]]
        es_grupo_foco = grupo["contiene_foco"]
        eje.scatter(
            puntos["pc1"], puntos["pc2"],
            marker=MARCADORES_GRUPO[(grupo["grupo"] - 1) % len(MARCADORES_GRUPO)],
            s=46 if es_grupo_foco else 34,
            color=COLOR_AZUL if es_grupo_foco else COLOR_GRIS,
            alpha=0.95 if es_grupo_foco else 0.7,
            edgecolor="white", linewidth=0.6,
            label=etiqueta_grupo(grupo), zorder=3 if es_grupo_foco else 2,
        )
    grupo_foco = next(g for g in muestra["grupos"] if g["contiene_foco"])
    miembros = [p for p in grupo_foco["miembros"] if p != foco]
    rotular_puntos(eje, coordenadas.loc[miembros, ["pc1", "pc2"]])
    x_foco, y_foco = coordenadas.loc[foco, ["pc1", "pc2"]]
    eje.scatter([x_foco], [y_foco], color=COLOR_FOCO, s=120, edgecolor="white", linewidth=1.2, zorder=5)
    eje.annotate(nombre_pais(foco), (x_foco, y_foco), xytext=(6, -12), textcoords="offset points", fontsize=10, fontweight="bold", color=COLOR_FOCO)
    varianza = pca["varianza_explicada"]
    eje.set_xlabel(f"Componente 1 ({formatear_numero(varianza[0] * 100, 1)} % de la varianza)")
    eje.set_ylabel(f"Componente 2 ({formatear_numero(varianza[1] * 100, 1)} % de la varianza)")
    eje.xaxis.set_major_formatter(formateador_numeros(None))
    eje.yaxis.set_major_formatter(formateador_numeros(None))
    eje.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8, handletextpad=0.3)
    eje.set_title(f"Grupos de países en el plano PCA (k = {muestra['k']})")
    panel_cargas(ejes[1], pca)
    fig.tight_layout()
    nota = (
        f"{fuente_owid()} {muestra['n_paises']} países con datos completos; "
        f"silueta = {formatear_numero(muestra['silueta'], 3)}; ARI frente a Ward = {formatear_numero(muestra['ari_ward'], 2)}."
    )
    agregar_fuente(fig, nota)
    return guardar_figura(fig, "perfiles_pca_mundial")


def rotular_puntos(eje, puntos: pd.DataFrame) -> None:
    """Escribe el nombre de cada punto; si choca con una etiqueta ya puesta, lo baja.

    La cercanía se mide como fracción del rango de cada eje, de modo que la
    regla funciona con cualquier escala.
    """
    rango_x = np.ptp(eje.get_xlim())
    rango_y = np.ptp(eje.get_ylim())
    colocadas = []
    for pais, (x, y) in puntos.sort_values("pc1").iterrows():
        cerca = any(abs(x - cx) < rango_x * 0.08 and abs(y - cy) < rango_y * 0.035 for cx, cy in colocadas)
        desplazamiento = (4, -10) if cerca else (4, 3)
        eje.annotate(nombre_pais(pais), (x, y), xytext=desplazamiento, textcoords="offset points", fontsize=7, color=COLOR_TEXTO_SUAVE)
        colocadas.append((x, y))


def panel_cargas(eje, pca: dict) -> None:
    """Flechas con la carga de cada variable en las dos componentes."""
    nombres = nombres_variables_perfiles()
    for columna, (carga1, carga2) in pca["cargas"].items():
        eje.annotate("", xy=(carga1, carga2), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": COLOR_AZUL, "lw": 1.4})
        alineacion = "left" if carga1 >= 0 else "right"
        eje.text(carga1 * 1.08, carga2 * 1.08, nombres[columna], fontsize=8, ha=alineacion, va="center", color=COLOR_TEXTO)
    limite = max(abs(v) for par in pca["cargas"].values() for v in par) * 1.6
    eje.set_xlim(-limite, limite)
    eje.set_ylim(-limite, limite)
    eje.axhline(0, color=COLOR_GRIS, lw=0.8)
    eje.axvline(0, color=COLOR_GRIS, lw=0.8)
    eje.set_aspect("equal")
    eje.xaxis.set_major_formatter(formateador_numeros(None))
    eje.yaxis.set_major_formatter(formateador_numeros(None))
    eje.set_xlabel("Carga en la componente 1")
    eje.set_ylabel("Carga en la componente 2")
    eje.set_title("Cargas de las variables")


def figura_perfiles_grupos(perfiles: dict) -> str:
    """Mapa de calor del perfil estandarizado medio de cada grupo (muestra mundial).

    Se muestran valores z para que todas las variables compartan escala: un
    valor de +1 significa una desviación estándar por encima de la media mundial.
    """
    muestra = perfiles["mundial"]
    nombres = nombres_variables_perfiles()
    columnas = list(nombres)
    matriz = np.array([[g["perfil_z"][c] for c in columnas] for g in muestra["grupos"]])
    etiquetas_filas = []
    for grupo in muestra["grupos"]:
        marca = f"  ← {nombre_pais(CONFIG['pais']['nombre'])}" if grupo["contiene_foco"] else ""
        etiquetas_filas.append(f"{etiqueta_grupo(grupo)}{marca}")
    fig, eje = nueva_figura(1, 1, escala=(1.5, 1.1))
    limite = np.ceil(np.abs(matriz).max())
    imagen = eje.imshow(matriz, cmap="RdBu_r", vmin=-limite, vmax=limite, aspect="auto")
    for fila in range(matriz.shape[0]):
        for columna in range(matriz.shape[1]):
            valor = matriz[fila, columna]
            color = "white" if abs(valor) > limite * 0.55 else COLOR_TEXTO
            eje.text(columna, fila, formatear_numero(valor, 1), ha="center", va="center", fontsize=9, color=color)
    eje.set_xticks(range(len(columnas)), labels=[nombres[c] for c in columnas], rotation=30, ha="right")
    eje.set_yticks(range(len(etiquetas_filas)), labels=etiquetas_filas)
    eje.grid(False)
    for borde in eje.spines.values():
        borde.set_visible(False)
    barra = fig.colorbar(imagen, ax=eje, fraction=0.03, pad=0.02)
    barra.set_label("Desviaciones estándar respecto de la media")
    barra.ax.yaxis.set_major_formatter(formateador_numeros(None))
    eje.set_title("Perfil medio de cada grupo (muestra mundial, valores estandarizados)")
    fig.tight_layout()
    agregar_fuente(fig, f"{fuente_owid()} Entre paréntesis, número de países por grupo.")
    return guardar_figura(fig, "perfiles_grupos")


def figura_pca_regional(perfiles: dict) -> str:
    """Plano PCA de la muestra regional con el nombre de cada país."""
    muestra = perfiles["regional"]
    pca = muestra["pca"]
    foco = CONFIG["pais"]["nombre"]
    coordenadas = pd.DataFrame(pca["coordenadas"], index=["pc1", "pc2"]).T
    coordenadas["grupo"] = pd.Series(muestra["asignacion"])
    fig, eje = nueva_figura(1, 1, escala=(1.1, 1.15))
    for grupo in muestra["grupos"]:
        puntos = coordenadas[coordenadas["grupo"] == grupo["grupo"]]
        eje.scatter(
            puntos["pc1"], puntos["pc2"], s=70,
            marker=MARCADORES_GRUPO[(grupo["grupo"] - 1) % len(MARCADORES_GRUPO)],
            color=COLOR_AZUL if grupo["contiene_foco"] else COLOR_GRIS,
            edgecolor="white", linewidth=0.8, label=etiqueta_grupo(grupo), zorder=3,
        )
    for pais, fila in coordenadas.iterrows():
        es_foco = pais == foco
        if es_foco:
            eje.scatter([fila["pc1"]], [fila["pc2"]], s=130, color=COLOR_FOCO, edgecolor="white", linewidth=1.2, zorder=5)
        eje.annotate(
            nombre_pais(pais), (fila["pc1"], fila["pc2"]), xytext=(7, 4), textcoords="offset points",
            fontsize=10 if es_foco else 9, fontweight="bold" if es_foco else "normal",
            color=COLOR_FOCO if es_foco else COLOR_TEXTO,
        )
    varianza = pca["varianza_explicada"]
    eje.set_xlabel(f"Componente 1 ({formatear_numero(varianza[0] * 100, 1)} %)")
    eje.set_ylabel(f"Componente 2 ({formatear_numero(varianza[1] * 100, 1)} %)")
    eje.xaxis.set_major_formatter(formateador_numeros(None))
    eje.yaxis.set_major_formatter(formateador_numeros(None))
    eje.margins(x=0.2, y=0.15)
    eje.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, fontsize=8)
    eje.set_title(f"{muestra['etiqueta']}: k = {muestra['k']}, silueta = {formatear_numero(muestra['silueta'], 2)}")
    fig.tight_layout()
    agregar_fuente(fig, fuente_owid())
    return guardar_figura(fig, "perfiles_regional")


def generar_figuras_perfiles() -> list[str]:
    """Dibuja todas las figuras del Modelo 1."""
    perfiles = leer_json("perfiles")
    archivos = [
        figura_silueta(perfiles),
        figura_pca_mundial(perfiles),
        figura_perfiles_grupos(perfiles),
        figura_pca_regional(perfiles),
    ]
    return archivos


# -----------------------------------------------------------------------------
# Figuras del Modelo 2 · Pronóstico e índice de presión sanitaria
# -----------------------------------------------------------------------------
AZUL_CLARO = "#92C5DE"   # tono claro de la misma escala RdBu, para bandas de intervalo
AZUL_MUY_CLARO = "#D1E5F0"


def formato_eje_fechas(eje, inicio: pd.Timestamp, fin: pd.Timestamp) -> None:
    """Marcas mensuales con nombres de mes en español entre dos fechas."""
    eje.set_xticks(pd.date_range(inicio, fin, freq="MS"))
    eje.xaxis.set_major_formatter(formateador_meses())


def serie_foco(series: dict) -> pd.Series:
    """Serie semanal del país foco como `pd.Series` con índice de fechas."""
    return pd.Series(series["serie"]["casos"], index=pd.to_datetime(series["serie"]["fechas"]))


def figura_validacion(series: dict) -> str:
    """Serie real y los pronósticos a 4 semanas del mejor modelo desde cada corte.

    Mostrar todas las trayectorias deja ver dónde falla el modelo (por
    ejemplo, al no anticipar el pico), algo que una métrica resumida oculta.
    """
    real = serie_foco(series)
    mejor = series["mejor_modelo"]
    validacion = pd.DataFrame(series["validacion_mejor"])
    validacion["fecha"] = pd.to_datetime(validacion["fecha"])
    fig, eje = nueva_figura(1, 1, escala=(1.3, 1.1))
    eje.plot(real.index, real.to_numpy(), color=COLOR_FOCO, lw=2.2, marker="o", markersize=4, label="Casos semanales observados", zorder=3)
    trayectorias = validacion[validacion["modelo"] == mejor]
    primera = True
    for origen, grupo in trayectorias.groupby("origen"):
        anterior = real.index[origen - 1]
        fechas = [anterior] + grupo["fecha"].tolist()
        valores = [real.iloc[origen - 1]] + grupo["pronostico"].tolist()
        etiqueta = f"Pronósticos a {series['horizonte']} semanas ({series['cortes']} cortes)" if primera else None
        eje.plot(fechas, valores, color=COLOR_AZUL, lw=1.1, alpha=0.75, label=etiqueta, zorder=2)
        primera = False
    inicio_validacion = real.index[series["primer_origen"]]
    eje.axvline(inicio_validacion, color=COLOR_GRIS, lw=1, ls=":")
    eje.annotate("Inicio de la validación", (inicio_validacion, 1.0), xycoords=("data", "axes fraction"),
                 xytext=(-4, -4), textcoords="offset points", ha="right", va="top", fontsize=8, color=COLOR_TEXTO_SUAVE)
    fila_mejor = next(f for f in series["metricas"] if f["modelo"] == mejor)
    eje.set_title(f"Validación con origen móvil: {fila_mejor['nombre']}")
    eje.set_ylabel("Casos nuevos por semana")
    eje.yaxis.set_major_formatter(formateador_numeros(0))
    eje.set_ylim(bottom=0)
    formato_eje_fechas(eje, real.index[0], real.index[-1])
    eje.legend(loc="upper left")
    fig.tight_layout()
    agregar_fuente(fig, f"{fuente_owid()} Semanas que cierran en sábado.")
    return guardar_figura(fig, "series_validacion")


def figura_metricas(series: dict) -> str:
    """RMSE de cada modelo y su evolución según el horizonte."""
    metricas = pd.DataFrame(series["metricas"])
    mejor = series["mejor_modelo"]
    referencia = CONFIG["series"]["modelo_referencia_mase"]
    nombres = CONFIG["series"]["nombres_modelos"]
    fig, ejes = nueva_figura(1, 2, escala=(1.5, 1.05))
    ordenadas = metricas.sort_values("RMSE", ascending=False)
    colores = [COLOR_AZUL if m == mejor else COLOR_GRIS for m in ordenadas["modelo"]]
    ejes[0].barh(ordenadas["nombre"], ordenadas["RMSE"], color=colores, height=0.65)
    for posicion, valor in enumerate(ordenadas["RMSE"]):
        ejes[0].text(valor, posicion, f" {formatear_numero(valor)}", va="center", fontsize=8, color=COLOR_TEXTO)
    ejes[0].set_xlim(0, ordenadas["RMSE"].max() * 1.2)
    ejes[0].xaxis.set_major_formatter(formateador_numeros(0))
    ejes[0].grid(axis="y", visible=False)
    ejes[0].set_title("RMSE en la validación (casos por semana)")
    horizontes = range(1, series["horizonte"] + 1)
    for modelo, valores in series["rmse_por_horizonte"].items():
        es_mejor = modelo == mejor
        es_referencia = modelo == referencia
        if es_mejor:
            color = COLOR_AZUL
        elif es_referencia:
            color = COLOR_TEXTO_SUAVE
        else:
            color = "#C8C8C8"
        ejes[1].plot(
            horizontes, valores, color=color,
            lw=2.2 if es_mejor else 1.6, ls="--" if es_referencia else "-", marker="o", markersize=4,
            zorder=3 if es_mejor else 2,
        )
        if es_mejor or es_referencia:
            ejes[1].annotate(nombres[modelo], (horizontes[-1], valores[-1]), xytext=(6, 0), textcoords="offset points",
                             va="center", fontsize=8, color=COLOR_TEXTO)
    ejes[1].set_xticks(list(horizontes))
    ejes[1].set_xlabel("Semanas hacia adelante (h)")
    ejes[1].yaxis.set_major_formatter(formateador_numeros(0))
    ejes[1].set_xlim(0.7, series["horizonte"] + 1.3)
    ejes[1].set_ylim(bottom=0)
    ejes[1].set_title("RMSE según el horizonte")
    fig.tight_layout()
    fila_mejor = metricas[metricas["modelo"] == mejor].iloc[0]
    nota = (
        f"{fuente_owid()} En gris claro, el resto de los modelos. "
        f"R² de {fila_mejor['nombre']} = {formatear_numero(fila_mejor['R2'], 2)}: "
        "todos los modelos tienen R² negativo en la validación."
        if (metricas["R2"] < 0).all() else fuente_owid()
    )
    agregar_fuente(fig, nota)
    return guardar_figura(fig, "series_metricas")


def panel_proyeccion(eje, real: pd.Series, proyeccion: dict, titulo: str) -> None:
    """Serie observada más una proyección con bandas de 80 % y 95 %."""
    semanas = pd.DataFrame(proyeccion["semanas"])
    fechas = pd.to_datetime(semanas["fecha"])
    niveles = sorted(CONFIG["series"]["niveles_intervalo"], reverse=True)
    tonos = {niveles[0]: AZUL_MUY_CLARO, niveles[-1]: AZUL_CLARO}
    ultima_fecha = real.index[-1]
    ultimo_valor = real.iloc[-1]
    for nivel in niveles:
        inferior = [ultimo_valor] + semanas[f"inferior_{nivel}"].tolist()
        superior = [ultimo_valor] + semanas[f"superior_{nivel}"].tolist()
        eje.fill_between([ultima_fecha] + fechas.tolist(), inferior, superior, color=tonos[nivel], lw=0, label=f"Intervalo {nivel} %")
    eje.plot(real.index, real.to_numpy(), color=COLOR_FOCO, lw=2, marker="o", markersize=3, label="Observado")
    eje.plot([ultima_fecha] + fechas.tolist(), [ultimo_valor] + semanas["pronostico"].tolist(),
             color=COLOR_AZUL, lw=2, marker="o", markersize=4, label="Pronóstico")
    ultima = semanas.iloc[-1]
    eje.annotate(formatear_numero(ultima["pronostico"]), (fechas.iloc[-1], ultima["pronostico"]),
                 xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=COLOR_TEXTO)
    eje.set_title(titulo)
    eje.yaxis.set_major_formatter(formateador_numeros(0))
    eje.set_ylim(bottom=0)
    formato_eje_fechas(eje, real.index[0], fechas.iloc[-1])


def figura_proyeccion(series: dict) -> str:
    """Proyección del mejor modelo (intervalos empíricos) junto a ARIMA (intervalos teóricos)."""
    real = serie_foco(series)
    fig, ejes = plt.subplots(1, 2, figsize=(ESTILO["tamano_base"][0] * 1.6, ESTILO["tamano_base"][1] * 1.05), sharey=True)
    proyeccion = series["proyeccion"]
    panel_proyeccion(ejes[0], real, proyeccion, f"{proyeccion['nombre']} · intervalos empíricos")
    arima = series["proyeccion_arima"]
    orden = ", ".join(str(v) for v in arima["orden"])
    panel_proyeccion(ejes[1], real, arima, f"ARIMA({orden}) sobre log1p · intervalos del modelo")
    ejes[0].set_ylabel("Casos nuevos por semana")
    ejes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    nota = (
        f"{fuente_owid()} Intervalos empíricos: cuantiles de los errores de validación por horizonte "
        f"({series['cortes']} cortes); límites inferiores truncados en cero."
    )
    agregar_fuente(fig, nota)
    return guardar_figura(fig, "series_proyeccion")


def figura_ips(series: dict) -> str:
    """IPS diario de los países comparados, con el pico del país foco señalado."""
    tabla = pd.read_csv(RUTAS.procesados / "ips_series.csv", parse_dates=["date"])
    foco = CONFIG["pais"]["nombre"]
    ips = series["ips"]
    inicio = pd.Timestamp(CONFIG["series"]["fecha_inicio"])
    corte = pd.Timestamp(CONFIG["fecha_corte"])
    tabla = tabla[tabla["date"] >= inicio]
    fig, eje = nueva_figura(1, 1, escala=(1.3, 1.15))
    paises = [p for p in tabla["location"].unique() if p != foco] + [foco]
    finales = []
    for pais in paises:
        datos_pais = tabla[tabla["location"] == pais].dropna(subset=["ips"])
        es_foco = pais == foco
        eje.plot(datos_pais["date"], datos_pais["ips"], color=color_pais(pais), lw=2.4 if es_foco else 1.2, zorder=3 if es_foco else 2)
        final = datos_pais.iloc[-1]
        finales.append((final["ips"], nombre_pais(pais), es_foco, final["date"]))
    eje.set_ylim(bottom=0)
    for valor, nombre, es_foco, fecha in separar_etiquetas(finales, eje):
        eje.annotate(nombre, (fecha, valor), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9,
                     color=COLOR_FOCO if es_foco else COLOR_TEXTO_SUAVE, fontweight="bold" if es_foco else "normal")
    fila_foco = next(f for f in ips["paises"] if f["es_foco"])
    fecha_pico = pd.Timestamp(fila_foco["fecha_pico"])
    eje.scatter([fecha_pico], [fila_foco["ips_pico"]], color=COLOR_FOCO, s=50, zorder=4, edgecolor="white")
    eje.annotate(
        f"Pico: {formatear_numero(fila_foco['ips_pico'], 2)} ({fecha_pico:%d/%m})\n{formatear_numero(fila_foco['camas'])} camas",
        (fecha_pico, fila_foco["ips_pico"]), xytext=(-10, 12), textcoords="offset points", ha="right", fontsize=8, color=COLOR_TEXTO,
    )
    eje.axhline(ips["mediana_ips_pico_otros"], color=COLOR_TEXTO_SUAVE, lw=1, ls="--")
    eje.annotate(f"Mediana del pico de los otros países: {formatear_numero(ips['mediana_ips_pico_otros'], 2)}",
                 (inicio, ips["mediana_ips_pico_otros"]), xytext=(2, 4), textcoords="offset points", fontsize=8, color=COLOR_TEXTO_SUAVE)
    formato_eje_fechas(eje, inicio, corte)
    eje.set_xlim(inicio, corte + pd.Timedelta(days=24))
    eje.spines["bottom"].set_bounds(mdates.date2num(inicio), mdates.date2num(corte))
    eje.yaxis.set_major_formatter(formateador_numeros(None))
    eje.set_ylabel("Casos de 14 días por cama")
    eje.set_title(f"Índice de presión sanitaria: casos nuevos de {ips['ventana_dias']} días por cama hospitalaria")
    fig.tight_layout()
    agregar_fuente(fig, f"{fuente_owid()} Camas hospitalarias totales (no de terapia intensiva).")
    return guardar_figura(fig, "series_ips")


def figura_escenarios(series: dict) -> str:
    """IPS pico del país foco con distintos aumentos de camas frente a la mediana regional."""
    ips = series["ips"]
    escenarios = pd.DataFrame(ips["escenarios"])
    etiquetas = [
        "Actual" if incremento == 0 else f"+{formatear_numero(incremento, 0)} %"
        for incremento in escenarios["incremento_pct"]
    ]
    colores = [COLOR_FOCO if fila["incremento_pct"] == 0 else COLOR_AZUL for _, fila in escenarios.iterrows()]
    fig, eje = nueva_figura(1, 1, escala=(1.0, 1.0))
    posiciones = np.arange(len(escenarios))
    eje.bar(posiciones, escenarios["ips_pico"], color=colores, width=0.6)
    for posicion, fila in escenarios.iterrows():
        texto = f"{formatear_numero(fila['ips_pico'], 2)}\n{formatear_numero(fila['camas'])} camas"
        eje.text(posicion, fila["ips_pico"], texto, ha="center", va="bottom", fontsize=8, color=COLOR_TEXTO)
    mediana = ips["mediana_ips_pico_otros"]
    eje.axhline(mediana, color=COLOR_TEXTO_SUAVE, lw=1, ls="--")
    eje.annotate(
        f"Mediana regional: {formatear_numero(mediana, 2)} → requiere {formatear_numero(ips['camas_necesarias'])} camas "
        f"(+{formatear_numero(ips['incremento_necesario_pct'], 1)} %)",
        (1.0, mediana), xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8, color=COLOR_TEXTO_SUAVE,
    )
    eje.set_xticks(posiciones, labels=etiquetas)
    eje.set_xlabel("Camas hospitalarias adicionales")
    eje.set_ylabel("IPS en el pico")
    eje.yaxis.set_major_formatter(formateador_numeros(None))
    eje.set_ylim(0, escenarios["ips_pico"].max() * 1.25)
    eje.grid(axis="x", visible=False)
    eje.set_title(f"IPS pico de {nombre_pais(CONFIG['pais']['nombre'])} con más camas")
    fig.tight_layout()
    agregar_fuente(fig, "Mismos casos de 14 días del pico; solo cambia el número de camas totales.")
    return guardar_figura(fig, "series_escenarios")


def generar_figuras_series() -> list[str]:
    """Dibuja todas las figuras del Modelo 2."""
    series = leer_json("series")
    archivos = [
        figura_validacion(series),
        figura_metricas(series),
        figura_proyeccion(series),
        figura_ips(series),
        figura_escenarios(series),
    ]
    return archivos


# -----------------------------------------------------------------------------
# Figuras del Modelo 3 · Riesgo de letalidad alta
# -----------------------------------------------------------------------------
def etiqueta_modelo_riesgo(fila: dict) -> str:
    """Nombre del modelo con su conjunto de variables (las líneas base no llevan conjunto)."""
    if fila["conjunto"] == "-" or fila["modelo"] == "mayoritaria":
        return CONFIG["riesgo"]["nombres_modelos"][fila["modelo"]].split(" (")[0]
    return f"{fila['nombre']} · {fila['conjunto']}"


def panel_intervalos(eje, filas: list[dict], metrica: str, persistencia: dict) -> None:
    """Media e IC 95 % de una métrica por modelo, con la banda de la persistencia de fondo."""
    banda = persistencia[metrica]
    eje.axvspan(banda["ic_inf"], banda["ic_sup"], color=AZUL_MUY_CLARO, lw=0, zorder=0)
    eje.axvline(banda["media"], color=COLOR_AZUL, lw=1, ls="--", zorder=1)
    for posicion, fila in enumerate(filas):
        valor = fila[metrica]
        es_persistencia = fila["modelo"] == "persistencia"
        if es_persistencia:
            color = COLOR_AZUL
        elif fila["conjunto"] == "B":
            color = COLOR_TEXTO_SUAVE
        else:
            color = "#B5B5B5"
        eje.errorbar(
            valor["media"], posicion,
            xerr=[[valor["media"] - valor["ic_inf"]], [valor["ic_sup"] - valor["media"]]],
            fmt="o", color=color, ecolor=color, elinewidth=1.6, capsize=3, markersize=7 if es_persistencia else 6, zorder=3,
        )
        eje.text(valor["ic_sup"] + 0.015, posicion, formatear_numero(valor["media"], 2), va="center", fontsize=8, color=COLOR_TEXTO)
    eje.set_yticks(range(len(filas)), labels=[etiqueta_modelo_riesgo(f) for f in filas])
    eje.set_xlim(0, 1.08)
    eje.xaxis.set_major_formatter(formateador_numeros(1))
    eje.grid(axis="y", visible=False)
    eje.invert_yaxis()
    eje.set_title(CONFIG["riesgo"]["nombres_metricas"][metrica])


def figura_metricas_riesgo(riesgo: dict) -> str:
    """F1 y AUC con IC 95 % de cada modelo, ordenados por F1, frente a la persistencia."""
    principal = riesgo["decision"]["metrica"]
    # La clase mayoritaria no usa variables: sus resultados con A y B son idénticos, se muestra una vez.
    filas = [f for f in riesgo["metricas"] if not (f["modelo"] == "mayoritaria" and f["conjunto"] != "A")]
    filas = sorted(filas, key=lambda f: f[principal]["media"], reverse=True)
    persistencia = next(f for f in filas if f["modelo"] == "persistencia")
    fig, ejes = plt.subplots(1, 2, figsize=(ESTILO["tamano_base"][0] * 1.5, ESTILO["tamano_base"][1] * 1.2), sharey=True)
    panel_intervalos(ejes[0], filas, principal, persistencia)
    panel_intervalos(ejes[1], filas, "roc_auc", persistencia)
    fig.tight_layout()
    validacion = riesgo["validacion"]
    nota = (
        f"Validación cruzada estratificada {validacion['pliegues']} × {validacion['repeticiones']}; "
        f"IC {formatear_numero(validacion['nivel_confianza'] * 100, 0)} % {validacion['metodo_ic']}. "
        "Banda azul: IC de la persistencia. A = tamaño + departamento; B = A + letalidad y fallecidos de la SE de referencia."
    )
    agregar_fuente(fig, nota)
    return guardar_figura(fig, "riesgo_metricas")


def panel_confusion(eje, matriz: dict, titulo: str) -> None:
    """Matriz de confusión media (en municipios) con escala de un solo tono."""
    valores = np.array([
        [matriz["verdaderos_negativos"], matriz["falsos_positivos"]],
        [matriz["falsos_negativos"], matriz["verdaderos_positivos"]],
    ])
    eje.imshow(valores, cmap="Blues", vmin=0, vmax=valores.sum())
    for fila in range(2):
        for columna in range(2):
            valor = valores[fila, columna]
            color = "white" if valor > valores.sum() * 0.4 else COLOR_TEXTO
            eje.text(columna, fila, formatear_numero(valor, 1), ha="center", va="center", fontsize=12, color=color)
    eje.set_xticks([0, 1], labels=["Baja", "Alta"])
    eje.set_yticks([0, 1], labels=["Baja", "Alta"])
    eje.set_xlabel("Predicción")
    eje.set_ylabel("Letalidad real")
    eje.grid(False)
    for borde in eje.spines.values():
        borde.set_visible(False)
    eje.set_title(titulo)


def figura_confusion(riesgo: dict) -> str:
    """Matrices de confusión medias por repetición de la persistencia y del mejor modelo."""
    mejor = riesgo["mejor_modelo"]
    clave_mejor = f"{mejor['modelo']}_{mejor['conjunto']}"
    matrices = riesgo["matrices_confusion"]
    fig, ejes = nueva_figura(1, 2, escala=(1.2, 1.05))
    panel_confusion(ejes[0], matrices["persistencia"], "Persistencia")
    panel_confusion(ejes[1], matrices[clave_mejor], f"{mejor['nombre']} · {mejor['conjunto']}")
    fig.tight_layout()
    total = formatear_numero(matrices["persistencia"]["total"], 0)
    agregar_fuente(fig, f"Promedio de {riesgo['validacion']['repeticiones']} repeticiones; cada repetición suma sus pliegues ({total} municipios).")
    return guardar_figura(fig, "riesgo_confusion")


def figura_importancia(riesgo: dict) -> str:
    """Importancia por permutación (caída del AUC en pliegues de prueba) con rango 5–95 %."""
    tabla = pd.DataFrame(riesgo["importancia"]).iloc[::-1]
    fig, eje = nueva_figura(1, 1, escala=(1.0, 0.9))
    posiciones = np.arange(len(tabla))
    eje.barh(posiciones, tabla["media"], color=COLOR_AZUL, height=0.55)
    eje.errorbar(
        tabla["media"], posiciones,
        xerr=[tabla["media"] - tabla["p05"], tabla["p95"] - tabla["media"]],
        fmt="none", ecolor=COLOR_TEXTO_SUAVE, elinewidth=1.2, capsize=3,
    )
    for posicion, valor in zip(posiciones, tabla["media"]):
        eje.text(max(valor, 0), posicion + 0.32, formatear_numero(valor, 3), fontsize=8, color=COLOR_TEXTO)
    eje.axvline(0, color=COLOR_TEXTO_SUAVE, lw=0.8)
    eje.set_yticks(posiciones, labels=tabla["nombre"])
    eje.xaxis.set_major_formatter(formateador_numeros(None))
    eje.set_xlabel("Caída media del AUC al permutar la variable")
    eje.grid(axis="y", visible=False)
    detalle = riesgo["importancia_modelo"]
    nombre = CONFIG["riesgo"]["nombres_modelos"][detalle["modelo"]]
    eje.set_title(f"Importancia por permutación · {nombre} {detalle['conjunto']}")
    fig.tight_layout()
    agregar_fuente(fig, f"Calculada en pliegues de prueba (validación {riesgo['validacion']['pliegues']} × {detalle['repeticiones_cv']}); barras de error: percentiles 5 y 95.")
    return guardar_figura(fig, "riesgo_importancia")


def figura_persistencia(riesgo: dict) -> str:
    """Letalidad en la semana de referencia frente a la semana final, con el umbral en ambos ejes.

    Los cuadrantes muestran directamente por qué la persistencia funciona: la
    mayoría de los municipios se queda del mismo lado del umbral.
    """
    from src import datos as modulo_datos
    from src import modelo_riesgo

    muestra = modelo_riesgo.construir_muestra(modulo_datos.cargar_municipios())
    descripcion = riesgo["muestra"]
    umbral = descripcion["umbral_letalidad"]
    fig, eje = nueva_figura(1, 1, escala=(0.95, 1.25))
    altos = muestra["objetivo"] == 1
    eje.scatter(muestra.loc[~altos, "letalidad_ref"], muestra.loc[~altos, "letalidad_fin"], s=30, color=COLOR_GRIS, edgecolor="white", linewidth=0.5, label="Letalidad baja en la semana final")
    eje.scatter(muestra.loc[altos, "letalidad_ref"], muestra.loc[altos, "letalidad_fin"], s=30, color=COLOR_AZUL, edgecolor="white", linewidth=0.5, label="Letalidad alta en la semana final")
    eje.axvline(umbral, color=COLOR_TEXTO_SUAVE, lw=1, ls="--")
    eje.axhline(umbral, color=COLOR_TEXTO_SUAVE, lw=1, ls="--")
    limite = max(muestra["letalidad_ref"].max(), muestra["letalidad_fin"].max()) * 1.05
    eje.plot([0, limite], [0, limite], color=COLOR_GRILLA, lw=1, zorder=0)
    # Márgenes libres arriba y abajo para escribir los conteos sin tapar puntos.
    eje.set_xlim(-limite * 0.02, limite)
    eje.set_ylim(-limite * 0.1, limite * 1.1)
    cuadrantes = [
        (descripcion["bajo_en_ambas"], 0.03, 0.01, "left", "bottom"),
        (descripcion["pasan_a_alto"], 0.03, 0.99, "left", "top"),
        (descripcion["alto_en_ambas"], 0.98, 0.99, "right", "top"),
        (descripcion["pasan_a_bajo"], 0.98, 0.01, "right", "bottom"),
    ]
    for cantidad, x, y, horizontal, vertical in cuadrantes:
        eje.text(x, y, f"{cantidad} municipios", transform=eje.transAxes, ha=horizontal, va=vertical, fontsize=9,
                 color=COLOR_TEXTO, fontweight="bold", zorder=5,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "boxstyle": "round,pad=0.2"})
    eje.set_xlabel(f"Letalidad en la SE {descripcion['semana_referencia']} (%)")
    eje.set_ylabel(f"Letalidad en la SE {descripcion['semana_fin']} (%)")
    eje.xaxis.set_major_formatter(formateador_numeros(None))
    eje.yaxis.set_major_formatter(formateador_numeros(None))
    eje.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8, ncol=1)
    eje.set_title(f"¿Se mantiene la letalidad? Umbral = {formatear_numero(umbral, 1)} %")
    fig.tight_layout()
    agregar_fuente(fig, f"{fuente_municipal()} {descripcion['municipios']} municipios con {descripcion['min_casos']} casos o más en la SE {descripcion['semana_fin']}.")
    return guardar_figura(fig, "riesgo_persistencia")


def generar_figuras_riesgo() -> list[str]:
    """Dibuja todas las figuras del Modelo 3."""
    riesgo = leer_json("riesgo")
    archivos = [
        figura_metricas_riesgo(riesgo),
        figura_confusion(riesgo),
        figura_importancia(riesgo),
        figura_persistencia(riesgo),
    ]
    return archivos


def generar_todas() -> list[str]:
    """Dibuja todas las figuras disponibles del proyecto.

    Cada bloque se dibuja solo si su JSON existe, para poder generar las
    figuras de un módulo sin haber ejecutado los siguientes.
    """
    aplicar_estilo()
    archivos = generar_figuras_eda()
    if (RUTAS.resultados / "perfiles.json").exists():
        archivos += generar_figuras_perfiles()
    if (RUTAS.resultados / "series.json").exists():
        archivos += generar_figuras_series()
    if (RUTAS.resultados / "riesgo.json").exists():
        archivos += generar_figuras_riesgo()
    return archivos


if __name__ == "__main__":
    for archivo_generado in generar_todas():
        print(archivo_generado)
