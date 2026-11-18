"""Genera la página web de resultados a partir de los JSON de `resultados/`.

La plantilla `plantillas/index.html.j2` no contiene ningún número escrito a
mano: todos los valores llegan desde aquí, formateados en español. La salida
es `docs/index.html` más las figuras que la página usa (y solo esas), lista
para publicarse con GitHub Pages.
"""

import re
import shutil

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.config import CONFIG, RUTAS, leer_json, ruta_proyecto

MODULOS = ["calidad", "eda", "perfiles", "series", "riesgo"]
PATRON_FIGURA = re.compile(r'figuras/([\w\-]+\.png)')


# -----------------------------------------------------------------------------
# Formato en español (filtros de Jinja2)
# -----------------------------------------------------------------------------
def formatear(valor: float, decimales: int = 0) -> str:
    """Número con espacio duro como separador de miles y coma decimal.

    El espacio duro (U+00A0) impide que el navegador corte un número en dos
    líneas. Las pruebas usan esta misma función para buscar los valores en la
    página generada.
    """
    if valor is None:
        return "—"
    valor = round(float(valor), decimales)
    if valor == 0:
        valor = 0.0
    texto = f"{valor:,.{decimales}f}"
    texto = texto.replace(",", "\u0000")
    texto = texto.replace(".", CONFIG["figuras"]["separador_decimal"])
    return texto.replace("\u0000", " ")


def formatear_pct(valor: float, decimales: int = 1) -> str:
    """Porcentaje en español: "5,8 %"."""
    return f"{formatear(valor, decimales)} %"


def formatear_fecha(texto: str) -> str:
    """Fecha ISO a formato dd/mm/aaaa."""
    return pd.Timestamp(texto).strftime("%d/%m/%Y")


def formatear_p(valor: float, decimales: int = 4) -> str:
    """Valor p; si es menor que la precisión mostrada se escribe con "<"."""
    minimo = 10 ** (-decimales)
    if valor < minimo:
        return f"p < {formatear(minimo, decimales)}"
    return f"p = {formatear(valor, decimales)}"


def nombre_pais(pais: str) -> str:
    """Nombre del país en español según config."""
    return CONFIG["pais"]["nombres_es"].get(pais, pais)


def nombre_departamento(codigo: str) -> str:
    """Nombre del departamento con tildes según config."""
    return CONFIG["municipios"]["nombres_departamento"].get(codigo, codigo.title())


def crear_entorno() -> Environment:
    """Entorno de Jinja2 con los filtros de formato.

    `StrictUndefined` hace fallar la generación si la plantilla pide un dato
    que no existe, en vez de dejar un hueco silencioso en la página.
    """
    entorno = Environment(
        loader=FileSystemLoader(RUTAS.plantillas),
        undefined=StrictUndefined,
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    entorno.filters["num"] = formatear
    entorno.filters["pct"] = formatear_pct
    entorno.filters["fecha"] = formatear_fecha
    entorno.filters["valor_p"] = formatear_p
    entorno.filters["pais"] = nombre_pais
    entorno.filters["dep"] = nombre_departamento
    return entorno


# -----------------------------------------------------------------------------
# Datos derivados para la página
# -----------------------------------------------------------------------------
def puesto(filas: list[dict], clave: str) -> int:
    """Puesto del país foco (1 = valor más alto) en una lista de países."""
    ordenadas = sorted(filas, key=lambda fila: fila[clave], reverse=True)
    return next(i for i, fila in enumerate(ordenadas, start=1) if fila["es_foco"])


def resumen_perfiles(perfiles: dict) -> dict:
    """Grupo del país foco, sus rasgos más extremos y avisos sobre la muestra regional."""
    mundial = perfiles["mundial"]
    grupo = next(g for g in mundial["grupos"] if g["contiene_foco"])
    z = mundial["foco"]["z"]
    nombres = {v["columna"]: v["nombre"] for v in perfiles["variables"]}
    extremos = sorted(z.items(), key=lambda par: abs(par[1]), reverse=True)[:3]
    nucleo = [p for p, v in mundial["coasignacion_foco"].items() if v >= 0.95]
    limite = [p for p, v in mundial["coasignacion_foco"].items() if 0.05 < v < 0.95]
    return {
        "grupo": grupo,
        "extremos": [{"nombre": nombres[c], "z": v} for c, v in extremos],
        "nucleo": nucleo,
        "limite": [{"pais": p, "proporcion": mundial["coasignacion_foco"][p]} for p in limite],
        "regional_debil": perfiles["regional"]["silueta"] < CONFIG["web"]["umbral_silueta_debil"],
        "umbral_silueta_debil": CONFIG["web"]["umbral_silueta_debil"],
    }


def resumen_series(series: dict) -> dict:
    """Métricas del mejor modelo y del ingenuo, avisos y datos del IPS."""
    metricas = {fila["modelo"]: fila for fila in series["metricas"]}
    mejor = metricas[series["mejor_modelo"]]
    referencia = CONFIG["series"]["modelo_referencia_mase"]
    rmse_h = series["rmse_por_horizonte"]
    horizontes_peor = [
        h + 1 for h, (a, b) in enumerate(zip(rmse_h[series["mejor_modelo"]], rmse_h[referencia])) if a > b
    ]
    ips = series["ips"]
    foco = next(f for f in ips["paises"] if f["es_foco"])
    return {
        "mejor": mejor,
        "ingenuo": metricas[referencia],
        "todos_r2_negativos": all(f["R2"] < 0 for f in series["metricas"]),
        "mejora_ingenuo_pct": (1 - mejor["MAE_relativo_ingenuo"]) * 100,
        "horizontes_peor_que_ingenuo": horizontes_peor,
        "ips_foco": foco,
        "ips_puesto": puesto(ips["paises"], "ips_pico"),
        "ips_paises": sorted(ips["paises"], key=lambda f: f["ips_pico"], reverse=True),
        "ultima_semana": series["proyeccion"]["semanas"][-1],
    }


def resumen_riesgo(riesgo: dict) -> dict:
    """Filas de la tabla de métricas, persistencia, mejor modelo y transiciones."""
    metricas = riesgo["metricas"]
    persistencia = next(f for f in metricas if f["modelo"] == "persistencia")
    mejor_clave = riesgo["mejor_modelo"]
    mejor = next(f for f in metricas if f["modelo"] == mejor_clave["modelo"] and f["conjunto"] == mejor_clave["conjunto"])
    principal = riesgo["decision"]["metrica"]
    tabla = [f for f in metricas if not (f["modelo"] == "mayoritaria" and f["conjunto"] != "A")]
    tabla = sorted(tabla, key=lambda f: f[principal]["media"], reverse=True)
    muestra = riesgo["muestra"]
    return {
        "persistencia": persistencia,
        "mejor": mejor,
        "tabla": tabla,
        "estables": muestra["alto_en_ambas"] + muestra["bajo_en_ambas"],
        "importancia_principal": riesgo["importancia"][0],
    }


def datos_simulador(series: dict) -> dict:
    """Valores que necesita el simulador de camas (se insertan como JSON en la página)."""
    ips = series["ips"]
    foco = next(f for f in ips["paises"] if f["es_foco"])
    return {
        "casos14": foco["casos_14d_pico"],
        "camas": foco["camas"],
        "mediana": ips["mediana_ips_pico_otros"],
        "maxPct": CONFIG["web"]["simulador"]["max_pct"],
        "pasoPct": CONFIG["web"]["simulador"]["paso_pct"],
        "pais": nombre_pais(foco["pais"]),
        "otros": [
            {"nombre": f["nombre"], "ips": f["ips_pico"]}
            for f in ips["paises"] if not f["es_foco"]
        ],
    }


def hallazgos(contexto: dict) -> list[dict]:
    """Los cuatro hallazgos clave del resumen, redactados con los valores calculados."""
    eda = contexto["eda"]
    pais = nombre_pais(CONFIG["pais"]["nombre"])
    indicadores = eda["internacional"]["indicadores"]
    foco = next(f for f in indicadores if f["es_foco"])
    puestos = eda["internacional"]["puesto_foco"]
    perfiles = contexto["perfiles_resumen"]
    vecino = contexto["perfiles"]["mundial"]["vecinos_foco"][0]
    series = contexto["series_resumen"]
    ips = contexto["series"]["ips"]
    riesgo = contexto["riesgo_resumen"]
    muestra = contexto["riesgo"]["muestra"]
    total = puestos["total_paises"]
    # Los títulos dependen de los resultados, para seguir siendo ciertos con otro país o periodo.
    subregistro = puestos["muertes_por_millon"] < puestos["casos_por_millon"]
    if subregistro:
        titulo_mortalidad = "Mortalidad alta con pocos casos detectados"
    else:
        titulo_mortalidad = "Mortalidad acorde con los casos detectados"
    ips_alto = series["ips_puesto"] <= total / 2
    titulo_ips = "Presión hospitalaria entre las más altas" if ips_alto else "Presión hospitalaria moderada"
    if contexto["riesgo"]["decision"]["persistencia_principal"]:
        titulo_riesgo = "La letalidad municipal se mantiene"
        cierre_riesgo = "ningún modelo supera a esa regla"
    else:
        titulo_riesgo = "Los modelos superan a la persistencia"
        cierre_riesgo = "el mejor modelo supera a esa regla"
    return [
        {
            "cifra": formatear(foco["muertes_por_millon"]),
            "unidad": "fallecidos por millón",
            "titulo": titulo_mortalidad,
            "subregistro": subregistro,
            "texto": (
                f"Al {formatear_fecha(eda['internacional']['fecha_corte'])}, {pais} ocupaba el puesto "
                f"{puestos['muertes_por_millon']}.º de {total} en fallecidos por millón y el {puestos['letalidad']}.º "
                f"en letalidad ({formatear_pct(foco['letalidad'])}), pero el {puestos['casos_por_millon']}.º en casos por millón."
            ),
        },
        {
            "cifra": f"G{perfiles['grupo']['grupo']}",
            "unidad": f"grupo de {perfiles['grupo']['tamano']} países",
            "titulo": perfiles["grupo"]["nombre"],
            "texto": (
                f"Entre {contexto['perfiles']['mundial']['n_paises']} países, k-means ubica a {pais} junto a "
                f"{len(perfiles['nucleo'])} países que lo acompañan en casi todas las corridas; "
                f"el más parecido es {vecino['nombre']}."
            ),
        },
        {
            "cifra": formatear(series["ips_foco"]["ips_pico"], 2),
            "unidad": "casos de 14 días por cama",
            "titulo": titulo_ips,
            "texto": (
                f"El pico del índice de presión sanitaria ({formatear_fecha(series['ips_foco']['fecha_pico'])}) fue el "
                f"{series['ips_puesto']}.º de {total} países. Igualar la mediana regional habría requerido "
                f"{formatear(ips['camas_adicionales'])} camas más (+{formatear_pct(ips['incremento_necesario_pct'])})."
            ),
        },
        {
            "cifra": formatear(riesgo["persistencia"]["f1"]["media"], 2),
            "unidad": "F1 de la persistencia",
            "titulo": titulo_riesgo,
            "texto": (
                f"{riesgo['estables']} de {muestra['municipios']} municipios quedaron del mismo lado del umbral entre "
                f"la semana {muestra['semana_referencia']} y la {muestra['semana_fin']}; {cierre_riesgo} "
                f"(mejor F1: {formatear(riesgo['mejor']['f1']['media'], 2)})."
            ),
        },
    ]


def construir_contexto() -> dict:
    """Reúne los JSON, los resúmenes derivados y los parámetros que usa la plantilla."""
    contexto = {nombre: leer_json(nombre) for nombre in MODULOS}
    contexto["config"] = CONFIG
    contexto["pais"] = nombre_pais(CONFIG["pais"]["nombre"])
    contexto["perfiles_resumen"] = resumen_perfiles(contexto["perfiles"])
    contexto["series_resumen"] = resumen_series(contexto["series"])
    contexto["riesgo_resumen"] = resumen_riesgo(contexto["riesgo"])
    contexto["simulador"] = datos_simulador(contexto["series"])
    contexto["foco_indicadores"] = next(f for f in contexto["eda"]["internacional"]["indicadores"] if f["es_foco"])
    contexto["hallazgos"] = hallazgos(contexto)
    return contexto


# -----------------------------------------------------------------------------
# Escritura
# -----------------------------------------------------------------------------
def copiar_figuras(html: str) -> list[str]:
    """Copia a `docs/figuras/` solo las figuras citadas en la página y borra las demás.

    Así la carpeta publicada nunca acumula figuras viejas que ya no se usan.
    Falla si la página cita una figura que no existe.
    """
    usadas = sorted(set(PATRON_FIGURA.findall(html)))
    RUTAS.docs_figuras.mkdir(parents=True, exist_ok=True)
    for archivo in RUTAS.docs_figuras.glob("*.png"):
        if archivo.name not in usadas:
            archivo.unlink()
    for nombre in usadas:
        origen = RUTAS.figuras / nombre
        if not origen.exists():
            raise FileNotFoundError(f"La página usa {nombre}, pero no existe en {RUTAS.figuras}. Ejecute src.figuras.")
        shutil.copy2(origen, RUTAS.docs_figuras / nombre)
    return usadas


def ejecutar() -> dict:
    """Renderiza la plantilla, escribe `docs/index.html`, copia figuras y crea `.nojekyll`."""
    entorno = crear_entorno()
    plantilla = entorno.get_template(CONFIG["web"]["plantilla"])
    html = plantilla.render(**construir_contexto())
    salida = ruta_proyecto(CONFIG["web"]["salida"])
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(html, encoding="utf-8", newline="\n")
    figuras = copiar_figuras(html)
    # .nojekyll evita que GitHub Pages procese la carpeta con Jekyll.
    (RUTAS.docs / ".nojekyll").write_text("", encoding="utf-8")
    return {"salida": str(salida), "figuras": figuras}


if __name__ == "__main__":
    resultado = ejecutar()
    print(f"Página: {resultado['salida']} ({len(resultado['figuras'])} figuras)")
