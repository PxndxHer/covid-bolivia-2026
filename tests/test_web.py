"""Pruebas de la página generada: contiene los valores de los JSON y no tiene números a mano."""

import re

from src.config import RUTAS
from src.web import PATRON_FIGURA, formatear, formatear_pct


def test_pagina_contiene_valores_eda(pagina, eda):
    final = eda["nacional"][-1]
    assert formatear(final["confirmados"]) in pagina
    assert formatear(final["fallecidos"]) in pagina
    assert formatear_pct(final["letalidad"], 2) in pagina
    assert formatear(eda["kruskal"]["H"], 1) in pagina


def test_pagina_contiene_valores_perfiles(pagina, perfiles):
    mundial = perfiles["mundial"]
    assert formatear(mundial["silueta"], 3) in pagina
    assert formatear(mundial["ari_ward"], 2) in pagina
    for grupo in mundial["grupos"]:
        assert grupo["nombre"] in pagina
    assert mundial["vecinos_foco"][0]["nombre"] in pagina


def test_pagina_contiene_valores_series(pagina, series):
    for fila in series["metricas"]:
        assert formatear(fila["RMSE"]) in pagina
        assert formatear(fila["R2"], 2) in pagina
    foco = next(f for f in series["ips"]["paises"] if f["es_foco"])
    assert formatear(foco["ips_pico"], 2) in pagina
    assert formatear(foco["camas"]) in pagina
    assert formatear(series["ips"]["camas_adicionales"]) in pagina


def test_pagina_contiene_valores_riesgo(pagina, riesgo):
    for fila in riesgo["metricas"]:
        assert formatear(fila["f1"]["media"], 2) in pagina
        assert formatear(fila["roc_auc"]["media"], 2) in pagina
    assert str(riesgo["muestra"]["municipios"]) in pagina


def test_figuras_publicadas_coinciden_con_las_usadas(pagina):
    usadas = set(PATRON_FIGURA.findall(pagina))
    publicadas = {archivo.name for archivo in RUTAS.docs_figuras.glob("*.png")}
    assert usadas == publicadas
    assert (RUTAS.docs / ".nojekyll").exists()


def test_plantilla_sin_numeros_escritos_a_mano():
    """El texto visible de la plantilla no debe contener cifras: todas llegan de los JSON.

    Se quitan las expresiones de Jinja, los bloques de estilo y de script y las
    etiquetas HTML (cuyos atributos sí llevan números de diseño), y se busca
    cualquier dígito en lo que queda. Se permiten solo identificadores que no
    son datos: nombres propios (COVID-19, la licencia CC BY 4.0), el nombre de
    la métrica F1 y la numeración de los modelos.
    """
    permitidos = [r"COVID-19", r"CC BY 4\.0", r"\bF1\b", r"Modelo [123]\b"]
    texto = (RUTAS.plantillas / "index.html.j2").read_text(encoding="utf-8")
    texto = re.sub(r"\{#.*?#\}", " ", texto, flags=re.S)
    texto = re.sub(r"\{\{.*?\}\}", " ", texto, flags=re.S)
    texto = re.sub(r"\{%.*?%\}", " ", texto, flags=re.S)
    texto = re.sub(r"<style>.*?</style>", " ", texto, flags=re.S)
    texto = re.sub(r"<script.*?</script>", " ", texto, flags=re.S)
    texto = re.sub(r"<!--.*?-->", " ", texto, flags=re.S)
    texto = re.sub(r"<[^>]+>", " ", texto)
    for patron in permitidos:
        texto = re.sub(patron, " ", texto)
    encontrados = re.findall(r".{0,30}\d.{0,30}", texto)
    assert not encontrados, f"Números escritos a mano en la plantilla: {encontrados}"
