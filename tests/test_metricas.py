"""Pruebas de forma y rango de los resultados, y de los valores de aceptación."""

import pytest

from src.config import CONFIG
from tests.conftest import solo_configuracion_por_defecto


# -----------------------------------------------------------------------------
# Análisis exploratorio
# -----------------------------------------------------------------------------
def test_eda_forma(eda):
    assert eda["nacional"][-1]["semana"] == CONFIG["municipios"]["semana_fin"]
    assert 0 <= eda["kruskal"]["p"] <= 1
    assert eda["kruskal"]["gl"] == eda["kruskal"]["grupos"] - 1
    paises = 1 + len(CONFIG["pais"]["comparacion"])
    assert len(eda["internacional"]["indicadores"]) == paises
    for fila in eda["spearman"]["rho"]:
        assert all(-1 <= valor <= 1 for valor in fila)


@solo_configuracion_por_defecto
def test_eda_aceptacion(eda):
    assert eda["kruskal"]["H"] == pytest.approx(26.5, abs=0.05)
    assert eda["kruskal"]["p"] == pytest.approx(0.0002, abs=0.00005)


# -----------------------------------------------------------------------------
# Modelo 1
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("muestra", ["regional", "mundial"])
def test_perfiles_forma(perfiles, muestra):
    resultado = perfiles[muestra]
    assert -1 <= resultado["silueta"] <= 1
    assert -1 <= resultado["ari_ward"] <= 1
    assert resultado["k"] >= resultado["k_minimo_aceptado"]
    assert sum(g["tamano"] for g in resultado["grupos"]) == resultado["n_paises"]
    assert sum(g["contiene_foco"] for g in resultado["grupos"]) == 1
    assert len(resultado["pca"]["varianza_explicada"]) == CONFIG["perfiles"]["n_componentes_pca"]
    assert all(not p.startswith(CONFIG["perfiles"]["prefijo_agregados"]) for p in resultado["paises"])


@solo_configuracion_por_defecto
def test_perfiles_aceptacion(perfiles):
    mundial = perfiles["mundial"]
    assert mundial["n_paises"] == 88
    assert mundial["k"] == 5
    assert mundial["silueta"] == pytest.approx(0.353, abs=0.002)
    assert mundial["ari_ward"] == pytest.approx(0.75, abs=0.01)
    grupo_foco = next(g for g in mundial["grupos"] if g["contiene_foco"])
    assert grupo_foco["tamano"] == 13
    assert mundial["vecinos_foco"][0]["pais"] == "Colombia"


# -----------------------------------------------------------------------------
# Modelo 2
# -----------------------------------------------------------------------------
def test_series_forma(series):
    horizonte = CONFIG["series"]["horizonte"]
    n = series["serie"]["n_semanas"]
    assert series["cortes"] == n - horizonte - CONFIG["series"]["primer_origen"] + 1
    assert len(series["metricas"]) == 6
    for fila in series["metricas"]:
        assert fila["RMSE"] >= fila["MAE"] > 0
        assert fila["MAPE"] >= 0
        assert fila["MASE"] > 0
        assert fila["R2"] <= 1
        assert fila["n"] == series["cortes"] * horizonte
    metrica = CONFIG["series"]["metrica_seleccion"].upper()
    valores = [fila[metrica] for fila in series["metricas"]]
    assert valores == sorted(valores)
    assert series["mejor_modelo"] == series["metricas"][0]["modelo"]


def test_intervalos_ordenados(series):
    niveles = sorted(CONFIG["series"]["niveles_intervalo"])
    for semana in series["proyeccion"]["semanas"]:
        estrecho, ancho = niveles[0], niveles[-1]
        assert semana[f"inferior_{ancho}"] <= semana[f"inferior_{estrecho}"]
        assert semana[f"superior_{estrecho}"] <= semana[f"superior_{ancho}"]
        assert semana[f"inferior_{ancho}"] >= 0


def test_ips_coherente(series):
    ips = series["ips"]
    escenarios = [e["ips_pico"] for e in ips["escenarios"]]
    assert escenarios == sorted(escenarios, reverse=True)
    foco = next(f for f in ips["paises"] if f["es_foco"])
    assert foco["casos_14d_pico"] / foco["camas"] == pytest.approx(foco["ips_pico"], abs=0.001)
    assert ips["camas_necesarias"] == foco["camas"] + ips["camas_adicionales"]


@solo_configuracion_por_defecto
def test_series_aceptacion(series):
    assert series["serie"]["n_semanas"] == 28
    assert series["serie"]["fecha_pico"] == "2020-08-08"
    assert series["serie"]["casos_pico"] == 11102
    assert series["mejor_modelo"] == "holt"
    # Tolerancia del 1 %: el ajuste de Holt varía levemente con la versión de statsmodels.
    assert series["metricas"][0]["RMSE"] == pytest.approx(2782, rel=0.01)
    foco = next(f for f in series["ips"]["paises"] if f["es_foco"])
    assert foco["ips_pico"] == pytest.approx(1.76, abs=0.005)
    assert foco["fecha_pico"] == "2020-07-28"
    assert foco["camas"] == 12840


# -----------------------------------------------------------------------------
# Modelo 3
# -----------------------------------------------------------------------------
def test_riesgo_metricas_en_rango(riesgo):
    for fila in riesgo["metricas"]:
        for metrica in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            valor = fila[metrica]
            assert 0 <= valor["ic_inf"] <= valor["media"] <= valor["ic_sup"] <= 1
            assert valor["de"] >= 0


def test_matrices_confusion_suman_la_muestra(riesgo):
    total = riesgo["muestra"]["municipios"]
    for matriz in riesgo["matrices_confusion"].values():
        assert matriz["total"] == pytest.approx(total)


def test_mayoritaria_y_persistencia(riesgo):
    mayoritaria = next(f for f in riesgo["metricas"] if f["modelo"] == "mayoritaria")
    assert mayoritaria["roc_auc"]["media"] == pytest.approx(0.5)
    assert mayoritaria["f1"]["media"] == 0
    persistencia = next(f for f in riesgo["metricas"] if f["modelo"] == "persistencia")
    assert persistencia["f1"]["media"] > 0


@solo_configuracion_por_defecto
def test_riesgo_aceptacion(riesgo):
    muestra = riesgo["muestra"]
    assert (muestra["municipios"], muestra["altos"], muestra["bajos"]) == (94, 39, 55)
    bosque = next(f for f in riesgo["metricas"] if f["modelo"] == "bosque" and f["conjunto"] == "B")
    assert bosque["roc_auc"]["media"] == pytest.approx(0.83, abs=0.01)
    assert bosque["f1"]["media"] == pytest.approx(0.76, abs=0.01)
    persistencia = next(f for f in riesgo["metricas"] if f["modelo"] == "persistencia")
    assert persistencia["f1"]["media"] == pytest.approx(0.79, abs=0.01)
    assert persistencia["roc_auc"]["media"] == pytest.approx(0.83, abs=0.01)
    assert riesgo["decision"]["persistencia_principal"]
