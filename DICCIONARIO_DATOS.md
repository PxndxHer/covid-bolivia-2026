# Diccionario de datos

Este documento describe cada columna que usa el estudio: su significado, su unidad, de dónde
viene y bajo qué licencia se distribuye. Los archivos originales están en `datos/crudos/` y
**no se modifican nunca**; los scripts escriben las versiones limpias en `datos/procesados/`.

| Archivo | Fuente | Cobertura | Fecha de corte | Licencia |
|---|---|---|---|---|
| `mun_covid_se30.csv` | Reportes epidemiológicos municipales de Bolivia *(institución y URL por confirmar, ver nota)* | 256 municipios, semanas epidemiológicas 20 a 30 de 2020 (1 639 filas) | Semana epidemiológica 30 de 2020 | *Por confirmar* |
| `owid-covid-data.csv` | Our World in Data, *COVID-19 dataset* — <https://github.com/owid/covid-19-data/tree/master/public/data> | Todos los países y agregados regionales, datos diarios | 19/09/2020 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

> **Nota sobre la fuente municipal.** El archivo se recibió ya consolidado. Antes de publicar,
> el autor debe completar aquí la institución que publicó los datos, la URL exacta y su licencia.

---

## 1. `mun_covid_se30.csv` — casos acumulados por municipio

Las columnas se renombran **por posición** (ver `columnas_municipios` en `config.yaml`), de modo
que el análisis no depende de cómo vengan escritos los encabezados originales.

Todas las cantidades son **acumuladas** desde el inicio de la epidemia hasta la semana indicada.

**Particularidades conocidas** (las detecta `src/datos.py` y quedan en `resultados/calidad.json`):
- Falta la semana 26. Las figuras interrumpen la línea en lugar de unir las semanas 25 y 27.
- Un municipio aparece recién cuando registra casos, por eso la cantidad de municipios crece de
  93 (SE 20) a 254 (SE 30); en total hay 256 códigos distintos. La excepción son 10 filas de la
  SE 20 con 0 confirmados, cuya letalidad queda indefinida.
- 4 códigos de municipio aparecen con dos grafías distintas; se usa la de la semana más reciente.
- Hay 5 descensos de confirmados y 6 de fallecidos entre semanas consecutivas (correcciones de la
  fuente). Se informan, pero no se modifican.

| # | Nombre en el proyecto | Tipo | Unidad | Descripción |
|---|---|---|---|---|
| 1 | `cod_departamento` | entero | código | Código del departamento (división político-administrativa de primer nivel). |
| 2 | `departamento` | texto | — | Nombre del departamento (9 en Bolivia). |
| 3 | `cod_municipio` | entero | código | Código único del municipio. |
| 4 | `municipio` | texto | — | Nombre del municipio. |
| 5 | `semana` | entero | semana epidemiológica | Semana epidemiológica de 2020 (20 a 30). |
| 6 | `confirmados` | entero | personas | Casos confirmados acumulados. |
| 7 | `activos` | entero | personas | Casos que siguen activos al cierre de la semana. |
| 8 | `fallecidos` | entero | personas | Fallecidos acumulados por COVID-19. |
| 9 | `recuperados` | entero | personas | Recuperados acumulados. |

**Identidad contable que se verifica:** `confirmados = activos + fallecidos + recuperados`.

**Clave única:** (`cod_municipio`, `semana`). Si hay duplicados, el control de calidad se detiene.

### Variables derivadas

| Nombre | Unidad | Definición | Uso |
|---|---|---|---|
| `letalidad` | % | `fallecidos / confirmados × 100` (indefinida si `confirmados = 0`) | EDA y Modelo 3 |
| `log10_confirmados` | log₁₀(personas) | `log10(confirmados)` de la semana de referencia (27) | Modelo 3 |
| `letalidad_alta` | 0/1 | 1 si la letalidad de la semana 30 es ≥ 3,7 % | Objetivo del Modelo 3 |

---

## 2. `owid-covid-data.csv` — Our World in Data

Solo se usan las columnas de la tabla. Se filtran las filas con `date ≤ 2020-09-19`
(`fecha_corte` en `config.yaml`) y se excluyen los agregados cuyo `iso_code` empieza con `OWID_`
(mundo, continentes, grupos de ingreso).

| Columna OWID | Tipo | Unidad | Descripción | Uso |
|---|---|---|---|---|
| `iso_code` | texto | ISO 3166-1 alfa-3 | Código del país. | Filtro de agregados |
| `continent` | texto | — | Continente. | Descriptivo |
| `location` | texto | — | Nombre del país (en inglés). | Selección de países |
| `date` | fecha | AAAA-MM-DD | Fecha del registro. | Series temporales |
| `total_cases` | real | personas | Casos confirmados acumulados. | Letalidad, Modelo 1 |
| `new_cases` | real | personas/día | Casos confirmados nuevos. | Modelo 2, IPS |
| `total_deaths` | real | personas | Fallecidos acumulados. | Letalidad, Modelo 1 |
| `total_cases_per_million` | real | casos por millón de hab. | Casos acumulados por millón. | EDA, Modelo 1 |
| `new_cases_per_million` | real | casos por millón de hab./día | Casos nuevos por millón. | EDA (media móvil) |
| `total_deaths_per_million` | real | fallecidos por millón de hab. | Fallecidos acumulados por millón. | EDA, Modelo 1 |
| `total_tests_per_thousand` | real | pruebas por mil hab. | Pruebas acumuladas; se toma el **último dato disponible** hasta la fecha de corte porque muchos países no informan a diario. | Modelo 1 |
| `stringency_index` | real | índice 0–100 | Índice de rigurosidad de Oxford (OxCGRT); se usa su **media** en el periodo. | Modelo 1 |
| `population` | real | personas | Población (estimación de Naciones Unidas). | IPS |
| `hospital_beds_per_thousand` | real | camas por mil hab. | Camas hospitalarias **totales** (no de terapia intensiva), último año disponible. | Modelo 1, IPS |
| `human_development_index` | real | índice 0–1 | Índice de Desarrollo Humano (PNUD). | Modelo 1 |
| `median_age` | real | años | Edad mediana de la población. | Modelo 1 |

### Variables derivadas

| Nombre | Unidad | Definición | Uso |
|---|---|---|---|
| `letalidad` | % | `total_deaths / total_cases × 100` a la fecha de corte | EDA, Modelo 1 |
| `casos_millon_mm` | casos por millón de hab./día | Media móvil de 7 días de `new_cases_per_million` (calculada en `src/eda.py`) | EDA |
| `casos_semanales` | personas/semana | Suma de `new_cases` en semanas que cierran en sábado | Modelo 2 |
| `camas_totales` | camas | `hospital_beds_per_thousand × population / 1000` | IPS |
| `ips` | casos por cama | Casos nuevos de los últimos 14 días / `camas_totales` | Modelo 2 |

> **Limitación del IPS.** Usa camas hospitalarias totales porque OWID no publica camas de terapia
> intensiva para la mayoría de los países sudamericanos. Por eso mide presión relativa sobre el
> sistema hospitalario, no ocupación real de UTI.

---

## Cita de las fuentes

- Mathieu, E., Ritchie, H., Rodés-Guirao, L., *et al.* (2020). *Coronavirus Pandemic (COVID-19)*.
  Our World in Data. <https://ourworldindata.org/coronavirus>. Licencia CC BY 4.0.
- Hale, T., *et al.* (2021). A global panel database of pandemic policies (Oxford COVID-19
  Government Response Tracker). *Nature Human Behaviour*, 5, 529–538 (origen de `stringency_index`).
