# Pipelines finales del grupo VLOM (Investigathon, task Y-Hat)

Para cada licitación pública, un ranking de hasta 100 empresas candidatas por probabilidad de ganarla, usando sólo
información anterior a su fecha. Dos pipelines:

| | **limpio** | **completo** |
|---|---|---|
| qué es | grafo conjunto de oficinas y empresas + texto + historial (los 5 niveles del B0) | el limpio + EmbeddingGemma 2 + empresas nombradas + un cross-encoder afinado |
| variables | 15 | 22 |
| MRR en val (2023S2, modelo `train`, promedio de las cinco) | 0,352 | 0,385 |
| MRR en 2024S1 (modelo `sin_test`) | 0,315 | 0,342 |
| para predecir | CPU, minutos | GPU recomendada (cross-encoder) |

El B0 da 0,137 en val y 0,120 en 2024S1. Detalle del modelo: `vlom/limpio.py` y `vlom/completo.py`.

## Uso rápido: predecir el test

```
uv sync
uv run bajar_datos.py            # data/: los datos del task
uv run bajar_test.py             # test/: el test de los organizadores
uv run bajar_precalculado.py     # generados/: embeddings, grafos y modelos (necesita gh con acceso al repo)
uv run predecir.py               # los dos pipelines, modelo `todo`
```

Escribe `predicciones/test/<pipeline>_<modelo>/{rd,j,d,r,it}.csv`, con el formato de la entrega:

```
id,parent_uei,puesto
d-00001,ABC123DEF456,1
d-00001,GHI789JKL777,2
...
```

Una fila por empresa rankeada, hasta 100 por licitación, puestos desde 1, sólo empresas de `test/<v>/empresas.parquet`.
Cada archivo se valida con `evaluar.validar` (el chequeo de formato de los organizadores) antes de escribirse.

Opciones:

| | |
|---|---|
| `--modelo todo` (por defecto) | modelos entrenados con train + val + test (2022S1–2024S1) |
| `--modelo sin_test` | entrenados con train + val (2022S1–2023S2) |
| `--modelo train` | entrenados sólo con train (el que se evalúa en val; hay que entrenarlo, no está en el release) |
| `--pipeline limpio` / `completo` | uno solo (por defecto, los dos) |
| `--datos <carpeta>` | otras licitaciones nuevas: `<carpeta>/<v>/licitaciones.parquet` y `<carpeta>/<v>/empresas.parquet` |

Qué calcula para las licitaciones nuevas (en `generados/<datos>/`): sus embeddings, el recorrido del grafo desde su
oficina (con los grafos del 1/7/2024, que tienen todo el historial), el pool de candidatas, las variables y, en el
completo, los nombres, la evidencia y el cross-encoder. Una licitación sin ninguna candidata en el pool se rankea con el
B0.

## Entrenar

```
uv run entrenar.py                     # modelo todo: embeddings, caché, limpio y completo
uv run entrenar.py --modelo sin_test
uv run entrenar.py --modelo train      # y lo evalúa en val contra el B0
uv run entrenar.py cache               # un paso: embeddings | cache | limpio | completo | evaluar
```

Cada paso saltea lo que ya está en `generados/`. Lo caro es construir: los embeddings del historial (~1 millón de
textos, con GPU), los grafos de cada mes (2022-02 … 2024-06) y el recorrido de cada licitación, y el cross-encoder (un
fold por semestre de entrenamiento, ~15 min cada uno con GPU). **Hace falta una máquina con GPU y memoria** (los dos
grafos de un corte con sus vectores y el TF-IDF ocupan varios GB). El caché no depende del modelo: se calcula una vez.

Para publicar lo precalculado (el repo es privado: se baja con `gh`, con una cuenta con acceso): `uv run empaquetar.py` (zips en `generados/release/`) y
`gh release create precalculado-v1 generados/release/*.zip --repo olistarosta/pipeline-investigathon-VLOM`. El repo y la etiqueta están en
`vlom/release.py`.

## Estructura

| | |
|---|---|
| `data/` | los datos originales (vacía; `bajar_datos.py`) |
| `test/` | el test de los organizadores (vacía; `bajar_test.py`) |
| `generados/` | lo calculado y lo precalculado (vacía; `bajar_precalculado.py` o `entrenar.py`) |
| `predicciones/` | los rankings de `predecir.py` |
| `b0.py`, `evaluar.py` | del repo de los organizadores, sin cambios |
| `vlom/config.py` | rutas, verticales, modelos y todos los parámetros fijos |
| `vlom/tablas.py` | carga de licitaciones, historial, candidatas y ganadores; split por semestre |
| `vlom/texto.py`, `vlom/embeddings.py` | boilerplate, TF-IDF, embeddings (`bge-small`, Gemma) y LSA |
| `vlom/grafo.py` | construcción y guardado del grafo conjunto |
| `vlom/recorrido.py` | el recorrido desde la oficina (suma sobre caminos, camino más corto; CPU y GPU) y el pool |
| `vlom/cache.py` | pool de texto, observables, variables, Gemma y niveles del B0, por fuente y vertical |
| `vlom/limpio.py`, `vlom/completo.py` | los dos pipelines |
| `vlom/salida.py` | completar con el B0, validar, escribir, evaluar en val |

## El modelo, en corto

1. **Grafo conjunto.** Oficinas y empresas de las cinco verticales, unidas por sus contratos (un grafo por mes, con los
   contratos de fecha anterior al 1.º del mes). Rondas que unen oficinas que le compraron cosas parecidas a la misma
   empresa y empresas que vendieron cosas parecidas a oficinas vecinas. Dos versiones: aristas comparadas con embeddings
   `bge-small` y con TF-IDF (LSA).
2. **Recorrido.** Desde la oficina de la licitación, con el costo de cada arista según su peso y su parecido con el texto
   de la licitación: la masa que llega a cada empresa en K pasos (R = 1) y el camino más probable (R = 3).
3. **Pool.** Top 100 de cada uno de esos cuatro recorridos ∪ top 100 por parecido de texto con sus contratos previos
   (~285 candidatas; el ganador está en el 77 % de las listas de val).
4. **Variables y LambdaRank.** Puestos en el grafo, parecido de texto, fracción de contratos en el PSC y los 5 niveles
   del B0; un LambdaRank de LightGBM para las cinco verticales (la vertical como categórica).
5. **Completo.** Sobre el top 50 del limpio, un cross-encoder (`ms-marco-MiniLM-L12-v2` afinado con nuestras
   licitaciones) lee la licitación contra los 8 contratos previos más parecidos de la empresa; más el parecido con
   EmbeddingGemma 2 y si la licitación nombra a la empresa. LambdaRank de 22 variables.

**Sin fuga:** todo con contratos de fecha < t; TF-IDF, LSA y boilerplate ajustados con licitaciones de train y contratos
anteriores al 1/7/2023; la primera etapa y el cross-encoder, fuera de muestra en las licitaciones de entrenamiento (un
modelo por semestre); los embeddings son de modelos preentrenados.

## Datos nuevos

El test de los organizadores va del 1/7/2024 al 30/6/2025 y el historial termina el 30/6/2024: todas sus licitaciones se
recorren con el mismo grafo (el del 1/7/2024) y se rankean contra la foto de `empresas` (sin `semestre`). El B0 oficial
también corre directo: `uv run b0.py test/<v> --historial data/<v>/historial.parquet --salida b0_<v>.csv`.
