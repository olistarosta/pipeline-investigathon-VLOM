# Pipeline final del grupo VLOM (Investigathon, task Y-Hat)

Para cada licitación pública de EE. UU., un ranking de hasta 100 empresas candidatas por probabilidad de ganarla,
usando sólo información anterior a su fecha de publicación.

**El pipeline final es el completo** (`predecir.py`, `entrenar.py`): un grafo conjunto de oficinas y empresas, texto,
el historial y un cross-encoder afinado, ordenados por un LambdaRank de 22 variables. La versión simple (sin el
cross-encoder ni Gemma, 15 variables) está en [`simple/`](simple/); comparten el código de `vlom/`.

| | **completo (final)** | simple | B0 |
|---|---|---|---|
| MRR en val (2023S2, modelo `train`, promedio de las cinco verticales) | **0,378** | 0,350 | 0,137 |
| MRR en nuestro test (2024S1, modelo `sin_test`, promedio de las cinco) | **0,341** | 0,313 | 0,120 |
| Δ MRR contra el B0 en val, por vertical (IC95) | j +0,206 (+0,183, +0,229) · r +0,214 (+0,178, +0,250) · d +0,388 (+0,324, +0,451) · it +0,241 (+0,200, +0,282) · rd +0,157 (+0,077, +0,243) | j +0,186 · r +0,192 · d +0,342 · it +0,211 · rd +0,133 | — |
| para predecir el test con lo precalculado | < 1 min | < 1 min | |

Tres modelos, según con qué se entrenan: `train` (2022S1–2023S1) se evalúa en val; `sin_test` (2022S1–2023S2), en
nuestro test; `todo` (2022S1–2024S1, todo `data/`) es el que predice el test de los organizadores. Detalle de val por
vertical (Hit@1, Hit@5, Hit@100, S_inc / S_new): `generados/modelos/<pipeline>/train/resumen_val.csv` (viene en el
release).

## Para los jueces: obtener las predicciones del test

**Requisitos:** [uv](https://docs.astral.sh/uv/) (instala Python y las dependencias), y
[GitHub CLI](https://cli.github.com/) (`gh`) con una sesión iniciada (`gh auth login`) en la cuenta invitada al repo:
el repo y su release son privados. Unos 10 GB de disco: el entorno con PyTorch y lo precalculado. No hace falta GPU.

```
gh repo clone olistarosta/pipeline-investigathon-VLOM vlom
cd vlom
uv sync
uv run bajar_datos.py            # data/: los datos del task (release datos-v1 de los organizadores)
uv run bajar_test.py             # test/: el test de los organizadores (release test-v1)
uv run bajar_precalculado.py     # generados/: embeddings, grafos, modelos y lo ya calculado para el test
uv run predecir.py               # el completo, modelo `todo`
```

Tarda unos minutos más lo que tarden las descargas (~5 GB). Escribe `predicciones/test/completo_todo/{rd,j,d,r,it}.csv`,
**idénticos byte a byte a los CSV entregados** (verificado desde un clon nuevo), con el formato de `b0.py`:

```
id,parent_uei,puesto
d-00001,ABC123DEF456,1
d-00001,GHI789JKL777,2
...
```

Una fila por empresa rankeada, hasta 100 por licitación, puestos desde 1, sólo empresas de `test/<v>/empresas.parquet`.
Antes de escribirse, cada archivo pasa por `evaluar.validar` (el chequeo de formato de los organizadores). Los CSV
entregados también están en el release (`predicciones_1.zip`).

- **Acceso**: con el repo privado, `bajar_precalculado.py` baja el release con `gh` y la cuenta iniciada; si el repo
  fuera público, baja por HTTPS sin nada más.
- **Sin lo del test** (`uv run bajar_precalculado.py --sin-test`), `predecir.py` calcula todo para las licitaciones del
  test: embeddings, recorrido del grafo, variables y el cross-encoder (GPU recomendada).
- **Windows**: clonar el repo en una ruta corta (p. ej. `C:\Users\<usuario>\vlom`). Con rutas de más de 260 caracteres,
  Python no carga algunas extensiones de scikit-learn (`ModuleNotFoundError` dentro de `sklearn`). Si `uv sync` falla con
  un error de hardlinks (carpetas de OneDrive), correrlo con la variable de entorno `UV_LINK_MODE=copy`.

Opciones de `predecir.py`:

| | |
|---|---|
| `--modelo todo` (por defecto) | entrenado con train + val + test de `data/` (2022S1–2024S1): el de la entrega |
| `--modelo sin_test` | entrenado con train + val (2022S1–2023S2) |
| `--modelo train` | entrenado sólo con train (2022S1–2023S1): el que se evalúa en val |
| `--pipeline limpio` / `ambos` | el simple, o los dos (por defecto, el completo) |
| `--datos <carpeta>` | otras licitaciones nuevas: `<carpeta>/<v>/licitaciones.parquet` y `<carpeta>/<v>/empresas.parquet` |

Lo que se calcula para unas licitaciones nuevas queda en `generados/<carpeta>/` y se reusa. **Si cambian los datos de
esa carpeta, borrar `generados/<carpeta>/`** (y `generados/embeddings/*/lic/<carpeta>/`) para que se recalcule.

## Entrenar desde cero

Para reproducir la entrega sin lo precalculado, después de `uv sync`, `bajar_datos.py` y `bajar_test.py`:

```
uv run entrenar.py                     # modelo todo (train + val + test de data/): ~5–6 h con GPU
uv run predecir.py                     # -> predicciones/test/completo_todo/
```

Las predicciones salen equivalentes pero no idénticas byte a byte a las entregadas: los embeddings (float16 en GPU) y el
cross-encoder afinado no son deterministas en GPU. En val, dos corridas difieren en ±0,006 de MRR.

EmbeddingGemma 2 (`google/embeddinggemma-2`) puede pedir aceptar su licencia en Hugging Face: en ese caso, exportar
`HF_TOKEN` con un token de una cuenta que la aceptó. Otros modelos y pasos:

```
uv run entrenar.py --modelo train      # sólo train, y lo evalúa en val (2023S2) contra el B0
uv run entrenar.py --modelo sin_test   # train + val (para evaluar en 2024S1)
uv run entrenar.py cache               # un paso: embeddings | cache | limpio | completo | evaluar
```

Cada paso saltea lo que ya está en `generados/`, así que se puede cortar y volver a correr. Con
`uv run bajar_precalculado.py --entrenamiento` se baja el caché de `data/` (pool, recorrido y variables de cada
licitación) y se reentrena sin rearmar los grafos.

Hace falta GPU y memoria. Tiempos en una notebook con RTX 5060 (8 GB) y 32 GB de RAM:

| paso | qué calcula | tiempo |
|---|---|---|
| `embeddings` | `bge-small` y EmbeddingGemma 2 de ~1 millón de textos del historial y de las licitaciones; la LSA | ~45 min |
| `cache` | los dos grafos de cada mes (2022-02 … 2024-06), el recorrido de cada licitación, el pool y las variables | ~3 h |
| `limpio` | el LambdaRank de 15 variables | < 1 min |
| `completo` | nombres, primera etapa, evidencia y un cross-encoder por semestre de entrenamiento (~14 min cada uno) | 45 min (`train`) a 75 min (`todo`) |

El recorrido de los grafos ocupa mucha memoria (~20 GB de RAM y la GPU entera en los últimos meses). Si se pone muy
lento, cortarlo y volver a correr `entrenar.py cache`: retoma desde el último corte terminado.

## El release `precalculado-v1`

| zip | qué tiene | lo baja |
|---|---|---|
| `prediccion_*.zip` | embeddings del historial (`bge-small`, Gemma, LSA y su modelo), los dos grafos del 1/7/2024, los modelos `todo`, `sin_test` y `train` de los dos pipelines (con sus cross-encoders) | `bajar_precalculado.py` |
| `test_*.zip` | todo lo calculado para el test de los organizadores (embeddings, pool, recorrido, variables, cross-encoder) | `bajar_precalculado.py` (salvo `--sin-test`) |
| `entrenamiento_*.zip` | el caché de `data/`: pool, recorrido y variables de cada licitación, y los embeddings de sus licitaciones | `bajar_precalculado.py --entrenamiento` |
| `predicciones_1.zip` | los CSV del test de los organizadores (completo y simple, modelos `todo` y `sin_test`) | a mano |

Se arma con `uv run empaquetar.py` (zips en `generados/release/`); el repo y la etiqueta están en `vlom/release.py`.

## El modelo, en corto

1. **Grafo conjunto.** Oficinas y empresas de las cinco verticales, unidas por sus contratos (un grafo por mes, con los
   contratos de fecha anterior al 1.º del mes). Rondas que unen oficinas que le compraron cosas parecidas a la misma
   empresa y empresas que vendieron cosas parecidas a oficinas vecinas. Dos versiones: aristas comparadas con embeddings
   `bge-small` y con TF-IDF (LSA).
2. **Recorrido.** Desde la oficina de la licitación, con el costo de cada arista según su peso y su parecido con el texto
   de la licitación: la masa que llega a cada empresa en K pasos (R = 1) y el camino más probable (R = 3).
3. **Pool.** Top 100 de cada uno de esos cuatro recorridos ∪ top 100 por parecido de texto con sus contratos previos
   (~285 candidatas; el ganador está en el 67–82 % de las listas de val, según la vertical).
4. **Variables y LambdaRank (simple).** Puestos en el grafo, parecido de texto, fracción de contratos en el PSC y los 5
   niveles del B0; un LambdaRank de LightGBM para las cinco verticales (la vertical como categórica).
5. **Completo (final).** Sobre el top 50 del simple, un cross-encoder (`ms-marco-MiniLM-L12-v2` afinado con nuestras
   licitaciones) lee la licitación contra los 8 contratos previos más parecidos de la empresa; más el parecido con
   EmbeddingGemma 2 y si la licitación nombra a la empresa. LambdaRank de 22 variables.

**Sin fuga:** todo con contratos de fecha < t; TF-IDF, LSA y boilerplate ajustados con licitaciones de train y contratos
anteriores al 1/7/2023; la primera etapa y el cross-encoder, fuera de muestra en las licitaciones de entrenamiento (un
modelo por semestre); los embeddings son de modelos preentrenados. Una licitación sin ninguna candidata en el pool se
rankea con el B0.

**El test de los organizadores** va del 1/7/2024 al 30/6/2025 y el historial termina el 30/6/2024: todas sus
licitaciones se recorren con el mismo grafo (el del 1/7/2024) y se rankean contra la foto de `empresas` (sin
`semestre`). El B0 oficial corre directo: `uv run b0.py test/<v> --historial data/<v>/historial.parquet --salida b0_<v>.csv`.

## Estructura

| | |
|---|---|
| `predecir.py`, `entrenar.py` | el pipeline final (completo): predecir y entrenar |
| `simple/` | el pipeline simple: `simple/predecir.py`, `simple/entrenar.py` |
| `bajar_datos.py`, `bajar_test.py`, `bajar_precalculado.py` | bajan los datos, el test y el release |
| `empaquetar.py` | arma los zips del release |
| `b0.py`, `evaluar.py` | del repo de los organizadores, sin cambios |
| `vlom/config.py` | rutas, verticales, modelos y todos los parámetros fijos |
| `vlom/tablas.py` | carga de licitaciones, historial, candidatas y ganadores; split por semestre |
| `vlom/texto.py`, `vlom/embeddings.py` | boilerplate, TF-IDF, embeddings (`bge-small`, Gemma) y LSA |
| `vlom/grafo.py` | construcción y guardado del grafo conjunto |
| `vlom/recorrido.py` | el recorrido desde la oficina (suma sobre caminos, camino más corto; CPU y GPU) y el pool |
| `vlom/cache.py` | pool de texto, observables, variables, Gemma y niveles del B0, por fuente y vertical |
| `vlom/limpio.py`, `vlom/completo.py` | el simple y el completo |
| `vlom/salida.py` | completar con el B0, validar, escribir, evaluar en val |
| `vlom/release.py`, `vlom/bajar.py` | qué va en el release; bajar y descomprimir |
| `data/`, `test/`, `generados/`, `predicciones/` | datos, test, lo calculado y las predicciones (no van a git) |
