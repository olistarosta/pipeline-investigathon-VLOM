# Pipeline simple (limpio)

La versión simple del pipeline del grupo VLOM: el grafo conjunto de oficinas y empresas, el parecido de texto y el
historial (los 5 niveles del B0), ordenados por un LambdaRank de 15 variables. **No es el pipeline final**: el final
es el completo (`predecir.py` en la raíz), que usa este ranking como primera etapa y le suma un cross-encoder afinado,
EmbeddingGemma 2 y las empresas nombradas en la licitación.

| MRR, promedio de las cinco verticales | simple | completo (final) | B0 |
|---|---|---|---|
| nuestro test (2024S1, entrenado con train + val) | 0,315 | 0,342 | 0,120 |
| val (2023S2, entrenado con train) | 0,350 | 0,378 | 0,137 |

Usa el mismo código (`vlom/limpio.py`, con `vlom/cache.py`, `vlom/grafo.py` y `vlom/recorrido.py`) y lo mismo
precalculado que el completo. Desde la raíz del repo:

```
uv sync
uv run bajar_datos.py
uv run bajar_test.py
uv run bajar_precalculado.py
uv run simple/predecir.py                  # -> predicciones/test/limpio_todo/{rd,j,d,r,it}.csv
```

Para entrenarlo: `uv run simple/entrenar.py` (modelo `todo`) o `uv run simple/entrenar.py --modelo train` (y lo evalúa
en val contra el B0). Las variables, el pool y los parámetros están en `vlom/config.py` (`VARIABLES_LIMPIO`, `POOL`,
`LAMBDARANK`).
