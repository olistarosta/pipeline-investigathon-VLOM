"""Rankea licitaciones nuevas con los dos pipelines y escribe un CSV por vertical (el formato de la entrega).

    uv run predecir.py                         # test/, los dos pipelines, modelo `todo` (train + val + test)
    uv run predecir.py --modelo sin_test       # el modelo entrenado sin 2024S1 (train + val)
    uv run predecir.py --pipeline limpio       # sólo el limpio (sin GPU, en minutos)
    uv run predecir.py --datos otra_carpeta    # otra carpeta con <v>/licitaciones.parquet y <v>/empresas.parquet

Necesita `data/` (bajar_datos.py), la carpeta de datos nuevos (bajar_test.py) y lo precalculado
(bajar_precalculado.py) o entrenado (entrenar.py). Calcula lo que falta de las licitaciones nuevas (embeddings,
recorrido del grafo, variables) en `generados/<datos>/` y escribe `predicciones/<datos>/<pipeline>_<modelo>/<v>.csv`
con columnas `id,parent_uei,puesto` (hasta 100 empresas por licitación, sólo de `<datos>/<v>/empresas.parquet`).
"""
from __future__ import annotations

import argparse
import time

from vlom import cache, completo, embeddings, limpio, salida
from vlom.config import MODELOS, RAIZ


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datos", default="test", help="carpeta (en la raíz) con las licitaciones nuevas")
    ap.add_argument("--modelo", choices=list(MODELOS), default="todo")
    ap.add_argument("--pipeline", choices=["ambos", "limpio", "completo"], default="ambos")
    a = ap.parse_args()
    if not (RAIZ / a.datos / "j" / "licitaciones.parquet").exists():
        raise SystemExit(f"no hay licitaciones en {a.datos}/<v>/ (para el test: uv run bajar_test.py)")
    t0 = time.time()
    print("embeddings de las licitaciones nuevas", flush=True)
    embeddings.todo(a.datos)
    print("pool y variables", flush=True)
    cache.todo(a.datos)
    pipelines = ["limpio", "completo"] if a.pipeline == "ambos" else [a.pipeline]
    for p in pipelines:
        print(f"pipeline {p}, modelo {a.modelo}", flush=True)
        rk = (limpio if p == "limpio" else completo).predecir(a.datos, a.modelo)
        salida.escribir(rk, a.datos, f"{p}_{a.modelo}")
    print(f"listo ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
