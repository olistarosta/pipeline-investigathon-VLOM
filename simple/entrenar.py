"""Pipeline simple (limpio): lo entrena desde los datos de `data/`.

    uv run simple/entrenar.py                     # modelo `todo` (train + val + test de data/)
    uv run simple/entrenar.py --modelo train      # sólo train, y lo evalúa en val (2023S2) contra el B0

Pasos (cada uno saltea lo ya calculado en `generados/`, que comparte con el completo): embeddings, caché (grafos,
recorrido, pool y variables) y el LambdaRank de 15 variables. Sin el cross-encoder: sólo los embeddings y el recorrido
del grafo usan GPU (si hay).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pandas as pd  # noqa: E402

from vlom import cache, embeddings, limpio, salida  # noqa: E402
from vlom.config import FUENTE, MODELOS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modelo", choices=list(MODELOS), default="todo")
    a = ap.parse_args()
    t0 = time.time()
    print("== embeddings", flush=True)
    embeddings.todo(FUENTE)
    print("== cache", flush=True)
    cache.todo(FUENTE)
    print("== limpio", flush=True)
    limpio.entrenar(a.modelo)
    if a.modelo == "train":
        R = salida.evaluar_val("limpio")
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
            print("limpio, val (2023S2):")
            print(R.to_string(index=False))
        print(f"promedio de las cinco: MRR {R.MRR.mean():.4f}")
    print(f"listo ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
