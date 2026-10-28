"""Entrena los dos pipelines desde los datos de `data/`.

    uv run entrenar.py                          # todo, modelo `todo` (train + val + test)
    uv run entrenar.py --modelo sin_test        # train + val
    uv run entrenar.py --modelo train           # sólo train, y lo evalúa en val (2023S2) contra el B0
    uv run entrenar.py cache                    # un paso: embeddings, cache, limpio, completo o evaluar

Pasos (cada uno saltea lo que ya está calculado en `generados/`):
1. `embeddings`: `bge-small` y EmbeddingGemma 2 del historial y de las licitaciones, y la LSA (horas sin GPU).
2. `cache`: los grafos de cada mes (2022-02 … 2024-06), el recorrido de cada licitación, su pool y sus variables
   (horas; con GPU, ~1–2 h). No depende del modelo: se calcula una vez para los tres.
3. `limpio`: el LambdaRank de 15 variables (~1 min).
4. `completo`: nombres, primera etapa, evidencia, cross-encoder (un fold por semestre del entrenamiento, ~15 min cada
   uno con GPU) y el LambdaRank de 22 variables.
5. `evaluar`: sólo con `--modelo train`, MRR en val y Δ contra el B0.

Los modelos quedan en `generados/modelos/<pipeline>/<modelo>/`; `predecir.py` los usa.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from vlom import cache, completo, embeddings, limpio, salida
from vlom.config import FUENTE, MODELOS

PASOS = ["embeddings", "cache", "limpio", "completo", "evaluar"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paso", nargs="?", choices=["todo"] + PASOS, default="todo")
    ap.add_argument("--modelo", choices=list(MODELOS), default="todo")
    a = ap.parse_args()
    if a.paso == "evaluar" and a.modelo != "train":
        raise SystemExit("sólo el modelo train se evalúa en val (los otros tienen val en su entrenamiento)")
    t0 = time.time()
    pasos = [p for p in PASOS if p != "evaluar" or a.modelo == "train"] if a.paso == "todo" else [a.paso]
    for p in pasos:
        print(f"== {p}", flush=True)
        if p == "embeddings":
            embeddings.todo(FUENTE)
        elif p == "cache":
            cache.todo(FUENTE)
        elif p == "limpio":
            limpio.entrenar(a.modelo)
        elif p == "completo":
            completo.preparar(a.modelo)
            completo.entrenar(a.modelo)
        else:
            with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format",
                                   "{:.3f}".format):
                for pipeline in ("limpio", "completo"):
                    R = salida.evaluar_val(pipeline)
                    print(f"{pipeline}, val (2023S2):")
                    print(R.to_string(index=False))
                    print(f"promedio de las cinco: MRR {R.MRR.mean():.4f}\n")
    print(f"listo ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
