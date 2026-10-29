"""Pipeline simple (limpio): rankea las licitaciones nuevas y escribe un CSV por vertical.

    uv run simple/predecir.py                     # test/, modelo `todo` -> predicciones/test/limpio_todo/<v>.csv
    uv run simple/predecir.py --modelo train      # el modelo entrenado sólo con train
    uv run simple/predecir.py --datos otra_carpeta

Es `predecir.py --pipeline limpio`: el mismo código (`vlom/`), sin el cross-encoder ni Gemma en el ranking.
"""
from __future__ import annotations

import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import predecir  # noqa: E402

if __name__ == "__main__":
    if "--pipeline" in sys.argv:
        raise SystemExit("simple/predecir.py corre sólo el pipeline simple (limpio); para el completo: predecir.py")
    sys.argv = [sys.argv[0], "--pipeline", "limpio", *sys.argv[1:]]
    predecir.main()
