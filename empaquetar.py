"""Arma los zips del release de lo precalculado (después de `entrenar.py --modelo todo` y `--modelo sin_test`).

    uv run empaquetar.py                   # generados/release/prediccion_<k>.zip
    uv run empaquetar.py --entrenamiento   # además, entrenamiento_<k>.zip
    gh release create precalculado-v1 generados/release/*.zip --repo <repo> --title "..." --notes "..."
"""
from __future__ import annotations

import argparse

from vlom.cache import grafos_corte
from vlom.release import CORTE_NUEVO, empaquetar


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--entrenamiento", action="store_true")
    a = ap.parse_args()
    grafos_corte(CORTE_NUEVO)                  # los grafos con todo el historial, para los datos nuevos
    for parte in ("prediccion", "entrenamiento") if a.entrenamiento else ("prediccion",):
        for z in empaquetar(parte):
            print(f"  {z}")


if __name__ == "__main__":
    main()
