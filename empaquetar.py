"""Arma los zips del release de lo precalculado (después de entrenar y de `predecir.py` sobre el test).

    uv run empaquetar.py              # generados/release/{prediccion,test,entrenamiento,predicciones}_<k>.zip
    uv run empaquetar.py prediccion   # sólo una parte
    gh release create precalculado-v1 generados/release/*.zip --repo <repo> --title "..." --notes "..."
"""
from __future__ import annotations

import argparse

from vlom.cache import grafos_corte
from vlom.release import CORTE_NUEVO, PARTES, empaquetar


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("partes", nargs="*", choices=PARTES, default=list(PARTES))
    a = ap.parse_args()
    if "prediccion" in a.partes:
        grafos_corte(CORTE_NUEVO)              # los grafos con todo el historial, para los datos nuevos
    for parte in a.partes:
        for z in empaquetar(parte):
            print(f"  {z}")


if __name__ == "__main__":
    main()
