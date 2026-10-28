"""Baja los datos del task (release `datos-v1` del repo de los organizadores) y los deja, sin tocar, en `data/`.

    uv run bajar_datos.py            # deja data/<v>/{historial,licitaciones,ganadores,empresas}.parquet y
                                     # data/{psc,naics,office}.parquet (~175 MB)
    uv run bajar_datos.py --forzar   # volver a bajarlos aunque ya estén
"""
from __future__ import annotations

import argparse

from vlom.bajar import bajar, descomprimir
from vlom.config import DATOS

URL = "https://github.com/bravoivanagustin/ranking-licitaciones/releases/download/datos-v1/data.zip"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--forzar", action="store_true")
    a = ap.parse_args()
    if (DATOS / "j" / "licitaciones.parquet").exists() and not a.forzar:
        print(f"ya están en {DATOS} (--forzar para volver a bajarlos)")
        return
    descomprimir(bajar(URL))             # el zip trae todo bajo data/
    print(f"listo: {DATOS}")


if __name__ == "__main__":
    main()
