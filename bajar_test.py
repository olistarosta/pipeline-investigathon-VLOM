"""Baja el test de los organizadores (release `test-v1`) y lo deja en `test/`.

    uv run bajar_test.py             # deja test/<v>/{licitaciones,empresas}.parquet
    uv run bajar_test.py --forzar

Licitaciones del 1/7/2024 al 30/6/2025, con id propio (`d-00001`) y sin ganadores; `empresas` es una sola foto al
1/7/2024 (sin `semestre`). Se rankean contra el historial de `data/`, que termina el 30/6/2024.
"""
from __future__ import annotations

import argparse

from vlom.bajar import bajar, descomprimir
from vlom.config import RAIZ

URL = "https://github.com/bravoivanagustin/ranking-licitaciones/releases/download/test-v1/test.zip"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--forzar", action="store_true")
    a = ap.parse_args()
    if (RAIZ / "test" / "j" / "licitaciones.parquet").exists() and not a.forzar:
        print("ya está en test/ (--forzar para volver a bajarlo)")
        return
    descomprimir(bajar(URL))             # el zip trae todo bajo test/
    print(f"listo: {RAIZ / 'test'}")


if __name__ == "__main__":
    main()
