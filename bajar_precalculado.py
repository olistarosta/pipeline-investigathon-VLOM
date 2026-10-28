"""Baja lo precalculado (release del repo) y lo deja en `generados/`.

    uv run bajar_precalculado.py                   # lo necesario para predecir.py con los modelos todo y sin_test
    uv run bajar_precalculado.py --entrenamiento   # además, el caché de data/ (para reentrenar sin rearmar el grafo)

El repo es privado: usa `gh` (GitHub CLI) con una cuenta que tenga acceso (`gh auth login`).
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from vlom.bajar import descomprimir
from vlom.release import ETIQUETA, REPO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--entrenamiento", action="store_true")
    a = ap.parse_args()
    partes = ("prediccion", "entrenamiento") if a.entrenamiento else ("prediccion",)
    with tempfile.TemporaryDirectory() as tmp:
        for parte in partes:
            print(f"bajando {parte}_*.zip de {REPO} ({ETIQUETA})", flush=True)
            subprocess.run(["gh", "release", "download", ETIQUETA, "--repo", REPO, "--pattern", f"{parte}_*.zip",
                            "--dir", tmp], check=True)
        for z in sorted(Path(tmp).glob("*.zip")):
            descomprimir(z.read_bytes())
    print("listo: generados/")


if __name__ == "__main__":
    main()
