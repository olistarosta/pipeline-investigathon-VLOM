"""Baja lo precalculado (release `precalculado-v1` de este repo) y lo deja en `generados/`.

    uv run bajar_precalculado.py                   # lo necesario para predecir (embeddings, grafos, modelos) y lo ya
                                                   # calculado para el test de los organizadores
    uv run bajar_precalculado.py --sin-test        # sin lo del test: predecir.py lo recalcula (GPU recomendada)
    uv run bajar_precalculado.py --entrenamiento   # además, el caché de data/ (para reentrenar sin rearmar el grafo)

Si el repo es público, baja por HTTPS sin nada más. Si es privado, usa `gh` (GitHub CLI) con una cuenta que tenga
acceso (`gh auth login`).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from vlom.bajar import bajar_a, descomprimir
from vlom.release import ETIQUETA, REPO


def assets_publicos() -> dict[str, str] | None:
    """{nombre: url} de los archivos del release si el repo es público; None si no (privado o sin acceso)."""
    try:
        with urllib.request.urlopen(f"https://api.github.com/repos/{REPO}/releases/tags/{ETIQUETA}") as r:
            return {a["name"]: a["browser_download_url"] for a in json.load(r)["assets"]}
    except (urllib.error.URLError, KeyError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sin-test", action="store_true")
    ap.add_argument("--entrenamiento", action="store_true")
    a = ap.parse_args()
    partes = ["prediccion"] + ([] if a.sin_test else ["test"]) + (["entrenamiento"] if a.entrenamiento else [])
    publicos = assets_publicos()
    with tempfile.TemporaryDirectory() as tmp:
        for parte in partes:
            if publicos is not None:
                for nombre in sorted(n for n in publicos if n.startswith(f"{parte}_") and n.endswith(".zip")):
                    bajar_a(publicos[nombre], Path(tmp) / nombre)
            else:
                print(f"bajando {parte}_*.zip de {REPO} ({ETIQUETA}) con gh", flush=True)
                subprocess.run(["gh", "release", "download", ETIQUETA, "--repo", REPO, "--pattern", f"{parte}_*.zip",
                                "--dir", tmp], check=True)
            for z in sorted(Path(tmp).glob(f"{parte}_*.zip")):
                print(f"descomprimiendo {z.name}", flush=True)
                descomprimir(z)
                z.unlink()
    print("listo: generados/")


if __name__ == "__main__":
    main()
