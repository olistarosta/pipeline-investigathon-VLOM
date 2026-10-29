"""Bajar y descomprimir los zips de un release de GitHub en la raíz del repo."""
from __future__ import annotations

import io
import pathlib
import shutil
import urllib.request
import zipfile

from .config import RAIZ


def descomprimir(contenido: bytes | pathlib.Path, destino: pathlib.Path = RAIZ) -> None:
    """Descomprime en `destino` (sólo rutas dentro de él). `contenido`: los bytes del zip o su ruta."""
    destino = destino.resolve()
    fuente = io.BytesIO(contenido) if isinstance(contenido, bytes) else contenido
    with zipfile.ZipFile(fuente) as z:
        for info in z.infolist():
            salida = (destino / info.filename).resolve()
            if not salida.is_relative_to(destino):
                raise SystemExit(f"ruta fuera del repo en el zip: {info.filename}")
            if info.is_dir():
                continue
            salida.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(salida, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 24)


def bajar(url: str) -> bytes:
    print(f"bajando {url}", flush=True)
    with urllib.request.urlopen(url) as r:
        return r.read()


def bajar_a(url: str, archivo: pathlib.Path) -> None:
    """Baja `url` a un archivo (zips grandes, sin tenerlos enteros en memoria)."""
    print(f"bajando {url}", flush=True)
    with urllib.request.urlopen(url) as r, open(archivo, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 24)
