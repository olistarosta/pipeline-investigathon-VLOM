"""Bajar y descomprimir los zips de un release de GitHub en la raíz del repo."""
from __future__ import annotations

import io
import pathlib
import urllib.request
import zipfile

from .config import RAIZ


def descomprimir(contenido: bytes, destino: pathlib.Path = RAIZ) -> None:
    """Descomprime en `destino` (sólo rutas dentro de él)."""
    destino = destino.resolve()
    with zipfile.ZipFile(io.BytesIO(contenido)) as z:
        for info in z.infolist():
            salida = (destino / info.filename).resolve()
            if not salida.is_relative_to(destino):
                raise SystemExit(f"ruta fuera del repo en el zip: {info.filename}")
            if info.is_dir():
                continue
            salida.parent.mkdir(parents=True, exist_ok=True)
            salida.write_bytes(z.read(info))


def bajar(url: str) -> bytes:
    print(f"bajando {url}", flush=True)
    with urllib.request.urlopen(url) as r:
        return r.read()
