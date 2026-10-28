"""Qué va en el release de lo precalculado y cómo se empaqueta.

- `prediccion` (lo que necesita `predecir.py`): los embeddings del historial (`bge-small`, Gemma, LSA y su modelo),
  los umbrales y los dos grafos del corte de los datos nuevos (1/7/2024), y los modelos `todo` y `sin_test` de los dos
  pipelines (con los cross-encoders del completo).
- `entrenamiento` (opcional, para reentrenar sin rearmar el grafo): el caché de `data/` (pool, observables y
  variables de cada licitación) y los embeddings de sus licitaciones.
"""
from __future__ import annotations

import zipfile

from .config import GEN, RAIZ

REPO = "olistarosta/pipeline-investigathon-VLOM"
ETIQUETA = "precalculado-v1"
CORTE_NUEVO = "2024-07-01"
TOPE = 1_900_000_000                 # bytes por zip (GitHub acepta hasta 2 GB por archivo)


def archivos(parte: str) -> list:
    e = GEN / "embeddings"
    if parte == "prediccion":
        out = [p for m in ("bge-small", "gemma2", "tfidf-lsa") for p in (e / m).glob("hist*")]
        out += [e / "tfidf-lsa" / "modelo.pkl", GEN / "grafos" / "umbrales_tfidf.json"]
        out += [p for g in ("gemb", "gtf") for p in (GEN / "grafos" / g / CORTE_NUEVO).iterdir()]
        out += [p for pl in ("limpio", "completo") for m in ("todo", "sin_test")
                for p in (GEN / "modelos" / pl / m).rglob("*") if p.is_file()]
    else:
        out = [p for p in (GEN / "data").rglob("*") if p.is_file() and "completo" not in p.parts]
        out += [p for m in ("bge-small", "gemma2", "tfidf-lsa") for p in (e / m / "lic" / "data").iterdir()]
    return sorted(out)


def empaquetar(parte: str) -> list:
    """Zips de hasta TOPE bytes en `generados/release/`, con rutas relativas a la raíz."""
    sal = GEN / "release"
    sal.mkdir(parents=True, exist_ok=True)
    zips, actual, tam = [], None, 0
    for p in archivos(parte):
        n = p.stat().st_size
        if actual is None or tam + n > TOPE:
            if actual is not None:
                actual.close()
            zips.append(sal / f"{parte}_{len(zips) + 1}.zip")
            actual, tam = zipfile.ZipFile(zips[-1], "w", zipfile.ZIP_STORED), 0
        actual.write(p, p.relative_to(RAIZ).as_posix())
        tam += n
    actual.close()
    return zips
