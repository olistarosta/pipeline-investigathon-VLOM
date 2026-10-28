"""Las tablas de cada vertical: licitaciones de una fuente, historial, candidatas y ganadores.

Una *fuente* es un conjunto de licitaciones a rankear contra el historial de `data/`:
- `data`: las licitaciones de `data/<v>/` (2022S1–2024S1), con sus ganadores. El split sale del semestre
  (`config.SPLIT`): train 2022S1–2023S1, val 2023S2, test 2024S1.
- datos nuevos: una carpeta en la raíz con `<v>/licitaciones.parquet` y `<v>/empresas.parquet` (p. ej. `test/`, el
  test de los organizadores), sin ganadores. Si `empresas` no tiene `semestre` (una sola foto), todas sus filas son
  candidatas de todas las licitaciones; las licitaciones quedan con semestre y split `nuevo`.

El historial es siempre el de `data/<v>/historial.parquet`.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from b0 import semestre, set_aside
from .config import DATOS, FUENTE, NUEVO, RAIZ, SPLIT, VERTICALES


def limpio(s: pd.Series) -> np.ndarray:
    s = s.fillna("").astype(str).str.strip().str.upper()
    return np.where(s.isin(["", "NAN", "NONE"]), "", s).astype(object)


def dias(x) -> np.ndarray:
    return pd.to_datetime(x).values.astype("datetime64[D]").astype(np.int64)


def carpeta(v: str, fuente: str = FUENTE):
    return DATOS / v if fuente == FUENTE else RAIZ / fuente / v


@functools.lru_cache(maxsize=None)
def historial(v: str) -> pd.DataFrame:
    """El historial de la vertical ordenado por fecha (estable), con columnas normalizadas."""
    h = pd.read_parquet(DATOS / v / "historial.parquet")
    h["descripcion"] = h.descripcion.fillna("").astype(str)
    h["ts"] = dias(h.fecha)
    h = h.sort_values("ts", kind="stable").reset_index(drop=True)
    h["office_u"], h["psc_u"], h["naics_u"] = limpio(h.office), limpio(h.psc), limpio(h.naics)
    sa = h.set_aside.map(set_aside)
    h["sa"] = np.where(sa == "NONE", "", sa).astype(object)
    return h


@functools.lru_cache(maxsize=None)
def licitaciones(v: str, fuente: str = FUENTE) -> pd.DataFrame:
    """Las licitaciones de la fuente, ordenadas por fecha e id, con semestre, split y columnas normalizadas."""
    lic = pd.read_parquet(carpeta(v, fuente) / "licitaciones.parquet")
    for c in ("title", "description"):
        lic[c] = lic[c].fillna("").astype(str)
    if fuente == FUENTE:
        lic["semestre"] = lic.t.map(semestre)
        lic["split"] = lic.semestre.map(SPLIT)
    else:
        con_sem = "semestre" in pd.read_parquet(carpeta(v, fuente) / "empresas.parquet").columns
        lic["semestre"] = lic.t.map(semestre) if con_sem else NUEVO
        lic["split"] = NUEVO
    lic["ti"] = dias(lic.t)
    lic["office_u"], lic["psc_u"], lic["naics_u"] = limpio(lic.office), limpio(lic.psc), limpio(lic.naics)
    lic["sa"] = lic.set_aside.map(set_aside)
    return lic.sort_values(["ti", "id"], kind="stable").reset_index(drop=True)


@functools.lru_cache(maxsize=None)
def empresas(v: str, fuente: str = FUENTE) -> pd.DataFrame:
    """Las candidatas de la fuente (`parent_uei`, `semestre`, `name`); sin semestre, con semestre `nuevo`."""
    e = pd.read_parquet(carpeta(v, fuente) / "empresas.parquet")
    if "semestre" not in e:
        e = e.assign(semestre=NUEVO)
    e["name"] = e.name.fillna("").astype(str)
    return e[["parent_uei", "semestre", "name"]].reset_index(drop=True)


def candidatas(v: str, fuente: str = FUENTE) -> dict[str, set]:
    """{semestre: candidatas}."""
    return empresas(v, fuente).groupby("semestre").parent_uei.apply(set).to_dict()


def ganadores(v: str) -> pd.DataFrame:
    return pd.read_parquet(DATOS / v / "ganadores.parquet")[["id", "parent_uei"]]


def fuentes_nuevas() -> list[str]:
    return [p.name for p in RAIZ.iterdir() if p.is_dir() and (p / "j" / "licitaciones.parquet").exists()
            and p.resolve() != DATOS.resolve()]


def viejos(v: str, ids: list) -> np.ndarray:
    """Algún ganador ya tenía contratos en la oficina antes de t (para separar viejos y nuevos en la evaluación)."""
    h = historial(v)
    lic = licitaciones(v)
    g = ganadores(v)
    g = g[g.id.isin(ids)].merge(lic[lic.id.isin(ids)][["id", "t", "office"]], on="id")
    m = g.merge(h[h.office.fillna("") != ""][["parent_uei", "office", "fecha"]], on=["parent_uei", "office"])
    return pd.Index(ids).isin(m[pd.to_datetime(m.fecha) < pd.to_datetime(m.t)].id.unique())


def textos_historial() -> pd.Index:
    """Los textos distintos de los contratos de las cinco verticales, en orden de aparición (j, r, d, it, rd)."""
    return pd.Index(pd.unique(np.concatenate([historial_crudo(v).descripcion.to_numpy(object) for v in VERTICALES])))


@functools.lru_cache(maxsize=None)
def historial_crudo(v: str) -> pd.DataFrame:
    """El historial en el orden del archivo (lo usan el grafo conjunto y el TF-IDF, que dependen del orden)."""
    h = pd.read_parquet(DATOS / v / "historial.parquet", columns=["fecha", "parent_uei", "office", "descripcion"])
    h["descripcion"] = h.descripcion.fillna("").astype(str)
    return h
