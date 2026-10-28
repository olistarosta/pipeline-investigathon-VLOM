"""Rankings en el formato de la entrega, completados con el B0 y validados, y la evaluación en val."""
from __future__ import annotations

import numpy as np
import pandas as pd

import b0 as B0
import evaluar as EV
from . import tablas as T
from .config import FUENTE, GEN, PREDICCIONES, VERTICALES


def b0(v: str, fuente: str, ids=None) -> pd.DataFrame:
    """El ranking del B0 oficial (`b0.py`) de las licitaciones de la fuente (o de `ids`)."""
    d = T.carpeta(v, fuente)
    lic = pd.read_parquet(d / "licitaciones.parquet")
    if ids is not None:
        lic = lic[lic.id.isin(set(ids))]
    emp = pd.read_parquet(d / "empresas.parquet")
    prior = B0.B0(pd.read_parquet(T.DATOS / v / "historial.parquet"))
    return pd.concat([prior.ranking(f, emp) for f in lic.itertuples()], ignore_index=True)[["id", "parent_uei", "puesto"]]


def completar(rk: pd.DataFrame, v: str, fuente: str) -> tuple[pd.DataFrame, int]:
    """Las licitaciones sin ranking (sin grafo ni contratos parecidos: ninguna candidata en el pool) se rankean con el
    B0. Devuelve (ranking, cuántas se completaron)."""
    ids = pd.read_parquet(T.carpeta(v, fuente) / "licitaciones.parquet", columns=["id"]).id
    faltan = sorted(set(ids) - set(rk.id))
    if faltan:
        rk = pd.concat([rk, b0(v, fuente, faltan)], ignore_index=True)
    return rk, len(faltan)


def validar(rk: pd.DataFrame, v: str, fuente: str, ids=None) -> None:
    """El formato de `evaluar.py`: todas las licitaciones, puestos 1..K, ≤ 100, sólo candidatas."""
    d = T.carpeta(v, fuente)
    lic = pd.read_parquet(d / "licitaciones.parquet")
    if ids is not None:
        lic = lic[lic.id.isin(set(ids))]
    EV.validar(rk, lic, pd.read_parquet(d / "empresas.parquet"), set(lic.id))


def escribir(rankings: dict[str, pd.DataFrame], fuente: str, nombre: str) -> None:
    d = PREDICCIONES / fuente / nombre
    d.mkdir(parents=True, exist_ok=True)
    for v in VERTICALES:
        rk, n = completar(rankings[v], v, fuente)
        rk = rk.sort_values(["id", "puesto"]).reset_index(drop=True)
        validar(rk, v, fuente)
        rk.to_csv(d / f"{v}.csv", index=False)
        print(f"  {v}: {rk.id.nunique():,} licitaciones" + (f" ({n} con el B0: sin pool)" if n else "")
              + f" -> {(d / f'{v}.csv').relative_to(PREDICCIONES.parent)}", flush=True)


def evaluar_val(pipeline: str) -> pd.DataFrame:
    """El modelo `train` del pipeline en val (2023S2): MRR, Hit@k y Δ MRR contra el B0 con IC95, en todas, S_inc /
    S_new (el B0 tiene o no a un ganador en su top 10) y viejos / nuevos (el ganador ya tenía o no contratos en la
    oficina)."""
    d = GEN / "modelos" / pipeline / "train"
    filas = []
    for v in VERTICALES:
        lic = T.licitaciones(v)
        ids = sorted(lic.id[lic.split == "val"])
        gan = T.ganadores(v)
        gan = gan[gan.id.isin(ids)]
        r0 = GEN / FUENTE / v / "b0_val.csv"
        if not r0.exists():
            b0(v, FUENTE, ids).to_csv(r0, index=False)
        rk, rk0 = pd.read_csv(d / f"val_{v}.csv"), pd.read_csv(r0)
        rk, _ = completar_val(rk, rk0, ids)
        validar(rk, v, FUENTE, ids)
        x = EV.por_licitacion(rk, gan).reindex(ids)
        x0 = EV.por_licitacion(rk0, gan).reindex(ids)
        sinc = (x0.mejor.fillna(np.inf) <= EV.S_INC).values
        viejo = T.viejos(v, ids)
        a, b = x.MRR.fillna(0).values, x0.MRR.fillna(0).values
        f = {"vertical": v, "N": len(ids), "MRR": a.mean(), "B0": b.mean(), "Hit@1": x["Hit@1"].fillna(0).mean(),
             "Hit@5": x["Hit@5"].fillna(0).mean(), "Hit@100": (x.mejor.fillna(np.inf) <= 100).mean()}
        for g, mk in {"todas": np.ones(len(ids), bool), "S_inc": sinc, "S_new": ~sinc, "viejos": viejo,
                      "nuevos": ~viejo}.items():
            lo, hi = EV.bootstrap_delta(a[mk], b[mk])
            f[f"Δ {g}"] = f"{(a[mk] - b[mk]).mean():+.3f} ({lo:+.3f}, {hi:+.3f})"
        filas.append(f)
    R = pd.DataFrame(filas)
    R.to_csv(d / "resumen_val.csv", index=False)
    return R


def completar_val(rk: pd.DataFrame, rk0: pd.DataFrame, ids) -> tuple[pd.DataFrame, int]:
    faltan = sorted(set(ids) - set(rk.id))
    return pd.concat([rk, rk0[rk0.id.isin(faltan)]], ignore_index=True), len(faltan)
