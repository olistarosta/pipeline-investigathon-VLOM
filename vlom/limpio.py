"""Pipeline limpio: el grafo conjunto + texto + los 5 niveles del B0, ordenados por un LambdaRank para las cinco.

Pool de cada licitación: el top 100 por suma sobre caminos (R = 1) y por camino más corto (R = 3) en cada grafo, más el
top 100 por parecido de texto. 15 variables (`config.VARIABLES_LIMPIO`). Un LambdaRank de LightGBM con las cinco
verticales juntas (la vertical como categórica), entrenado con las licitaciones del modelo que tienen algún ganador en
el pool.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import cache as C
from . import tablas as T
from .config import FUENTE, GEN, LAMBDARANK, MODELOS, NIVELES, POOL, VARIABLES_LIMPIO, VERTICALES

CODIGO = {v: k for k, v in enumerate(VERTICALES)}         # la vertical, como categórica


def carpeta_modelo(pipeline: str, modelo: str):
    d = GEN / "modelos" / pipeline / modelo
    d.mkdir(parents=True, exist_ok=True)
    return d


def tabla(v: str, fuente: str = FUENTE) -> pd.DataFrame:
    """Pares (licitación, empresa) del pool con las 15 variables, el semestre, el split y si ganó."""
    d = C.carpeta(fuente, v)
    P = pd.concat([pd.read_parquet(r) for r in sorted((d / "obs").glob("*.parquet"))], ignore_index=True)
    P = P.merge(T.licitaciones(v, fuente)[["id", "semestre", "split"]], on="id")
    if fuente == FUENTE:
        g = T.ganadores(v)
        P["es_gan"] = pd.MultiIndex.from_frame(P[["id", "parent_uei"]]).isin(pd.MultiIndex.from_frame(g))
    else:
        P["es_gan"] = False
    P = P.sort_values(["id", "parent_uei"]).reset_index(drop=True)
    P = P[P[POOL].any(axis=1)].reset_index(drop=True)
    for g in ("gemb_R1", "gtf_R1"):                           # relativas a la lista
        mx = P[f"{g}_s"].groupby(P.id).transform("max")
        P[f"r_{g}"] = np.where(mx > 0, P[f"{g}_s"] / mx.where(mx > 0, 1), np.nan).astype(np.float32)
    for archivo, cols in (("variables", ["sim_max_e", "sim_top3_e", "sim_max", "frac_psc"]), ("niveles", NIVELES)):
        P = pegar(P, d / f"{archivo}.parquet", cols)
    assert P.sim_max_e.notna().all()
    P["vertical"] = v
    return P


def pegar(P: pd.DataFrame, archivo, cols: list[str]) -> pd.DataFrame:
    n = len(P)
    P = P.merge(pd.read_parquet(archivo, columns=["id", "parent_uei", *cols]), on=["id", "parent_uei"], how="left",
                validate="one_to_one")
    assert len(P) == n
    return P


def tabla_conjunta(tablas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Las cinco verticales en una tabla, con un grupo `j` por licitación (un id puede repetirse entre verticales)."""
    partes, base = [], 0
    for P in tablas.values():
        cod, _ = pd.factorize(P.id, sort=False)
        partes.append(P.assign(j=cod + base))
        base += int(cod.max()) + 1
    return pd.concat(partes, ignore_index=True).sort_values("j", kind="stable").reset_index(drop=True)


def matriz(P: pd.DataFrame, variables: list[str]) -> np.ndarray:
    return np.column_stack([P[variables].values.astype(np.float32), P.vertical.map(CODIGO).values.astype(np.float32)])


def entrenar_ranker(PC: pd.DataFrame, variables: list[str]):
    """LambdaRank sobre las licitaciones de PC con algún ganador en el pool."""
    import lightgbm as lgb
    tr = PC[PC.groupby("j").es_gan.transform("any")]
    m = lgb.LGBMRanker(**LAMBDARANK)
    m.fit(matriz(tr, variables), tr.es_gan.values.astype(int), group=tr.groupby("j", sort=False).size().values,
          categorical_feature=[len(variables)])
    return m.booster_


def guardar(booster, d, modelo: str, variables: list[str], extra: dict | None = None) -> None:
    booster.save_model(str(d / "modelo.txt"))
    gan = pd.Series(booster.feature_importance("gain"), index=variables + ["vertical"])
    (gan / gan.sum()).sort_values(ascending=False).to_csv(d / "ganancia.csv", header=["fraccion"])
    (d / "modelo.json").write_text(json.dumps({"modelo": modelo, "entrenado_con": MODELOS[modelo], "variables": variables,
                                               "pool": POOL, "parametros": LAMBDARANK, **(extra or {})},
                                              indent=1, ensure_ascii=False), encoding="utf-8")


def rankear(P: pd.DataFrame, puntaje: np.ndarray) -> pd.DataFrame:
    """Top 100 por licitación (desempate: la suma del grafo de embeddings, después el parent_uei)."""
    x = P[["id", "parent_uei"]].assign(s=puntaje, e=P.gemb_R1_s.fillna(0).values)
    x = x.sort_values(["id", "s", "e", "parent_uei"], ascending=[True, False, False, True])
    x["puesto"] = x.groupby("id").cumcount() + 1
    return x[x.puesto <= 100][["id", "parent_uei", "puesto"]]


def entrenar(modelo: str) -> None:
    t0 = time.time()
    tablas = {v: tabla(v) for v in VERTICALES}
    PC = tabla_conjunta(tablas)
    b = entrenar_ranker(PC[PC.split.isin(MODELOS[modelo])], VARIABLES_LIMPIO)
    d = carpeta_modelo("limpio", modelo)
    guardar(b, d, modelo, VARIABLES_LIMPIO)
    if modelo == "train":
        for v, P in tablas.items():
            va = P[P.split == "val"].reset_index(drop=True)
            rankear(va, b.predict(matriz(va, VARIABLES_LIMPIO))).to_csv(d / f"val_{v}.csv", index=False)
    print(f"  limpio {modelo}: entrenado con {' + '.join(MODELOS[modelo])} ({time.time() - t0:.0f} s)", flush=True)


def cargar_modelo(pipeline: str, modelo: str):
    import lightgbm as lgb
    r = GEN / "modelos" / pipeline / modelo / "modelo.txt"
    if not r.exists():
        raise SystemExit(f"falta el modelo {pipeline} {modelo} ({r}): entrenarlo o bajar el precalculado")
    return lgb.Booster(model_file=str(r))


def predecir(fuente: str, modelo: str) -> dict[str, pd.DataFrame]:
    b = cargar_modelo("limpio", modelo)
    out = {}
    for v in VERTICALES:
        P = tabla(v, fuente)
        out[v] = rankear(P, b.predict(matriz(P, VARIABLES_LIMPIO)))
    return out
