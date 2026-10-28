"""El caché de una fuente: candidatas (pool) de cada licitación y sus variables, por par (licitación, empresa).

Por vertical, en `generados/<fuente>/<v>/`:
- `pool_texto.parquet`: las 100 candidatas con el contrato previo más parecido a la licitación (`bge-small`).
- `obs/<corte>_<k>.parquet`: el pool (las cuatro fuentes del grafo ∪ la de texto, con una marca por fuente) y los
  observables del grafo (`recorrido.corte`). Un archivo por corte y lote de licitaciones.
- `variables.parquet`: parecido de texto con sus contratos previos (`sim_max_e`, `sim_top3_e` con `bge-small`;
  `sim_max` con TF-IDF) y `frac_psc` (fracción de sus contratos previos en el PSC de la licitación).
- `gemma.parquet`: `sim_max_g`, el parecido máximo con EmbeddingGemma 2 (lo usa el pipeline completo).
- `niveles.parquet`: los 5 niveles del B0 (`b0.py`) por separado y sin decaimiento: la parte de la empresa en los
  contratos previos de la oficina+PSC, el PSC, el NAICS, todos, y el PSC con el mismo set-aside.

Sin fuga: todo con contratos de fecha < t (búsqueda binaria sobre el historial ordenado por fecha); el grafo, con los
de fecha < 1.º del mes de t; candidatas del semestre de la licitación (o la foto de `empresas`, en datos nuevos).
"""
from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd

import b0 as B0
from . import grafo as GR
from . import tablas as T
from .config import FUENTE, GEN, INICIO_GRAFO, LOTE_CORTE, NIVELES, RECORRIDOS, TOP, VERTICALES, W_TITULO
from .recorrido import combinar, hay_gpu, recorrer, reducir
from .recorrido import corte as corte_grafo
from .texto import Densos, consulta_tfidf, tfidf_vertical

LOTE = 32
SHIFT = np.int64(1 << 17)            # > cualquier fecha en días desde 1970


def carpeta(fuente: str, v: str):
    d = GEN / fuente / v
    d.mkdir(parents=True, exist_ok=True)
    return d


def uei(v: str, fuente: str) -> pd.Index:
    return pd.Index(pd.unique(np.concatenate([T.empresas(v, fuente).parent_uei.to_numpy(object),
                                              T.historial(v).parent_uei.to_numpy(object)])))


class Indice:
    """Contratos indexados por clave entera y fecha: cuántos hay con fecha < t."""

    def __init__(self, clave: np.ndarray, ts: np.ndarray):
        ok = clave >= 0
        self.c = np.sort(clave[ok].astype(np.int64) * SHIFT + ts[ok].astype(np.int64))

    def antes(self, clave: np.ndarray, t: np.ndarray) -> np.ndarray:
        clave = clave.astype(np.int64)
        base = np.where(clave >= 0, clave, 0) * SHIFT
        n = np.searchsorted(self.c, base + t.astype(np.int64), side="left") - np.searchsorted(self.c, base, side="left")
        return np.where(clave >= 0, n, 0)


def codigo(*cols: np.ndarray) -> list[np.ndarray]:
    """Factoriza conjuntamente columnas de texto; "" -> −1."""
    todo = np.concatenate(cols)
    cod, _ = pd.factorize(todo)
    cod = np.where(todo == "", -1, cod)
    return np.split(cod, np.cumsum([len(c) for c in cols])[:-1])


def max_top3(fis: np.ndarray, fi_c: np.ndarray, s: np.ndarray, marca: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Para cada empresa de `fis` (ordenado): el máximo y el promedio de los 3 mayores de `s` sobre sus contratos
    (`fi_c`); 0 sin contratos."""
    mx, t3 = np.zeros(len(fis)), np.zeros(len(fis))
    marca[fis] = True
    sel = marca[fi_c]
    marca[fis] = False
    fs, ss = fi_c[sel], s[sel]
    if len(fs):
        o = np.lexsort((-ss, fs))
        fs, ss = fs[o], ss[o]
        uu, ini, cnt = np.unique(fs, return_index=True, return_counts=True)
        m1 = ss[ini].astype(np.float64)
        tt = m1.copy()
        for k in (1, 2):
            tiene = cnt > k
            tt[tiene] += ss[ini[tiene] + k]
        pos = np.searchsorted(fis, uu)
        mx[pos], t3[pos] = m1, tt / 3
    return mx, t3


# ---------------------------------------------------------------- pool de texto

def pool_texto(v: str, fuente: str) -> None:
    """Las TOP candidatas con el contrato previo más parecido (`bge-small`), en orden de parecido."""
    r = carpeta(fuente, v) / "pool_texto.parquet"
    if r.exists():
        return
    t0 = time.time()
    L, h, U = T.licitaciones(v, fuente), T.historial(v), uei(v, fuente)
    fi_h = U.get_indexer(h.parent_uei)
    E = Densos(v, "bge-small", fuente)
    n_t = np.searchsorted(h.ts.values, L.ti.values, side="left")
    en_pool = {}
    for s, c in T.candidatas(v, fuente).items():
        en_pool[s] = np.zeros(len(U), bool)
        en_pool[s][U.get_indexer(list(c))] = True
    filas = []
    for a in range(0, len(L), LOTE):
        lote = np.arange(a, min(a + LOTE, len(L)))
        nmax = int(n_t[lote].max())
        if nmax == 0:
            continue
        S = E.sims(L.id.values[lote], W_TITULO["bge-small"][v], nmax)
        for q, j in enumerate(lote):
            s, fi_c = S[:n_t[j], q], fi_h[:n_t[j]]
            ok = en_pool[L.semestre[j]][fi_c]
            s_c, f_c = s[ok], fi_c[ok]
            m = min(len(s_c), 20_000)
            while True:                  # los m contratos más parecidos, hasta juntar TOP empresas distintas
                idx = np.argpartition(-s_c, m - 1)[:m] if m < len(s_c) else np.arange(len(s_c))
                idx = idx[np.lexsort((idx, -s_c[idx]))]
                uu, prim = np.unique(f_c[idx], return_index=True)
                if len(uu) >= TOP or m >= len(s_c):
                    break
                m = min(len(s_c), m * 4)
            filas.append(pd.DataFrame({"id": L.id[j], "parent_uei": U.values[f_c[idx][np.sort(prim)][:TOP]]}))
    pd.concat(filas, ignore_index=True).to_parquet(r, index=False)
    print(f"  {v}: pool de texto ({time.time() - t0:.0f} s)", flush=True)


# ---------------------------------------------------------------- observables del grafo

def cortes(lic: pd.DataFrame) -> pd.Series:
    """El corte del grafo de cada licitación: el 1.º de su mes (los grafos posteriores al fin del historial son todos
    iguales: se usa el primero). Enero de 2022 no tiene grafo (NaN)."""
    fin = (pd.Timestamp(max(T.historial(v).fecha.max() for v in VERTICALES)).to_period("M") + 1).start_time
    mes = pd.to_datetime(lic.t).dt.to_period("M").dt.start_time
    c = mes.where(mes <= fin, fin)
    return c.where(c >= pd.Timestamp(INICIO_GRAFO)).dt.strftime("%Y-%m-%d")


def observables(fuente: str) -> None:
    """Para cada corte: arma (si faltan) y carga los dos grafos, y recorre las licitaciones de las cinco verticales."""
    gpu = hay_gpu()
    D = GR.conjuntos((fuente,))
    lic = D["gemb"].lic.assign(corte=lambda x: cortes(x)).dropna(subset=["corte"])
    texto = {v: pd.read_parquet(carpeta(fuente, v) / "pool_texto.parquet").assign(id=lambda x, v=v: v + ":" + x.id)
             .groupby("id").parent_uei.apply(list).to_dict() for v in VERTICALES}
    print(f"  observables de {fuente}: {lic.corte.nunique()} cortes, {len(lic):,} licitaciones "
          f"({'GPU' if gpu else 'CPU'})", flush=True)
    for c, Lc in lic.groupby("corte"):
        pend = []
        for (v, sem), Lv in Lc.groupby(["vertical", "semestre"]):
            for k, a in enumerate(range(0, len(Lv), LOTE_CORTE)):
                r = carpeta(fuente, v) / "obs" / f"{c}_{sem}_{k}.parquet"
                if not r.exists():
                    pend.append((v, sem, Lv.iloc[a:a + LOTE_CORTE].reset_index(drop=True), r))
        if not pend:
            continue
        for g in D:
            GR.armar(D[g], g, c)
        print(f"  corte {c}: {sum(len(x[2]) for x in pend):,} licitaciones", flush=True)
        if fuente != FUENTE:
            _por_grafo(D, c, pend, texto, gpu)
            continue
        G = {g: GR.cargar(D[g], g, c, con_v=(g == "gemb")) for g in D}
        for v, sem, L, r in pend:
            X = corte_grafo(D, G, L, D["gemb"].cand[(v, sem)], texto[v], gpu)
            X["id"] = X.id.str[len(v) + 1:]
            r.parent.mkdir(exist_ok=True)
            X.to_parquet(r, index=False)


def _por_grafo(D: dict, c: str, pend: list, texto: dict, gpu: bool) -> None:
    """Datos nuevos (un solo corte): un grafo en memoria a la vez. Con cada uno recorre todos los lotes y guarda en
    disco sólo las columnas de las candidatas (`recorrido.reducir`); después arma el pool y los observables de cada
    lote (`recorrido.combinar`). Da lo mismo que `recorrido.corte` con los dos grafos cargados."""
    parcial = lambda r, nom: r.parent / "parcial" / f"{r.stem}_{nom}.npy"  # noqa: E731
    emp = None
    for g in D:
        t0 = time.time()
        G = GR.cargar(D[g], g, c, con_v=(g == "gemb"))
        if emp is None:
            emp = G.empresas
        assert emp.equals(G.empresas)
        for v, sem, L, r in pend:
            ok = np.asarray(emp.isin(D["gemb"].cand[(v, sem)]))
            for nom, cfg in RECORRIDOS.items():
                if cfg["grafo"] == g and not parcial(r, nom).exists():
                    parcial(r, nom).parent.mkdir(parents=True, exist_ok=True)
                    np.save(parcial(r, nom), reducir(recorrer(D[g], G, nom, L, gpu), ok, cfg))
        del G
        gc.collect()
        print(f"    grafo {g}: {time.time() - t0:.0f} s", flush=True)
    for v, sem, L, r in pend:
        cand = D["gemb"].cand[(v, sem)]
        ok = np.asarray(emp.isin(cand))
        X = combinar({nom: np.load(parcial(r, nom)) for nom in RECORRIDOS}, emp.values[ok], L, cand, texto[v])
        X["id"] = X.id.str[len(v) + 1:]
        X.to_parquet(r, index=False)
        for nom in RECORRIDOS:
            parcial(r, nom).unlink()


def pares(v: str, fuente: str) -> pd.DataFrame:
    """Los pares (id, parent_uei) del pool de la vertical."""
    return pd.concat([pd.read_parquet(r, columns=["id", "parent_uei"])
                      for r in sorted((carpeta(fuente, v) / "obs").glob("*.parquet"))], ignore_index=True)


def grafos_corte(corte: str) -> None:
    """Sólo arma los dos grafos de un corte (para el release: el corte de los datos nuevos)."""
    D = GR.conjuntos()
    for g in D:
        GR.armar(D[g], g, corte)


def armar_grafos(fuente: str) -> None:
    """Arma los grafos de todos los cortes de la fuente (sin recorrerlos)."""
    D = GR.conjuntos((fuente,))
    for c in sorted(cortes(D["gemb"].lic).dropna().unique()):
        for g in D:
            GR.armar(D[g], g, c)


# ---------------------------------------------------------------- variables por par

def _ordenados(v: str, fuente: str, U: pd.Index):
    X = pares(v, fuente)
    L = T.licitaciones(v, fuente)
    jx, fi = pd.Index(L.id).get_indexer(X.id), U.get_indexer(X.parent_uei)
    assert (jx >= 0).all() and (fi >= 0).all()
    return X.assign(j=jx, fi=fi).sort_values(["j", "fi"]).reset_index(drop=True), L


def variables(v: str, fuente: str) -> None:
    r = carpeta(fuente, v) / "variables.parquet"
    if r.exists():
        return
    t0 = time.time()
    h, U = T.historial(v), uei(v, fuente)
    X, L = _ordenados(v, fuente, U)
    fi_h = U.get_indexer(h.parent_uei)
    lj, fi = X.j.values, X.fi.values
    t = L.ti.values[lj]

    # rubro: fracción de sus contratos previos que son del PSC de la licitación
    ps_h, ps_l = codigo(h.psc_u.values, L.psc_u.values)
    m = int(max(ps_h.max(initial=-1), ps_l.max(initial=-1))) + 1
    k_psc = Indice(np.where((fi_h >= 0) & (ps_h >= 0), fi_h.astype(np.int64) * m + ps_h, -1), h.ts.values).antes(
        np.where(ps_l[lj] >= 0, fi.astype(np.int64) * m + ps_l[lj], -1), t).astype(np.float32)
    n = Indice(fi_h.astype(np.int64), h.ts.values).antes(fi.astype(np.int64), t)
    X["frac_psc"] = (pd.Series(k_psc) / pd.Series(n).where(n > 0)).astype(np.float32).values

    # texto: parecido con sus contratos previos, TF-IDF y bge-small
    tf, bp = tfidf_vertical(v)
    H, Q = tf(h.descripcion.values), tf(consulta_tfidf(L, bp))
    E = Densos(v, "bge-small", fuente)
    n_t = np.searchsorted(h.ts.values, L.ti.values, side="left")
    lic_u, ini_u = np.unique(lj, return_index=True)
    fin_u = np.r_[ini_u[1:], len(X)]
    cols = {c: np.zeros(len(X), np.float32) for c in ("sim_max", "sim_max_e", "sim_top3_e")}
    marca = np.zeros(len(U), bool)
    for a in range(0, len(lic_u), LOTE):
        filas = lic_u[a:a + LOTE]
        nmax = int(n_t[filas].max())
        if nmax == 0:
            continue
        St = (H[:nmax] @ Q[filas].T).tocsc()
        Se = E.sims(L.id.values[filas], W_TITULO["bge-small"][v], nmax)
        for q, j in enumerate(filas):
            rr = slice(ini_u[a + q], fin_u[a + q])
            b, c = St.indptr[q], St.indptr[q + 1]
            ok = St.indices[b:c] < n_t[j]
            cols["sim_max"][rr] = max_top3(fi[rr], fi_h[St.indices[b:c][ok]], St.data[b:c][ok], marca)[0]
            cols["sim_max_e"][rr], cols["sim_top3_e"][rr] = max_top3(fi[rr], fi_h[:n_t[j]], Se[:n_t[j], q], marca)
    for c, x in cols.items():
        X[c] = x
    X.drop(columns=["j", "fi"]).to_parquet(r, index=False)
    print(f"  {v}: variables de {len(X):,} pares ({time.time() - t0:.0f} s)", flush=True)


def gemma(v: str, fuente: str) -> None:
    r = carpeta(fuente, v) / "gemma.parquet"
    if r.exists():
        return
    t0 = time.time()
    h, U = T.historial(v), uei(v, fuente)
    X, L = _ordenados(v, fuente, U)
    fi_h = U.get_indexer(h.parent_uei)
    lj, fi = X.j.values, X.fi.values
    E = Densos(v, "gemma2", fuente)
    n_t = np.searchsorted(h.ts.values, L.ti.values, side="left")
    lic_u, ini_u = np.unique(lj, return_index=True)
    fin_u = np.r_[ini_u[1:], len(X)]
    mx_g = np.zeros(len(X), np.float32)
    marca = np.zeros(len(U), bool)
    for a in range(0, len(lic_u), LOTE):
        filas = lic_u[a:a + LOTE]
        nmax = int(n_t[filas].max())
        if nmax == 0:
            continue
        Se = E.sims(L.id.values[filas], W_TITULO["gemma2"][v], nmax)
        for q, j in enumerate(filas):
            rr = slice(ini_u[a + q], fin_u[a + q])
            mx_g[rr] = max_top3(fi[rr], fi_h[:n_t[j]], Se[:n_t[j], q], marca)[0]
    X["sim_max_g"] = mx_g
    X.drop(columns=["j", "fi"]).to_parquet(r, index=False)
    print(f"  {v}: Gemma de {len(X):,} pares ({time.time() - t0:.0f} s)", flush=True)


def niveles(v: str, fuente: str) -> None:
    """Los 5 niveles del B0 por par, sin decaimiento: cada contrato previo pesa 1, normalizado a una distribución
    sobre las empresas dentro de cada nivel (0 si el nivel no tiene contratos previos)."""
    r = carpeta(fuente, v) / "niveles.parquet"
    if r.exists():
        return
    t0 = time.time()
    X = pares(v, fuente)
    prior = B0.B0(pd.read_parquet(T.DATOS / v / "historial.parquet"))
    prior.w = np.ones_like(prior.w)
    prior.tablas = [prior._indexar(cols) for cols in B0.NIVELES]
    lic = T.licitaciones(v, fuente).set_index("id")
    pos = pd.Index(prior.empresas)
    partes = []
    for lid, g in X.groupby("id", sort=False):
        l = lic.loc[lid]
        ts = np.datetime64(pd.Timestamp(l.t), "D").astype(np.int64)
        ctx = {"psc": str(l.psc).upper(), "office": str(l.office).upper(), "naics": str(l.naics),
               "sa": B0.set_aside(l.set_aside)}
        niv = np.zeros((len(B0.NIVELES), len(prior.empresas)))
        for k, (cols, tabla) in enumerate(zip(B0.NIVELES, prior.tablas)):
            e = tabla.get(tuple(ctx[c] for c in cols))
            if e is None:
                continue
            t_arr, fi, w = e
            n = np.searchsorted(t_arr, ts, side="left")
            if n:
                x = np.bincount(fi[:n], weights=w[:n], minlength=len(prior.empresas))
                niv[k] = x / x.sum()
        idx = pos.get_indexer(g.parent_uei.values)
        out = pd.DataFrame({"id": lid, "parent_uei": g.parent_uei.values})
        for k, c in enumerate(NIVELES):
            out[c] = np.where(idx >= 0, niv[k][np.maximum(idx, 0)], 0.0).astype(np.float32)
        partes.append(out)
    pd.concat(partes, ignore_index=True).to_parquet(r, index=False)
    print(f"  {v}: niveles del B0 de {len(X):,} pares ({time.time() - t0:.0f} s)", flush=True)


def todo(fuente: str) -> None:
    for v in VERTICALES:
        pool_texto(v, fuente)
    observables(fuente)
    for v in VERTICALES:
        variables(v, fuente)
        gemma(v, fuente)
        niveles(v, fuente)
