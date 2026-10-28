"""El recorrido del grafo desde la oficina de cada licitación: observables y pool de candidatas.

Para una licitación con consulta q, cada arista cuesta

    c = (−log w − β·log h(cos(q, V)) − β_t·log h_t(cos_tfidf(q, textos de la arista))) · (λ si es interna),

con h(s) = clip((s − s_q) / (1 − s_q), ε, 1) y h_t(s) = clip(s / s_ref, piso, 1). Las aristas paralelas se suman
(exp(−c)) y se normaliza por fila: P[u, v] es la probabilidad de pasar de u a v. Desde la oficina de la licitación:

- **suma sobre caminos** (`tipo` "suma"): la masa que llega a cada empresa en 1..K pasos de la caminata, `s`; su
  puesto entre todas las empresas del grafo y, en el pool, el top 100.
- **camino más corto** (`tipo` "camino"): −log de la probabilidad del camino más probable (Dijkstra); su puesto y, en
  el pool, el top 100.

Si la oficina no está en el grafo, la caminata sale de un **nodo virtual** unido a las oficinas de su subagencia
(peso 1), a las 20 de su departamento con centroide más parecido (peso 0,25) y a las 10 con el contrato más parecido a
la licitación (peso 1). En R = 1 se recorren las rondas 0 y 1 (contratos, inferidas y oficina–oficina); en R = 3, todas.

Con GPU, el recorrido de un lote de licitaciones va junto (la caminata como sumas dispersas y el camino más corto
por Bellman-Ford); sin GPU, licitación por licitación con scipy. Dan lo mismo salvo redondeo (~1e−7 relativo).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from . import grafo as GR
from .config import RECORRIDOS, TOP

EPS = 1e-3


class Motor:
    """Las aristas de ronda ≤ R de un grafo, agrupadas por par de nodos, con la lista dirigida de los pares."""

    def __init__(self, G: GR.Grafo, R: int):
        m = G.r <= R
        a, b = np.minimum(G.u[m], G.v[m]), np.maximum(G.u[m], G.v[m])
        k = a.astype(np.int64) * G.n_nodos + b
        o = np.argsort(k, kind="stable")
        self.w, self.interno = G.w[m][o], G.r[m][o] > 0
        self.V = G.V[m][o] if G.V is not None else None
        self.ti, self.tj = G.ti[m][o], G.tj[m][o]
        k = k[o]
        self.ini = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
        pk = k[self.ini]
        pa, pb = pk // G.n_nodos, pk % G.n_nodos
        fil, col = np.r_[pa, pb], np.r_[pb, pa]
        idx = np.r_[np.arange(len(pk)), np.arange(len(pk))]
        o = np.lexsort((col, fil))
        self.perm, self.indices, self.fila = idx[o], col[o].astype(np.int32), fil[o].astype(np.int32)
        self.indptr = np.searchsorted(fil[o], np.arange(G.n_nodos + 1)).astype(np.int32)
        self.n, self.No, self.G = G.n_nodos, G.No, G
        self.mlogw = -np.log(self.w)
        self.n_aristas, self.n_pares = len(self.w), len(pk)
        self._nt = None

    def similitudes(self, Q: np.ndarray, lote: int = 400_000) -> np.ndarray:
        S = np.empty((len(self.w), len(Q)), np.float32)
        for a in range(0, len(self.w), lote):
            S[a:a + lote] = self.V[a:a + lote].astype(np.float32) @ Q.T
        return S

    def norma_t(self, HT) -> np.ndarray:
        """‖t_i + t_j‖ (TF-IDF) de cada arista; 1 si tiene un solo texto."""
        if self._nt is None:
            par = self.tj >= 0
            nt = np.ones(len(self.ti), np.float32)
            a, b = self.ti[par], self.tj[par]
            dot = np.asarray(HT[a].multiply(HT[b]).sum(1)).ravel()
            na = np.asarray(HT[a].multiply(HT[a]).sum(1)).ravel()
            nb = np.asarray(HT[b].multiply(HT[b]).sum(1)).ravel()
            nt[par] = np.sqrt(np.maximum(na + nb + 2 * dot, 1e-12))
            self._nt = nt
        return self._nt

    def similitudes_t(self, A: np.ndarray, HT) -> np.ndarray:
        """(aristas × consultas) coseno TF-IDF con norm(t_i + t_j), desde A = textos × consultas."""
        A = np.vstack([A, np.zeros((1, A.shape[1]), np.float32)])          # fila −1 = sin texto
        return (A[self.ti] + A[self.tj]) / self.norma_t(HT)[:, None]

    def costos(self, S, St, cfg: dict) -> np.ndarray:
        if cfg["beta"]:
            C = self.mlogw[:, None] - cfg["beta"] * np.log(np.clip((S - cfg["s_q"]) / (1 - cfg["s_q"]), EPS, 1))
        else:
            C = np.repeat(self.mlogw[:, None], St.shape[1], axis=1).astype(np.float32)
        if cfg.get("beta_t"):
            C -= cfg["beta_t"] * np.log(np.clip(St / cfg["s_ref"], cfg["piso"], 1))
        if cfg["lam"] != 1:
            C[self.interno] *= cfg["lam"]
        return C

    def _csr(self, datos: np.ndarray) -> csr_matrix:
        return csr_matrix((datos, self.indices, self.indptr), shape=(self.n, self.n))

    def virtuales(self, D: GR.Conjunto, oficinas, Q: np.ndarray, ids, k_v: int = 10, k_dep: int = 20,
                  kappa_sub: float = 1.0, kappa_dep: float = 0.25) -> list:
        """Para cada consulta cuya oficina no está en el grafo, las aristas de su nodo virtual (o, w, se, st)."""
        G = self.G
        HT, Qt = D.tfidf(G.corte)
        out = []
        for q, cod in enumerate(oficinas):
            if G.oficinas.get_indexer([cod])[0] >= 0:
                out.append(None)
                continue
            partes = []
            if cod in D.org.index:
                sc = G.centroide @ Q[q]
                sub, dep = D.org.subagencia[cod], D.org.departamento[cod]
                o1 = np.flatnonzero(G.subag == sub)
                o2 = np.flatnonzero((G.depto == dep) & (G.subag != sub))
                o2 = o2[np.argsort(-sc[o2])[:k_dep]]
                for oo, kap in ((o1, kappa_sub), (o2, kappa_dep)):
                    if len(oo):
                        partes.append(pd.DataFrame({"o": oo, "w": kap, "se": sc[oo], "st": 0.0}))
            sc = G.B @ Q[q]
            x = pd.DataFrame({"o": G.k_o, "se": sc, "c": np.arange(len(sc))})
            x = x.sort_values("se", ascending=False).drop_duplicates("o").head(k_v)
            qt = Qt[D.Q.index.get_indexer([ids[q]])]
            st = np.asarray((HT[G.kt[x.c.values]] @ qt.T).todense()).ravel()
            partes.append(pd.DataFrame({"o": x.o.values, "w": 1.0, "se": x.se.values, "st": st}))
            out.append(pd.concat(partes, ignore_index=True))
        return out


def aristas_virtuales(vq: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(oficinas, peso exp(−λ·c)) de las aristas del nodo virtual."""
    c = -np.log(vq.w.values) - cfg["beta"] * np.log(np.clip((vq.se.values - cfg["s_q"]) / (1 - cfg["s_q"]), EPS, 1))
    if cfg.get("beta_t"):
        c -= cfg["beta_t"] * np.log(np.clip(vq.st.values / cfg["s_ref"], cfg["piso"], 1))
    a = pd.Series(np.exp(-cfg["lam"] * c)).groupby(vq.o.values).sum()
    return a.index.values.astype(np.int64), a.values


# ---------------------------------------------------------------- CPU

def transicion_virtual(mot: Motor, d: np.ndarray, vq: pd.DataFrame, cfg: dict) -> csr_matrix:
    """Probabilidades de transición (fila = desde) con el nodo virtual n agregado."""
    n = mot.n
    o, a = aristas_virtuales(vq, cfg)
    fil = np.r_[mot.fila, np.full(len(o), n), o]
    col = np.r_[mot.indices, o, np.full(len(o), n)]
    dat = np.r_[d, a, a]
    g = np.bincount(fil, weights=dat, minlength=n + 1)
    return csr_matrix((dat / np.where(g > 0, g, 1)[fil], (fil, col)), shape=(n + 1, n + 1))


def observables_cpu(mot: Motor, P: np.ndarray, fuentes: np.ndarray, virt, cfg: dict) -> np.ndarray:
    """(consultas × empresas): la suma sobre caminos (0 = no alcanzable) o el −log del camino más probable (inf)."""
    n, No = mot.n, mot.No
    out = np.zeros((len(fuentes), n - No), np.float32) if cfg["tipo"] == "suma" else \
        np.full((len(fuentes), n - No), np.inf, np.float32)
    for q, f in enumerate(fuentes):
        vq = virt[q]
        if f < 0 and vq is None:
            continue
        d = P[mot.perm, q].astype(np.float64)
        if vq is not None:
            T, src = transicion_virtual(mot, d, vq, cfg), n
        else:
            g = np.bincount(mot.fila, weights=d, minlength=n)
            T, src = mot._csr(d / np.where(g > 0, g, 1)[mot.fila]), int(f)
        if cfg["tipo"] == "suma":
            M = T.T.tocsr()
            x = np.zeros(T.shape[0])
            x[src] = 1.0
            for _ in range(cfg["K"]):
                x = M @ x
                out[q] += x[No:n].astype(np.float32)
        else:
            T.data = -np.log(np.maximum(T.data, 1e-300)) + 1e-9
            out[q] = dijkstra(T, directed=True, indices=src)[No:n]
    return out


# ---------------------------------------------------------------- GPU

def hay_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def observables_gpu(mot: Motor, P: np.ndarray, fuentes: np.ndarray, virt, cfg: dict, max_iter: int = 2_000) -> np.ndarray:
    """Lo mismo que `observables_cpu`, con todas las consultas del lote juntas en la GPU (float64)."""
    import torch
    t = lambda x, dtype=None: torch.as_tensor(np.ascontiguousarray(x), device="cuda", dtype=dtype)  # noqa: E731
    n, No, nq = mot.n, mot.No, P.shape[1]
    fila, col = t(mot.fila.astype(np.int64)), t(mot.indices.astype(np.int64))
    d = t(P)[t(mot.perm)].double()                            # aristas dirigidas × consultas
    vf, vc, vqq, va, src = [], [], [], [], np.full(nq, -1, np.int64)
    for q, f in enumerate(fuentes):
        if virt[q] is not None:
            o, a = aristas_virtuales(virt[q], cfg)
            vf += [np.full(len(o), n), o]
            vc += [o, np.full(len(o), n)]
            vqq += [np.full(len(o), q)] * 2
            va += [a, a]
            src[q] = n
        elif f >= 0:
            src[q] = int(f)
    hay_v = len(vf) > 0
    if hay_v:
        vf, vc, vqq, va = (t(np.concatenate(x)) for x in (vf, vc, vqq, va))
    g = torch.zeros((n + 1, nq), device="cuda", dtype=torch.float64)
    g.index_add_(0, fila, d)
    if hay_v:
        g.index_put_((vf, vqq), va, accumulate=True)
    g = torch.where(g > 0, g, torch.ones_like(g))
    T = d / g[fila]
    Tv = va / g[vf, vqq] if hay_v else None
    del d
    ok = np.flatnonzero(src >= 0)
    if cfg["tipo"] == "suma":
        x = torch.zeros((n + 1, nq), device="cuda", dtype=torch.float64)
        x[t(src[ok]), t(ok)] = 1.0
        s = torch.zeros((nq, n - No), device="cuda", dtype=torch.float32)
        for _ in range(cfg["K"]):
            y = torch.zeros_like(x)
            y.index_add_(0, col, T * x[fila])
            if hay_v:
                y.index_put_((vc, vqq), Tv * x[vf, vqq], accumulate=True)
            x = y
            s += x[No:n].T.float()
        return s.cpu().numpy()
    T.clamp_(min=1e-300).log_().neg_().add_(1e-9)
    if hay_v:
        Tv = -torch.log(torch.clamp(Tv, min=1e-300)) + 1e-9
    dist = torch.full((n + 1, nq), torch.inf, device="cuda", dtype=torch.float64)
    dist[t(src[ok]), t(ok)] = 0.0
    colx = col[:, None].expand_as(T)
    for _ in range(max_iter):                                 # Bellman-Ford: relajar todo hasta que nada cambie
        nueva = dist.scatter_reduce(0, colx, dist[fila] + T, "amin", include_self=True)
        if hay_v:
            nueva.view(-1).scatter_reduce_(0, vc * nq + vqq, dist[vf, vqq] + Tv, "amin", include_self=True)
        if torch.equal(nueva, dist):
            break
        dist = nueva
    else:
        raise RuntimeError("Bellman-Ford no convergió")
    return dist[No:n].T.float().cpu().numpy()


# ---------------------------------------------------------------- un recorrido sobre un lote de licitaciones

def recorrer(D: GR.Conjunto, G: GR.Grafo, nombre: str, L: pd.DataFrame, gpu: bool, lote: int = 48) -> np.ndarray:
    """(licitaciones × empresas del grafo) el observable del recorrido `nombre` para las licitaciones L."""
    cfg = RECORRIDOS[nombre]
    mot = Motor(G, cfg["R"])
    if gpu:     # cada arreglo aristas dirigidas × consultas (float64) en ~0,6 GB
        lote = int(np.clip(0.6e9 // (8 * len(mot.perm)), 8, lote))
    HT = Qt = None
    if cfg.get("beta_t"):
        HT, Qt = D.tfidf(G.corte)
    fuentes = G.oficinas.get_indexer(L.office).astype(np.int64)
    out = np.empty((len(L), len(G.empresas)), np.float32)
    for a in range(0, len(L), lote):
        sub = L.iloc[a:a + lote]
        Q = np.stack(D.Q.loc[sub.id].values).astype(np.float32)
        S = mot.similitudes(Q) if cfg["beta"] else None
        St = None
        if cfg.get("beta_t"):
            A = np.asarray((HT @ Qt[D.Q.index.get_indexer(sub.id)].T).todense(), np.float32)
            St = mot.similitudes_t(A, HT)
        P = np.add.reduceat(np.exp(-mot.costos(S, St, cfg)), mot.ini, axis=0)
        virt = mot.virtuales(D, sub.office.values, Q, sub.id.values)
        f = observables_gpu if gpu else observables_cpu
        out[a:a + lote] = f(mot, P, fuentes[a:a + lote], virt, cfg)
    return out


def reducir(M: np.ndarray, ok: np.ndarray, cfg: dict) -> np.ndarray:
    """De los observables de un recorrido (licitaciones × empresas del grafo), sólo las columnas de las candidatas
    (`ok`): (2 × licitaciones × candidatas) el valor y su puesto entre todas las empresas del grafo (NaN si no la
    alcanza). Es lo único que usa `combinar`: se puede soltar el grafo antes de recorrer el otro."""
    suma = cfg["tipo"] == "suma"
    x = M[:, ok]
    p = np.full(x.shape, np.nan, np.float32)
    for q in range(len(M)):
        alc = x[q] > 0 if suma else np.isfinite(x[q])
        o = np.sort(-M[q] if suma else M[q])
        p[q, alc] = 1 + np.searchsorted(o, -x[q, alc] if suma else x[q, alc], side="left")
    return np.stack([x, p])


def combinar(R: dict, emp: np.ndarray, L: pd.DataFrame, cand: set, texto: dict) -> pd.DataFrame:
    """El pool y los observables de las licitaciones L, desde `R`: {recorrido: `reducir`}, con columnas `emp` (las
    empresas del grafo que son candidatas, en el orden del grafo)."""
    filas = []
    for q, lid in enumerate(L.id):
        por = {}
        for nom, cfg in RECORRIDOS.items():
            x = R[nom][0, q]
            if cfg["tipo"] == "suma":
                i = np.flatnonzero(x > 0)
                i = i[np.argsort(-x[i], kind="stable")[:TOP]]
            else:
                i = np.flatnonzero(np.isfinite(x))
                i = i[np.argsort(x[i], kind="stable")[:TOP]]
            por[f"pool_{nom}"] = set(emp[i])
        por["pool_texto"] = set(texto.get(lid, [])) & cand
        x = pd.DataFrame({"id": lid, "q": q, "parent_uei": sorted(set().union(*por.values()))})
        for c, e in por.items():
            x[c] = x.parent_uei.isin(e)
        filas.append(x)
    X = pd.concat(filas, ignore_index=True)
    col = pd.Index(emp).get_indexer(X.parent_uei)
    f, en, c = X.q.values, col >= 0, np.maximum(col, 0)
    for nom, cfg in RECORRIDOS.items():
        x = np.where(en, R[nom][0, f, c], 0 if cfg["tipo"] == "suma" else np.inf).astype(np.float32)
        p = np.where(en, R[nom][1, f, c], np.nan).astype(np.float32)
        if cfg["tipo"] == "suma":
            X[f"{nom}_s"] = x
            X[f"{nom}_puesto"] = p
        else:
            X[f"{nom}_puesto_c"] = p
    return X.drop(columns="q")


def corte(D: dict, grafos: dict, L: pd.DataFrame, cand: set, texto: dict, gpu: bool) -> pd.DataFrame:
    """Observables y pool de las licitaciones L (de una vertical, todas del mismo corte y semestre). `D` y `grafos`:
    {"gemb", "gtf"} -> Conjunto y Grafo; `texto`: {id: top 100 por parecido de texto}."""
    t0 = time.time()
    emp = grafos["gemb"].empresas
    assert emp.equals(grafos["gtf"].empresas)
    ok = np.asarray(emp.isin(cand))
    R = {nom: reducir(recorrer(D[cfg["grafo"]], grafos[cfg["grafo"]], nom, L, gpu), ok, cfg)
         for nom, cfg in RECORRIDOS.items()}
    X = combinar(R, emp.values[ok], L, cand, texto)
    print(f"    {len(L):,} licitaciones, {len(X) / max(len(L), 1):.0f} candidatas por lista ({time.time() - t0:.0f} s)",
          flush=True)
    return X
