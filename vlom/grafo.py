"""El grafo conjunto de oficinas y empresas de las cinco verticales: construcción y guardado.

Un grafo por corte (el 1.º de cada mes), con los contratos de fecha anterior al corte. Dos grafos por corte, con la
misma construcción y distinta representación del texto: `gemb` (embeddings `bge-small`) y `gtf` (la LSA del TF-IDF).

- **Ronda 0**: una arista oficina–empresa por (oficina, empresa, texto) de los contratos con oficina; peso 1 y vector
  = el embedding del contrato.
- **Ronda 1 (oficinas)**: dos oficinas que le compraron cosas parecidas a la misma empresa (contratos e1, e2 con coseno
  s ≥ s0) se unen con peso f(s) = ((s − s0) / (1 − s0))^p y vector norm(e1 + e2).
- **Ronda 2 (empresas)** y **3 (oficinas)**: lo mismo, alternando capas, a través de una arista E de la ronda anterior
  (o de un mismo nodo): c1 –e1– o1 –E– o2 –e2– c2 con e1, e2 parecidos a E da una arista c1–c2 con peso
  w_E · f(cos(e1, e2)) y vector norm(e1 + e2 + γ·E).
- Por par de nodos, las M aristas de mayor peso; el peso medio de cada ronda se lleva a `peso_medio`; como mucho `tope`
  aristas por ronda.
- **Aristas inferidas** (en la ronda 0): cada contrato sin oficina (los anteriores a 2022) se une a las `k_inf`
  oficinas con el contrato con oficina más parecido (s ≥ s0), con peso κ·f(s).

Sin fuga: el grafo de un corte usa sólo contratos con fecha < corte, y una licitación se recorre con el grafo del 1.º
de su mes. El grafo TF-IDF usa la LSA (ajustada con contratos anteriores al 1/7/2023 y licitaciones de train) y los
umbrales que dejan pasar la misma fracción de pares que los del grafo de embeddings.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import tablas as T
from .config import CONSTRUCCION, DATOS, FUENTE, GEN, GRAFOS, VERTICALES, W_TITULO_GRAFO
from .embeddings import lic_conjunto, tfidf_conjunto, umbrales
from .texto import consulta_tfidf, emb_historial, emb_licitaciones, mezclar

CARPETA = GEN / "grafos"
ARREGLOS = ("u", "v", "w", "r", "ti", "tj", "kt", "k_o", "madre")


# ---------------------------------------------------------------- datos

class Conjunto:
    """Las cinco verticales juntas, vistas con una representación del texto: los vectores de los textos del historial
    (`H`, `textos`), los contratos con y sin oficina, y las licitaciones de las fuentes (id "<v>:<id>") con su
    consulta (`Q`) y sus candidatas."""

    def __init__(self, modelo: str, fuentes: tuple[str, ...] = (FUENTE,)):
        self.modelo, self.fuentes = modelo, fuentes
        self.H, self.textos = emb_historial(modelo)
        h = pd.concat([T.historial_crudo(v) for v in VERTICALES], ignore_index=True)
        h["fecha"] = pd.to_datetime(h.fecha)
        con = h.office.fillna("") != ""
        self.h = h[con].reset_index(drop=True)
        self.h_sin = h[~con & (h.descripcion != "")].reset_index(drop=True)
        lic, Q, self.cand = [], [], {}
        for f in fuentes:
            idx, Et, Ed = emb_licitaciones(modelo, f)
            Q.append(pd.Series(list(mezclar(Et, Ed, W_TITULO_GRAFO)), index=idx))
            for v in VERTICALES:
                x = T.licitaciones(v, f)[["id", "t", "office", "semestre", "split"]]
                lic.append(x.assign(id=v + ":" + x.id.astype(str), vertical=v))
                self.cand.update({(v, s): c for s, c in T.candidatas(v, f).items()})
        self.lic = pd.concat(lic, ignore_index=True)
        self.Q = pd.concat(Q)
        assert self.Q.index.is_unique and self.lic.id.is_unique
        self.org = pd.read_parquet(DATOS / "office.parquet").drop_duplicates("codigo").set_index("codigo")

    def tfidf(self, corte: str):
        """(TF-IDF de los textos del historial, de las consultas en el orden de `Q`) ajustado con lo anterior al
        corte. Se calcula una vez por corte y se comparte entre los dos Conjunto (mismos textos y licitaciones)."""
        clave = (corte, self.fuentes)
        if clave not in _TFIDF:
            _TFIDF.clear()
            tf, bp = tfidf_conjunto(corte)
            lic = pd.concat([lic_conjunto(f) for f in self.fuentes]).set_index("id").reindex(self.Q.index)
            _TFIDF[clave] = (tf(self.textos), tf(consulta_tfidf(lic, bp)), self.textos, self.Q.index)
        HT, Qt, textos, qidx = _TFIDF[clave]
        pos = qidx.get_indexer(self.Q.index)         # los dos Conjunto pueden tener las licitaciones en otro orden
        assert textos.equals(self.textos) and (pos >= 0).all()
        return HT, Qt[pos]


_TFIDF: dict = {}


def conjuntos(fuentes: tuple[str, ...] = (FUENTE,)) -> dict[str, Conjunto]:
    return {g: Conjunto(modelo, fuentes) for g, modelo in GRAFOS.items()}


def construccion(g: str) -> dict:
    """Parámetros de construcción del grafo `g`: los comunes y, en el TF-IDF, sus umbrales."""
    return {**CONSTRUCCION, **(umbrales() if g == "gtf" else {})}


# ---------------------------------------------------------------- construcción

class Acumulador:
    """Aristas candidatas (y1 < y2, contratos i, j, arista fuente s, peso); conserva las M de mayor peso por par."""

    def __init__(self, ny: int, M: int):
        self.ny, self.M, self.partes, self.n, self.base = ny, M, [], 0, 0

    def agregar(self, y1, y2, i, j, s, w):
        a, b = np.minimum(y1, y2), np.maximum(y1, y2)
        self.partes.append((a.astype(np.int64) * self.ny + b, i.astype(np.int32), j.astype(np.int32),
                            s.astype(np.int32), w.astype(np.float32)))
        self.n += len(a)
        if self.n > max(4_000_000, 2 * self.base):         # compactar cuando se duplica lo conservado
            self.compactar()

    def compactar(self):
        if not self.partes:
            return
        k, i, j, s, w = (np.concatenate(c) for c in zip(*self.partes))
        o = np.lexsort((-w, k))                              # estable: en empates gana la que se agregó antes
        k, i, j, s, w = k[o], i[o], j[o], s[o], w[o]
        ini = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
        rango = np.arange(len(k)) - np.repeat(ini, np.diff(np.r_[ini, len(k)]))
        m = rango < self.M
        self.partes = [(k[m], i[m], j[m], s[m], w[m])]
        self.n = self.base = int(m.sum())

    def resultado(self):
        self.compactar()
        if not self.partes:
            z = np.zeros(0, np.int64)
            return z, z, z, z, np.zeros(0, np.float32)
        return self.partes[0]


def vectores(i, j, s, B, FV, gamma) -> np.ndarray:
    """norm(e_i + e_j + γ·E_s) (E_s: vector de la arista fuente; s = −1 es la identidad)."""
    V = B[i] + B[j]
    if FV is not None and gamma > 0:
        con = s >= 0
        V[con] += gamma * FV[s[con]]
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    return V


def f_peso(s: np.ndarray, s0: float, p: float) -> np.ndarray:
    return np.clip((s - s0) / (1 - s0), 0, 1) ** p


class Grafo:
    """Nodos: oficinas 0..No−1, empresas No..No+Ne−1. Aristas (u, v, peso w, ronda r, textos ti y tj de los dos
    contratos que la generan, madre = arista de la ronda anterior que la generó). `V` (vector de cada arista) no se
    guarda: se reconstruye con `reconstruir_v`."""

    def __init__(self, D: Conjunto, corte: str, prm: dict, verbose: bool = True):
        self.prm, self.corte = prm, corte
        c = pd.Timestamp(corte)
        h = D.h[D.h.fecha < c].drop_duplicates(["office", "parent_uei", "descripcion"])
        hs = D.h_sin[D.h_sin.fecha < c].drop_duplicates(["parent_uei", "descripcion"])
        self.oficinas = pd.Index(sorted(h.office.unique()))
        self.empresas = pd.Index(sorted(set(h.parent_uei) | set(hs.parent_uei)))
        self.No = No = len(self.oficinas)
        self.k_o = k_o = self.oficinas.get_indexer(h.office).astype(np.int32)
        k_e = self.empresas.get_indexer(h.parent_uei).astype(np.int32)
        self.kt = D.textos.get_indexer(h.descripcion).astype(np.int32)
        self._derivados(D)
        B = self.B
        self.u, self.v = [k_o], [No + k_e]
        self.ti, self.tj = [self.kt], [np.full(len(B), -1, np.int32)]
        self.w, self.V, self.r = [np.ones(len(B), np.float32)], [B.astype(np.float16)], [np.zeros(len(B), np.int8)]
        self.madre = [np.full(len(B), -1, np.int64)]
        n_acum, off_madre = len(B), 0
        t0 = time.time()
        nodo = {"o": k_o, "e": k_e}
        fuente = None                                        # aristas de la ronda anterior, en índices de su capa
        for r in range(1, prm["rondas"] + 1):
            Y = "o" if r % 2 else "e"                        # la ronda 1 une oficinas
            X = "e" if Y == "o" else "o"
            ny = No if Y == "o" else len(self.empresas)
            yu, yv, w, V, ci, cj, cs = self._ronda(nodo[X], nodo[Y], ny, fuente, r <= 2)
            self.madre.append(np.where(cs >= 0, off_madre + cs, -1))
            off_madre, n_acum = n_acum, n_acum + len(w)
            off = 0 if Y == "o" else No
            self.u.append(off + yu), self.v.append(off + yv), self.w.append(w)
            self.ti.append(np.where(ci >= 0, self.kt[np.maximum(ci, 0)], -1).astype(np.int32))
            self.tj.append(np.where(cj >= 0, self.kt[np.maximum(cj, 0)], -1).astype(np.int32))
            self.V.append(V), self.r.append(np.full(len(w), r, np.int8))
            fuente = (yu, yv, w, V.astype(np.float32))
            if verbose:
                print(f"    ronda {r}: {len(w):,} aristas ({time.time() - t0:.0f} s)", flush=True)
            if not len(w):
                break
        self.n_inferidas = 0
        if len(hs):
            self._inferidas(D, hs, verbose)
        self.u, self.v = np.concatenate(self.u), np.concatenate(self.v)
        self.w, self.V, self.r = np.concatenate(self.w), np.concatenate(self.V), np.concatenate(self.r)
        self.ti, self.tj = np.concatenate(self.ti), np.concatenate(self.tj)
        self.madre = np.concatenate(self.madre + [np.full(len(self.w) - n_acum, -1, np.int64)])
        self.n_nodos = No + len(self.empresas)

    def _derivados(self, D: Conjunto):
        """Vectores de los contratos de la ronda 0, centroide, subagencia y departamento de cada oficina."""
        self.B = D.H[self.kt].astype(np.float32)
        C = np.zeros((len(self.oficinas), self.B.shape[1]), np.float32)
        np.add.at(C, self.k_o, self.B)
        self.centroide = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
        self.subag = D.org.subagencia.reindex(self.oficinas).values
        self.depto = D.org.departamento.reindex(self.oficinas).values

    def _ronda(self, kx, ky, ny, fuente, identidad):
        p, B = self.prm, self.B
        ac = Acumulador(ny, p["M"])
        orden = np.argsort(kx, kind="stable")
        ini = np.searchsorted(kx[orden], np.arange(kx.max() + 2))
        grupo = lambda x: orden[ini[x]:ini[x + 1]]  # noqa: E731
        if identidad:            # x1 = x2: contratos del mismo nodo X con distinto extremo Y
            for x in range(len(ini) - 1):
                g = grupo(x)
                if len(g) < 2:
                    continue
                S = B[g] @ B[g].T
                a, b = np.nonzero(np.triu(S >= p["s0"], 1))
                m = ky[g[a]] != ky[g[b]]
                a, b = a[m], b[m]
                if len(a):
                    ac.agregar(ky[g[a]], ky[g[b]], g[a], g[b], np.full(len(a), -1), f_peso(S[a, b], p["s0"], p["p"]))
        if fuente is not None and len(fuente[0]):
            fu, fv, fw, FV = fuente
            listas = []
            for lado in (fu, fv):    # contratos de cada extremo parecidos al vector de la arista fuente
                L = [None] * len(lado)
                o2 = np.argsort(lado, kind="stable")
                cortes = np.r_[0, np.flatnonzero(np.diff(lado[o2])) + 1, len(o2)]
                for a, b in zip(cortes[:-1], cortes[1:]):
                    es = o2[a:b]
                    g = grupo(lado[es[0]])
                    if not len(g):
                        continue
                    A = FV[es] @ B[g].T
                    for fila, e in enumerate(es):
                        sel = np.flatnonzero(A[fila] >= p["s_fuente"])
                        if len(sel) > p["k_lado"]:
                            sel = sel[np.argpartition(-A[fila, sel], p["k_lado"])[:p["k_lado"]]]
                        L[e] = g[sel]
                listas.append(L)
            for e in range(len(fu)):
                a1, a2 = listas[0][e], listas[1][e]
                if a1 is None or a2 is None or not len(a1) or not len(a2):
                    continue
                S = B[a1] @ B[a2].T
                i, j = np.nonzero(S >= p["s0"])
                m = ky[a1[i]] != ky[a2[j]]
                i, j = a1[i[m]], a2[j[m]]
                if len(i):
                    w = fw[e] * f_peso((B[i] * B[j]).sum(1), p["s0"], p["p"])
                    ac.agregar(ky[i], ky[j], i, j, np.full(len(i), e), w)
        k, i, j, s, w = ac.resultado()
        if len(w):
            w = np.minimum(w * p["peso_medio"] / w.mean(), 1)
        idx = np.flatnonzero(w >= p["w_min"])
        if len(idx) > p["tope"]:
            idx = idx[np.argpartition(-w[idx], p["tope"])[:p["tope"]]]
        k, w = k[idx], w[idx]
        V = vectores(i[idx], j[idx], s[idx], B, fuente[3] if fuente is not None else None, p["gamma"])
        return ((k // ny).astype(np.int32), (k % ny).astype(np.int32), w.astype(np.float32), V.astype(np.float16),
                i[idx].astype(np.int64), j[idx].astype(np.int64), s[idx].astype(np.int64))

    def _inferidas(self, D: Conjunto, hs: pd.DataFrame, verbose: bool, top: int = 64):
        """Contratos sin oficina -> aristas empresa–oficina: por contrato, las `k_inf` oficinas con el contrato con
        oficina más parecido (s ≥ s0), peso κ·f(s), vector norm(e + e'); por par, las M de mayor peso. Los cosenos y
        el top 64 de cada contrato, en la GPU si hay."""
        p, B, t0 = self.prm, self.B, time.time()
        E = D.H[D.textos.get_indexer(hs.descripcion)].astype(np.float32)
        ke = self.empresas.get_indexer(hs.parent_uei)
        partes = []
        for a, s, j in _top_cosenos(E, B, top):
            x = pd.DataFrame({"i": np.repeat(np.arange(a, a + len(s)), s.shape[1]), "j": j.ravel(), "s": s.ravel()})
            x = x[x.s >= p["s0"]]
            x["o"] = self.k_o[x.j.values]
            x = x.sort_values("s", ascending=False).drop_duplicates(["i", "o"])
            partes.append(x.groupby("i", sort=False).head(p["k_inf"]))
        x = pd.concat(partes)
        x["e"] = ke[x.i.values]
        x["w"] = p["kappa"] * f_peso(x.s.values, p["s0"], p["p"])
        x = x.sort_values("w", ascending=False).groupby(["e", "o"], sort=False).head(p["M"])
        x = x[x.w >= p["w_min"]]
        V = E[x.i.values] + B[x.j.values]
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
        self.u.append(x.o.values.astype(np.int32)), self.v.append((self.No + x.e.values).astype(np.int32))
        self.w.append(x.w.values.astype(np.float32)), self.V.append(V.astype(np.float16))
        self.r.append(np.zeros(len(x), np.int8))
        self.ti.append(D.textos.get_indexer(hs.descripcion.values[x.i.values]).astype(np.int32))
        self.tj.append(self.kt[x.j.values])
        self.n_inferidas = len(x)
        if verbose:
            print(f"    inferidas: {len(hs):,} contratos sin oficina -> {len(x):,} aristas ({time.time() - t0:.0f} s)",
                  flush=True)


def _top_cosenos(E: np.ndarray, B: np.ndarray, top: int):
    """Por bloques de filas de E: (inicio, los `top` cosenos más altos contra B, sus columnas)."""
    try:
        import torch
        gpu = torch.cuda.is_available()
    except ImportError:
        gpu = False
    if gpu:
        Bg = torch.as_tensor(B, device="cuda")
        bloque = max(64, int(1e9 // (4 * len(B))))
        for a in range(0, len(E), bloque):
            s, j = torch.topk(torch.as_tensor(E[a:a + bloque], device="cuda") @ Bg.T, min(top, len(B)), dim=1)
            yield a, s.cpu().numpy(), j.cpu().numpy()
        del Bg
        torch.cuda.empty_cache()
        return
    for a in range(0, len(E), 512):
        S = E[a:a + 512] @ B.T
        j = np.argpartition(-S, top, axis=1)[:, :top]
        yield a, np.take_along_axis(S, j, 1), j


def reconstruir_v(D: Conjunto, G: Grafo) -> np.ndarray:
    """V de cada arista (float16) desde los embeddings de los contratos, ronda por ronda: ronda 0 = e_ti; inferidas =
    norm(e_ti + e_tj); ronda r ≥ 1 = norm(e_ti + e_tj + γ·V_madre). Idéntico a lo que calcula la construcción."""
    V = np.zeros((len(G.r), D.H.shape[1]), np.float16)
    for k in range(int(G.r.max()) + 1):
        e = np.flatnonzero(G.r == k)
        uno = e[G.tj[e] < 0]
        V[uno] = D.H[G.ti[uno]]
        dos = e[G.tj[e] >= 0]
        for a in range(0, len(dos), 200_000):
            x = dos[a:a + 200_000]
            X = D.H[G.ti[x]].astype(np.float32) + D.H[G.tj[x]].astype(np.float32)
            con = G.madre[x] >= 0
            if G.prm["gamma"] > 0 and con.any():
                X[con] += G.prm["gamma"] * V[G.madre[x[con]]].astype(np.float32)
            X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
            V[x] = X.astype(np.float16)
    return V


# ---------------------------------------------------------------- guardado

def carpeta(g: str, corte: str):
    return CARPETA / g / corte


def armar(D: Conjunto, g: str, corte: str) -> None:
    """Arma y guarda el grafo `g` del corte (sin `V`); si ya está, no hace nada."""
    d = carpeta(g, corte)
    if (d / "meta.json").exists():
        return
    t0 = time.time()
    G = Grafo(D, corte, construccion(g), verbose=False)
    d.mkdir(parents=True, exist_ok=True)
    for a in ARREGLOS:
        np.save(d / f"{a}.npy", getattr(G, a))
    pd.DataFrame({"codigo": G.oficinas}).to_parquet(d / "oficinas.parquet", index=False)
    pd.DataFrame({"parent_uei": G.empresas}).to_parquet(d / "empresas.parquet", index=False)
    meta = {"grafo": g, "corte": corte, "prm": G.prm, "No": int(G.No), "n_nodos": int(G.n_nodos),
            "n_aristas": int(len(G.w)), "n_inferidas": int(G.n_inferidas)}
    (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")       # al final: marca de completo
    print(f"  grafo {g} {corte}: {len(G.w):,} aristas ({time.time() - t0:.0f} s)", flush=True)


def cargar(D: Conjunto, g: str, corte: str, con_v: bool) -> Grafo:
    """El grafo guardado; con `con_v`, con los vectores de las aristas (los usa el recorrido con β > 0)."""
    d = carpeta(g, corte)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    G = object.__new__(Grafo)
    G.prm, G.corte, G.No, G.n_nodos, G.n_inferidas = meta["prm"], corte, meta["No"], meta["n_nodos"], meta["n_inferidas"]
    for a in ARREGLOS:
        setattr(G, a, np.load(d / f"{a}.npy"))
    G.oficinas = pd.Index(pd.read_parquet(d / "oficinas.parquet").codigo.values)
    G.empresas = pd.Index(pd.read_parquet(d / "empresas.parquet").parent_uei.values)
    G._derivados(D)
    G.V = reconstruir_v(D, G) if con_v else None
    return G
