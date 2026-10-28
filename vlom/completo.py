"""Pipeline completo: el limpio + EmbeddingGemma 2 + empresas nombradas + un cross-encoder afinado (22 variables).

    licitación ─► limpio: pool del grafo y 15 variables
                ─► 1. nombres: ¿la licitación nombra a la empresa?
                ─► 2. primera etapa: el limpio puntúa cada par (fuera de muestra en el entrenamiento) → top 50
                ─► 3. evidencia: los contratos previos de la empresa más parecidos a la licitación (fecha < t)
                ─► 4. cross-encoder: lee (licitación, evidencia) del top 50 → ce
                ─► 5. LambdaRank de 22 variables (las 15 del limpio + Gemma + nombres + cross-encoder) → top 100

Sin fuga (además de lo del limpio): la primera etapa y el cross-encoder son fuera de muestra en las licitaciones del
entrenamiento (un modelo por semestre, ajustado con los otros), así que ninguna tiene un puntaje de un modelo que vio
a su ganador; las demás se puntúan con el limpio del modelo y con el promedio de los cross-encoders. La evidencia usa
contratos de fecha < t y no lleva el nombre de la empresa; los nombres, sólo el texto de la licitación y las
candidatas de su semestre (el diccionario de palabras comunes sale de las licitaciones de train).
"""
from __future__ import annotations

import math
import re
import time
from collections import defaultdict

import numba
import numpy as np
import pandas as pd

from . import cache as C
from . import limpio as L
from . import tablas as T
from .config import (CE, CE_BASE, CE_PRM, DATOS, FUENTE, GEN, KA, KB, MODELOS, NOMBRES, TOP_CE, VARIABLES_COMPLETO,
                     VARIABLES_LIMPIO, VERTICALES, W_TITULO)
from .texto import Densos, consultas, lista_boilerplate, sin_boilerplate


def carpeta(fuente: str, modelo: str):
    d = GEN / fuente / "completo" / modelo
    d.mkdir(parents=True, exist_ok=True)
    return d


def semestres(modelo: str) -> list[str]:
    """Los semestres del entrenamiento del modelo (uno por fold)."""
    from .config import SPLIT
    return [s for s, sp in SPLIT.items() if sp in MODELOS[modelo]]


# ================================================================ 1. empresas nombradas

SUFIJOS = {"inc", "incorporated", "llc", "l", "c", "corp", "corporation", "co", "company", "ltd", "limited", "lp", "llp",
           "pc", "pllc", "plc", "the", "and", "of", "dba", "d", "b", "a", "na", "usa", "us", "group", "holdings",
           "international", "intl", "enterprises", "services", "service", "solutions", "systems", "technologies",
           "technology", "gmbh", "sa", "ag", "srl", "bv", "nv"}


def normalizar(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower().replace("&", " and ")).split())


def nucleo(nombre: str) -> str:
    """El nombre normalizado, sin sufijos societarios ni "the" al principio."""
    w = normalizar(nombre).split()
    while w and w[-1] in SUFIJOS:
        w.pop()
    while w and w[0] == "the":
        w.pop(0)
    return " ".join(w)


def frases(emp: pd.DataFrame, df: dict, n: int, nombres_ofi: set) -> dict:
    """núcleo del nombre -> empresas, sin nombres genéricos (una palabra con df > 0,5 % en las licitaciones de train, o
    todas sus palabras con df > 2 %) ni nombres de oficinas."""
    nuc = emp.drop_duplicates("parent_uei").assign(f=lambda d: d.name.map(nucleo))
    nuc = nuc[nuc.f.str.len() >= 5]
    out = defaultdict(list)
    for f, u in zip(nuc.f, nuc.parent_uei):
        w = f.split()
        if len(w) == 1 and df.get(w[0], 0) / n > 0.005:
            continue
        if len(w) >= 2 and all(df.get(x, 0) / n > 0.02 for x in w):
            continue
        if f not in nombres_ofi:
            out[f].append(u)
    return out


def buscar(texto: str, fr: dict, max_pal: int = 6) -> set:
    w = normalizar(texto).split()
    return {" ".join(w[i:i + k]) for i in range(len(w)) for k in range(1, max_pal + 1)
            if i + k <= len(w) and " ".join(w[i:i + k]) in fr}


def nombres(v: str, fuente: str) -> None:
    """Para cada licitación, las candidatas cuyo nombre aparece como frase en el título o la descripción."""
    r = C.carpeta(fuente, v) / "nombres.parquet"
    if r.exists():
        return
    lic, tr = T.licitaciones(v, fuente), T.licitaciones(v)
    tr = tr[tr.split == "train"]
    df = defaultdict(int)
    for s in tr.title + " " + tr.description:
        for w in set(normalizar(s).split()):
            df[w] += 1
    ofi = pd.read_parquet(DATOS / "office.parquet")
    nombres_ofi = {normalizar(x) for c in ("oficina", "subagencia", "departamento") for x in ofi[c].fillna("")}
    emp = T.empresas(v, fuente)
    filas = []
    for sem, Ls in lic.groupby("semestre"):
        fr = frases(emp[emp.semestre == sem], df, len(tr), nombres_ofi)
        for l in Ls.itertuples():
            ht, hd = buscar(l.title, fr), buscar(l.description, fr)
            ueis = {u for f in ht | hd for u in fr[f]}
            filas += [(l.id, u, float(f in ht), float(len(ueis))) for f in ht | hd for u in fr[f]]
    N = pd.DataFrame(filas, columns=["id", "parent_uei", "nom_titulo", "nom_n"]).groupby(
        ["id", "parent_uei"], as_index=False).max()
    N["nom_esta"] = np.float32(1)
    N.to_parquet(r, index=False)


# ================================================================ 2. primera etapa

def primera(modelo: str) -> None:
    """Puntaje del limpio para cada par de data/ y su puesto en la lista: en las licitaciones del entrenamiento de
    `modelo`, con el limpio entrenado sin su semestre; en las demás, con el limpio del modelo."""
    r = carpeta(FUENTE, modelo) / "primera.parquet"
    if r.exists():
        return
    t0 = time.time()
    PC = L.tabla_conjunta({v: L.tabla(v) for v in VERTICALES})
    s = L.cargar_modelo("limpio", modelo).predict(L.matriz(PC, VARIABLES_LIMPIO))
    ent = PC.split.isin(MODELOS[modelo]).values
    for sem in semestres(modelo):
        b = L.entrenar_ranker(PC[ent & (PC.semestre != sem).values], VARIABLES_LIMPIO)
        k = (PC.semestre == sem).values
        s[k] = b.predict(L.matriz(PC[k], VARIABLES_LIMPIO))
    guardar_primera(PC, s, r)
    print(f"  primera etapa {modelo} ({time.time() - t0:.0f} s)", flush=True)


def primera_nuevas(fuente: str, modelo: str) -> None:
    r = carpeta(fuente, modelo) / "primera.parquet"
    if r.exists():
        return
    PC = L.tabla_conjunta({v: L.tabla(v, fuente) for v in VERTICALES})
    guardar_primera(PC, L.cargar_modelo("limpio", modelo).predict(L.matriz(PC, VARIABLES_LIMPIO)), r)


def guardar_primera(PC: pd.DataFrame, s: np.ndarray, r) -> None:
    out = PC[["id", "parent_uei", "vertical", "j"]].assign(s1=s)
    out["puesto1"] = out.groupby("j").s1.rank(ascending=False, method="first").astype(np.int32)
    out.drop(columns="j").to_parquet(r, index=False)


def top_ce(fuente: str) -> pd.DataFrame:
    """Los pares del top 50 de la primera etapa de cualquier modelo de la fuente."""
    rs = sorted((GEN / fuente / "completo").glob("*/primera.parquet"))
    return pd.concat([pd.read_parquet(r).query("puesto1 <= @TOP_CE")[["vertical", "id", "parent_uei"]] for r in rs]
                     ).drop_duplicates().reset_index(drop=True)


# ================================================================ 3. evidencia

@numba.njit(cache=True)
def _insertar(vals, idx, txt, k, s, row, t):
    for a in range(k):
        if idx[a] >= 0 and txt[a] == t:
            return
    if idx[k - 1] >= 0 and s <= vals[k - 1]:
        return
    p = k - 1
    while p > 0 and (idx[p - 1] < 0 or s > vals[p - 1]):
        vals[p], idx[p], txt[p] = vals[p - 1], idx[p - 1], txt[p - 1]
        p -= 1
    vals[p], idx[p], txt[p] = s, row, t


@numba.njit(parallel=True, cache=True)
def _kernel(fi, qcol, ncut, ofl, ptr, rows, idx_h, of_h, S, ea, eb, sa, nprev, nof):
    for i in numba.prange(len(fi)):
        f = fi[i]
        a0, a1 = ptr[f], ptr[f + 1]
        lo, hi = a0, a1
        while lo < hi:                                    # contratos con fecha < t: un prefijo
            m = (lo + hi) // 2
            if rows[m] < ncut[i]:
                lo = m + 1
            else:
                hi = m
        va = np.full(KA, -9.0, np.float32)
        ia = np.full(KA, -1, np.int64)
        ta = np.full(KA, -1, np.int64)
        vb = np.full(KB, -9.0, np.float32)
        ib = np.full(KB, -1, np.int64)
        tb = np.full(KB, -1, np.int64)
        c_of = 0
        for k in range(lo - 1, a0 - 1, -1):              # del más reciente al más viejo
            r = rows[k]
            t = idx_h[r]
            s = S[t, qcol[i]]
            _insertar(va, ia, ta, KA, s, r, t)
            if ofl[i] >= 0 and of_h[r] == ofl[i]:
                c_of += 1
                _insertar(vb, ib, tb, KB, s, r, t)
        ea[i], eb[i], sa[i] = ia, ib, va[0]
        nprev[i], nof[i] = lo - a0, c_of


def evidencia(v: str, fuente: str) -> None:
    """Para los pares del top 50: los 5 contratos previos de la empresa más parecidos a la licitación y los 3 más
    parecidos en su oficina (sin repetir textos), cuántos contratos previos tiene y cuántos en la oficina."""
    pares = top_ce(fuente).query("vertical == @v")[["id", "parent_uei"]]
    r = C.carpeta(fuente, v) / "evidencia.parquet"
    if r.exists() and len(pares.merge(pd.read_parquet(r, columns=["id", "parent_uei"]))) == len(pares):
        return
    t0 = time.time()
    h = T.historial(v)
    uei = pd.Index(pd.unique(h.parent_uei.values))
    fi_h = uei.get_indexer(h.parent_uei.values).astype(np.int64)
    rows = np.argsort(fi_h, kind="stable").astype(np.int64)                 # contratos por empresa, por fecha
    ptr = np.zeros(len(uei) + 1, np.int64)
    ptr[1:] = np.cumsum(np.bincount(fi_h, minlength=len(uei)))
    E = Densos(v, "bge-small", fuente)
    lic = T.licitaciones(v, fuente).set_index("id").loc[pares.id.unique()].sort_values("ti")
    X = pares.assign(fi=uei.get_indexer(pares.parent_uei)).merge(
        lic[["ti"]].reset_index().reset_index().rename(columns={"index": "q"}), on="id")
    assert (X.fi >= 0).all()
    X = X.sort_values(["q", "fi"]).reset_index(drop=True)
    cod, _ = pd.factorize(pd.Series(np.concatenate([h.office_u.values, lic.office_u.values])))
    of_h = np.where(h.office_u.values == "", -1, cod[:len(h)]).astype(np.int64)
    of_l = np.where(lic.office_u.values == "", -1, cod[len(h):]).astype(np.int64)
    n_t = np.searchsorted(h.ts.values, lic.ti.values, side="left")
    N = len(X)
    ea, eb = np.full((N, KA), -1, np.int64), np.full((N, KB), -1, np.int64)
    sa, nprev, nof = np.zeros(N, np.float32), np.zeros(N, np.int64), np.zeros(N, np.int64)
    q = X.q.values
    for a in range(0, len(lic), 64):
        sl = slice(np.searchsorted(q, a), np.searchsorted(q, a + 64))
        S = np.ascontiguousarray(E.U @ consultas("bge-small", fuente, v, lic.index.values[a:a + 64],
                                                 W_TITULO["bge-small"][v]).T)
        _kernel(X.fi.values[sl], (q[sl] - a).astype(np.int64), n_t[q[sl]].astype(np.int64), of_l[q[sl]], ptr, rows,
                E.idx_h.astype(np.int64), of_h, S, ea[sl], eb[sl], sa[sl], nprev[sl], nof[sl])
    out = X[["id", "parent_uei"]].copy()
    for k in range(KA):
        out[f"ea{k}"] = ea[:, k]
    for k in range(KB):
        out[f"eb{k}"] = eb[:, k]
    out["sim0"], out["n_prev"], out["n_of"] = np.where(ea[:, 0] >= 0, sa, 0), nprev, nof
    out.to_parquet(r, index=False)
    print(f"  {v}: evidencia de {N:,} pares ({time.time() - t0:.0f} s)", flush=True)


# ================================================================ 4. cross-encoder

class Textos:
    """La consulta y el documento del cross-encoder para cada par con evidencia de la fuente."""

    def __init__(self, fuente: str):
        ofi = pd.read_parquet(DATOS / "office.parquet")
        nom_of = dict(zip(ofi.codigo.astype(str).str.upper(), ofi.oficina.fillna("").astype(str).str.title()))
        psc = pd.read_parquet(DATOS / "psc.parquet")
        nom_psc = dict(zip(psc.codigo.astype(str).str.upper(), psc.descripcion.fillna("").astype(str).str.lower()))
        partes, items, self.consultas = [], [], []
        off_h = off_q = 0
        for v in VERTICALES:
            E = pd.read_parquet(C.carpeta(fuente, v) / "evidencia.parquet").assign(vertical=v)
            h = T.historial(v)
            of_n = pd.Series(h.office_u.values).map(nom_of).fillna("")
            items.append((h.descripcion.str.slice(0, 150).str.lower() + " (" + pd.Series(h.psc_u.values) + "; "
                          + of_n.where(of_n != "", "unknown office") + "; " + h.fecha.astype(str).str[:4] + ")").values)
            tr = T.licitaciones(v)
            bp = lista_boilerplate(tr.description[tr.split == "train"])
            ql = T.licitaciones(v, fuente).set_index("id").loc[E.id.unique()]
            pos = {i: k + off_q for k, i in enumerate(ql.index)}
            self.consultas += [f"{l.title.strip()}. {sin_boilerplate(l.description, bp)[:700]} | PSC: "
                               f"{nom_psc.get(l.psc_u, l.psc_u)} | office: {nom_of.get(l.office_u, 'unknown')} | "
                               f"set-aside: {str(l.sa).lower()}" for l in ql.itertuples()]
            E["qi"] = E.id.map(pos)
            for c in [f"ea{k}" for k in range(KA)] + [f"eb{k}" for k in range(KB)]:
                E[c] = np.where(E[c] >= 0, E[c] + off_h, -1)
            off_h += len(h)
            off_q += len(ql)
            partes.append(E)
        self.items = np.concatenate(items)
        self.X = pd.concat(partes, ignore_index=True)
        self.EA = self.X[[f"ea{k}" for k in range(KA)]].values
        self.EB = self.X[[f"eb{k}" for k in range(KB)]].values
        self.QI, self.NP, self.NO = self.X.qi.values, self.X.n_prev.values, self.X.n_of.values
        self.pos = pd.Series(np.arange(len(self.X)), index=pd.MultiIndex.from_frame(self.X[["vertical", "id", "parent_uei"]]))

    def par(self, i: int) -> tuple[str, str]:
        ea, eb = self.EA[i], self.EB[i]
        vistos = set(ea[ea >= 0])
        partes = [self.items[r] for r in ea if r >= 0] + [self.items[r] for r in eb if r >= 0 and r not in vistos]
        cab = f"prior contracts: {self.NP[i]}; at this office: {self.NO[i]}."
        return self.consultas[self.QI[i]], (cab + " " + " || ".join(partes)) if partes else cab + " none"

    def indices(self, G: pd.DataFrame) -> np.ndarray:
        i = self.pos.reindex(pd.MultiIndex.from_frame(G[["vertical", "id", "parent_uei"]])).values
        assert not np.isnan(i).any(), "faltan pares en la evidencia"
        return i.astype(np.int64)


def dispositivo() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _tok(tok, pares):
    return tok([p[0] for p in pares], [p[1] for p in pares], truncation="longest_first", max_length=CE_PRM["max_len"],
               padding=True, return_tensors="pt")


def puntuar(m, tok, T_: Textos, filas: np.ndarray, lote: int = 128) -> np.ndarray:
    import torch
    disp = dispositivo()
    m.eval()
    nq = np.array([len(c) for c in T_.consultas])
    orden = np.argsort(nq[T_.QI[filas]] + 150 * ((T_.EA[filas] >= 0).sum(1) + (T_.EB[filas] >= 0).sum(1)))
    out = np.zeros(len(filas), np.float32)
    with torch.no_grad(), torch.autocast(disp, dtype=torch.bfloat16, enabled=disp == "cuda"):
        for a in range(0, len(filas), lote):
            sel = orden[a:a + lote]
            out[sel] = m(**_tok(tok, [T_.par(i) for i in filas[sel]]).to(disp)).logits[:, 0].float().cpu().numpy()
    return out


def grupos(G: pd.DataFrame, rng) -> list[np.ndarray]:
    """1 ganador + 15 negativas por licitación (9 del top 20 de la primera etapa, 3 del top 20 por parecido, el resto
    al azar del top 50), en índices de Textos."""
    out = []
    for _, g in G.groupby(["vertical", "id"], sort=False):
        pos = g.t_idx.values[g.es_gan.values]
        neg = g[~g.es_gan.values]
        if len(pos) == 0 or len(neg) < 4:
            continue
        a = neg.nsmallest(20, "puesto1").t_idx.values
        el = set(rng.choice(a, min(9, len(a)), replace=False))
        b = np.setdiff1d(neg.nlargest(20, "sim0").t_idx.values, list(el))
        if len(b):
            el |= set(rng.choice(b, min(3, len(b)), replace=False))
        resto = np.setdiff1d(neg.t_idx.values, list(el))
        k = CE_PRM["negativas"] - len(el)
        if k > 0 and len(resto):
            el |= set(rng.choice(resto, min(k, len(resto)), replace=False))
        out.append(np.r_[rng.choice(pos), np.array(sorted(el))])
    return out


def afinar(T_: Textos, G: pd.DataFrame, semilla: int):
    """Afina el cross-encoder base con softmax listwise (el ganador contra sus negativas), 2 épocas."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    assert dispositivo() == "cuda", "afinar el cross-encoder necesita GPU"
    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    tok = AutoTokenizer.from_pretrained(CE_BASE)
    m = AutoModelForSequenceClassification.from_pretrained(CE_BASE, num_labels=1, attn_implementation="sdpa").cuda()
    g0 = grupos(G, rng)
    gpl = CE_PRM["grupos_por_lote"]
    pasos = CE_PRM["epocas"] * math.ceil(len(g0) / gpl)
    opt = torch.optim.AdamW(m.parameters(), lr=CE_PRM["lr"], weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * pasos), pasos)
    for ep in range(CE_PRM["epocas"]):
        gs_ep = g0 if ep == 0 else grupos(G, rng)                 # negativas nuevas cada época
        rng.shuffle(gs_ep)
        m.train()
        for a in range(0, len(gs_ep), gpl):
            gs = gs_ep[a:a + gpl]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = m(**_tok(tok, [T_.par(i) for i in np.concatenate(gs)]).to("cuda")).logits[:, 0].float()
            loss, o = 0.0, 0
            for g in gs:
                loss = loss - torch.log_softmax(s[o:o + len(g)], 0)[0]
                o += len(g)
            opt.zero_grad()
            (loss / len(gs)).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sch.step()
    del opt, sch
    torch.cuda.empty_cache()
    return m, tok


def modelos_ce(modelo: str):
    """Los cross-encoders del modelo, uno por semestre del entrenamiento: (semestre, carpeta)."""
    return [(sem, GEN / "modelos" / "completo" / modelo / "ce" / f"fold{k}") for k, sem in enumerate(semestres(modelo))]


def cargar_ce(d):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not (d / "config.json").exists():
        raise SystemExit(f"falta el cross-encoder {d}: entrenarlo o bajar el precalculado")
    return AutoModelForSequenceClassification.from_pretrained(d).to(dispositivo()), AutoTokenizer.from_pretrained(d)


def ce(modelo: str) -> None:
    """Cross-encoder fuera de muestra en data/: el fold k se afina con los otros semestres del entrenamiento y puntúa
    el suyo; las licitaciones fuera del entrenamiento, con el promedio de los folds."""
    import torch
    r = carpeta(FUENTE, modelo) / "ce.parquet"
    if r.exists():
        return
    T_ = Textos(FUENTE)
    G = pd.read_parquet(carpeta(FUENTE, modelo) / "primera.parquet").query("puesto1 <= @TOP_CE")
    tabs = pd.concat([L.tabla(v)[["id", "parent_uei", "es_gan", "semestre", "split"]].assign(vertical=v)
                      for v in VERTICALES])
    G = G.merge(tabs, on=["vertical", "id", "parent_uei"])
    G["t_idx"] = T_.indices(G)
    G = G.merge(T_.X[["vertical", "id", "parent_uei", "sim0"]], on=["vertical", "id", "parent_uei"])
    ent = G.split.isin(MODELOS[modelo]).values
    sc = np.full(len(G), np.nan, np.float32)
    otros = np.zeros((~ent).sum(), np.float32)
    folds = modelos_ce(modelo)
    for k, (sem, d) in enumerate(folds):
        t0 = time.time()
        if (d / "config.json").exists():
            m, tok = cargar_ce(d)
        else:
            m, tok = afinar(T_, G[ent & (G.semestre != sem).values], semilla=k)
            m.save_pretrained(d)
            tok.save_pretrained(d)
        mk = ent & (G.semestre == sem).values
        sc[mk] = puntuar(m, tok, T_, G.t_idx.values[mk])
        otros += puntuar(m, tok, T_, G.t_idx.values[~ent]) / len(folds)
        print(f"    cross-encoder {modelo}, fold {sem} ({time.time() - t0:.0f} s)", flush=True)
        del m
        torch.cuda.empty_cache()
    sc[~ent] = otros
    G[["vertical", "id", "parent_uei"]].assign(ce=sc).to_parquet(r, index=False)


def ce_nuevas(fuente: str, modelo: str) -> None:
    """Los pares del top 50 de datos nuevos, con el promedio de los cross-encoders del modelo."""
    r = carpeta(fuente, modelo) / "ce.parquet"
    if r.exists():
        return
    T_ = Textos(fuente)
    G = pd.read_parquet(carpeta(fuente, modelo) / "primera.parquet").query("puesto1 <= @TOP_CE")
    idx = T_.indices(G)
    sc = np.zeros(len(G), np.float32)
    for sem, d in modelos_ce(modelo):
        t0 = time.time()
        m, tok = cargar_ce(d)
        sc += puntuar(m, tok, T_, idx) / len(modelos_ce(modelo))
        print(f"    cross-encoder {modelo}, fold {sem}: {len(G):,} pares ({time.time() - t0:.0f} s)", flush=True)
    G[["vertical", "id", "parent_uei"]].assign(ce=sc).to_parquet(r, index=False)


# ================================================================ 5. LambdaRank de 22 variables

def tabla(v: str, fuente: str, modelo: str) -> pd.DataFrame:
    P = L.tabla(v, fuente)
    d = C.carpeta(fuente, v)
    P = L.pegar(P, d / "gemma.parquet", ["sim_max_g"])
    assert P.sim_max_g.notna().all()
    P = L.pegar(P, d / "nombres.parquet", NOMBRES)
    P["nom_esta"] = P.nom_esta.fillna(0).astype(np.float32)
    Cv = pd.read_parquet(carpeta(fuente, modelo) / "ce.parquet").query("vertical == @v").drop(columns="vertical")
    n = len(P)
    P = P.merge(Cv, on=["id", "parent_uei"], how="left", validate="one_to_one")
    assert len(P) == n
    g = P.groupby("id").ce
    P["ce_dif"] = (P.ce - g.transform("max")).astype(np.float32)
    P["ce_q"] = g.rank(ascending=False, method="min").astype(np.float32)
    return P


def entrenar(modelo: str) -> None:
    t0 = time.time()
    tablas = {v: tabla(v, FUENTE, modelo) for v in VERTICALES}
    PC = L.tabla_conjunta(tablas)
    b = L.entrenar_ranker(PC[PC.split.isin(MODELOS[modelo])], VARIABLES_COMPLETO)
    d = L.carpeta_modelo("completo", modelo)
    L.guardar(b, d, modelo, VARIABLES_COMPLETO, {"top_ce": TOP_CE, "cross_encoder": {"base": CE_BASE, **CE_PRM},
                                                 "folds_ce": semestres(modelo)})
    if modelo == "train":
        for v, P in tablas.items():
            va = P[P.split == "val"].reset_index(drop=True)
            L.rankear(va, b.predict(L.matriz(va, VARIABLES_COMPLETO))).to_csv(d / f"val_{v}.csv", index=False)
    print(f"  completo {modelo}: entrenado con {' + '.join(MODELOS[modelo])} ({time.time() - t0:.0f} s)", flush=True)


def predecir(fuente: str, modelo: str) -> dict[str, pd.DataFrame]:
    for v in VERTICALES:
        nombres(v, fuente)
    primera_nuevas(fuente, modelo)
    for v in VERTICALES:
        evidencia(v, fuente)
    ce_nuevas(fuente, modelo)
    b = L.cargar_modelo("completo", modelo)
    out = {}
    for v in VERTICALES:
        P = tabla(v, fuente, modelo)
        out[v] = L.rankear(P, b.predict(L.matriz(P, VARIABLES_COMPLETO)))
    return out


def preparar(modelo: str) -> None:
    """Todo lo del completo antes del LambdaRank, en data/ (necesita el limpio del modelo entrenado)."""
    for v in VERTICALES:
        nombres(v, FUENTE)
    primera(modelo)
    for v in VERTICALES:
        evidencia(v, FUENTE)
    ce(modelo)
