"""Embeddings de los textos: contratos del historial y licitaciones, con `bge-small`, EmbeddingGemma 2 y la LSA.

- **Historial**: cada texto distinto de los contratos de las cinco verticales, una vez, como documento. Vacío → 0.
- **Licitaciones** (por fuente): el título y la descripción sin boilerplate, por separado, como consulta. El
  boilerplate sale de las descripciones de train de la vertical.
- **LSA** (`tfidf-lsa`): el TF-IDF de las cinco verticales (contratos anteriores al 1/7/2023 y licitaciones de train)
  reducido a 256 dimensiones con una SVD ajustada con 150.000 textos de esos contratos. Es la representación con que se
  arma el grafo TF-IDF. Se guarda el modelo para proyectar licitaciones nuevas.

Salida: `generados/embeddings/<modelo>/hist.npy` (float16, l2), `hist_textos.parquet` y
`lic/<fuente>/{ids.parquet,titulo.npy,desc.npy}`.
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd

from . import tablas as T
from .config import CONSTRUCCION, CORTE_TFIDF, DIM_LSA, EMBEDDINGS, FUENTE, GEN, LSA, VERTICALES
from .texto import TfIdf, carpeta_emb, consulta_tfidf, lista_boilerplate, sin_boilerplate


def cargar_modelo(modelo: str):
    import os
    import torch
    from sentence_transformers import SentenceTransformer
    cfg = EMBEDDINGS[modelo]
    disp = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = getattr(torch, cfg["dtype"]) if disp != "cpu" else torch.float32
    m = SentenceTransformer(cfg["hf"], device=disp, token=os.environ.get("HF_TOKEN") or None,
                            model_kwargs={"dtype": dtype}, config_kwargs=cfg["config_kwargs"])
    m.max_seq_length = cfg["max_tokens"]
    lote = cfg["lote"] if disp != "cpu" else max(16, cfg["lote"] // 2)
    print(f"  {cfg['hf']} en {disp} ({dtype})", flush=True)
    return m, lote


def embeber(m, lote: int, textos, prefijo: str) -> np.ndarray:
    """Normalizado l2, float16; textos vacíos -> 0."""
    textos = list(textos)
    vacio = np.array([len(x.strip()) == 0 for x in textos], bool)
    llenos = [prefijo + x for x, v in zip(textos, vacio) if not v] or [prefijo + "-"]
    E = m.encode(llenos, batch_size=lote, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    out = np.zeros((len(textos), E.shape[1]), np.float16)
    out[~vacio] = E[:(~vacio).sum()].astype(np.float16)
    return out


def historial(modelo: str) -> None:
    d = carpeta_emb(modelo)
    if (d / "hist.npy").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    textos = T.textos_historial()
    m, lote = cargar_modelo(modelo)
    H = embeber(m, lote, textos, EMBEDDINGS[modelo]["documento"])
    np.save(d / "hist.npy", H)
    pd.DataFrame({"texto": textos}).to_parquet(d / "hist_textos.parquet", index=False)
    print(f"  {modelo}: {len(textos):,} textos del historial ({time.time() - t0:.0f} s)", flush=True)


def licitaciones(modelo: str, fuente: str) -> None:
    d = carpeta_emb(modelo) / "lic" / fuente
    if (d / "ids.parquet").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m, lote = cargar_modelo(modelo)
    pre = EMBEDDINGS[modelo]["consulta"]
    ids, Et, Ed = [], [], []
    for v in VERTICALES:
        lic = T.licitaciones(v, fuente)
        tr = T.licitaciones(v)
        bp = lista_boilerplate(tr.description[tr.split == "train"])
        ids.append(lic[["id"]].assign(vertical=v))
        Et.append(embeber(m, lote, lic.title, pre))
        Ed.append(embeber(m, lote, lic.description.map(lambda s: sin_boilerplate(s, bp)), pre))
    np.save(d / "titulo.npy", np.concatenate(Et))
    np.save(d / "desc.npy", np.concatenate(Ed))
    pd.concat(ids, ignore_index=True).to_parquet(d / "ids.parquet", index=False)
    print(f"  {modelo}: licitaciones de {fuente} ({time.time() - t0:.0f} s)", flush=True)


# ---------------------------------------------------------------- LSA

def lic_conjunto(fuente: str) -> pd.DataFrame:
    """Las licitaciones de las cinco verticales en el orden de sus archivos, con id "<v>:<id>"."""
    partes = []
    for v in VERTICALES:
        lic = pd.read_parquet(T.carpeta(v, fuente) / "licitaciones.parquet", columns=["id", "t", "title", "description"])
        partes.append(lic.assign(id=v + ":" + lic.id.astype(str), vertical=v))
    lic = pd.concat(partes, ignore_index=True)
    for c in ("title", "description"):
        lic[c] = lic[c].fillna("").astype(str)
    lic["t"] = pd.to_datetime(lic.t)
    return lic


def tfidf_conjunto(corte: str):
    """TF-IDF de las cinco verticales ajustado con lo anterior a `corte`: licitaciones de train con t < corte y
    contratos con fecha < corte. Devuelve (TF-IDF, boilerplate)."""
    from .config import SPLIT
    from b0 import semestre
    c = pd.Timestamp(corte)
    lic = lic_conjunto(FUENTE)
    antes = lic[(lic.t < c) & (lic.t.map(semestre).map(SPLIT) == "train")]
    h = pd.concat([T.historial_crudo(v) for v in VERTICALES], ignore_index=True)
    h = h[pd.to_datetime(h.fecha) < c].descripcion
    corpus = pd.concat([antes.title + " " + antes.description, h[h.str.len() > 0]])
    return TfIdf(corpus), lista_boilerplate(antes.description)


def lsa() -> None:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    d = carpeta_emb(LSA)
    if (d / "modelo.pkl").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    transformar, bp = tfidf_conjunto(CORTE_TFIDF)
    textos = T.textos_historial()
    HT = transformar(textos)
    h = pd.concat([T.historial_crudo(v) for v in VERTICALES], ignore_index=True)
    h = h[pd.to_datetime(h.fecha) < pd.Timestamp(CORTE_TFIDF)].descripcion
    filas = textos.get_indexer(pd.unique(h[h.str.len() > 0]))
    filas = np.random.default_rng(0).permutation(filas[filas >= 0])[:150_000]
    svd = TruncatedSVD(DIM_LSA, random_state=0).fit(HT[filas])
    np.save(d / "hist.npy", normalize(svd.transform(HT)).astype(np.float16))
    pd.DataFrame({"texto": textos}).to_parquet(d / "hist_textos.parquet", index=False)
    with open(d / "modelo.pkl", "wb") as f:
        pickle.dump({"transformar": transformar, "bp": bp, "svd": svd}, f)
    print(f"  LSA: {len(textos):,} textos, varianza explicada {svd.explained_variance_ratio_.sum():.3f} "
          f"({time.time() - t0:.0f} s)", flush=True)


def lsa_licitaciones(fuente: str) -> None:
    """La consulta TF-IDF (título ×3 + descripción sin boilerplate) proyectada con la SVD. Título y descripción
    llevan el mismo vector (la consulta ya junta los dos)."""
    from sklearn.preprocessing import normalize
    d = carpeta_emb(LSA) / "lic" / fuente
    if (d / "ids.parquet").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    with open(carpeta_emb(LSA) / "modelo.pkl", "rb") as f:
        M = pickle.load(f)
    lic = lic_conjunto(fuente)
    Q = normalize(M["svd"].transform(M["transformar"](consulta_tfidf(lic, M["bp"])))).astype(np.float16)
    np.save(d / "titulo.npy", Q)
    np.save(d / "desc.npy", Q)
    lic.assign(id=lic.id.str.split(":", n=1).str[1])[["vertical", "id"]].to_parquet(d / "ids.parquet", index=False)


def umbrales() -> dict:
    """s0 y s_fuente del grafo TF-IDF: los que dejan pasar la misma fracción de pares de contratos (con oficina, al
    azar) que 0,8 y 0,75 en el grafo de embeddings."""
    r = GEN / "grafos" / "umbrales_tfidf.json"
    if r.exists():
        return json.loads(r.read_text(encoding="utf-8"))
    from .texto import emb_historial
    He, textos = emb_historial("bge-small")
    Hl, _ = emb_historial(LSA)
    h = pd.concat([T.historial_crudo(v) for v in VERTICALES], ignore_index=True)
    k = textos.get_indexer(h.descripcion[h.office.fillna("") != ""].values)
    rng = np.random.default_rng(0)
    a, b = rng.choice(k, 300_000), rng.choice(k, 300_000)
    se = (He[a].astype(np.float32) * He[b].astype(np.float32)).sum(1)
    st = (Hl[a].astype(np.float32) * Hl[b].astype(np.float32)).sum(1)
    out = {nom: float(np.quantile(st, (se < CONSTRUCCION[nom]).mean())) for nom in ("s0", "s_fuente")}
    r.parent.mkdir(parents=True, exist_ok=True)
    r.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"  umbrales del grafo TF-IDF: {out}", flush=True)
    return out


def todo(fuente: str) -> None:
    """Todos los embeddings que necesita la fuente (el historial y la LSA, si faltan)."""
    for modelo in EMBEDDINGS:
        historial(modelo)
        licitaciones(modelo, fuente)
    lsa()
    lsa_licitaciones(fuente)
    umbrales()
