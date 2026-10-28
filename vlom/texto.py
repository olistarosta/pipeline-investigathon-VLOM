"""Texto: boilerplate, TF-IDF y acceso a los embeddings guardados.

Sin fuga: el vocabulario, el IDF y la lista de boilerplate se ajustan con licitaciones de train y contratos
anteriores al corte; los embeddings son de modelos preentrenados (no se ajustan).
"""
from __future__ import annotations

import functools
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import tablas as T
from .config import CORTE_TFIDF, GEN, TOKEN


def lista_boilerplate(desc_train: pd.Series) -> set:
    """5-gramas que aparecen en al menos max(5, 1 %) de las descripciones de train."""
    from sklearn.feature_extraction.text import CountVectorizer
    cv5 = CountVectorizer(lowercase=True, token_pattern=TOKEN, ngram_range=(5, 5), binary=True,
                          min_df=max(5, int(0.01 * len(desc_train))), dtype=np.int32)
    try:
        return set(cv5.fit(desc_train).get_feature_names_out())
    except ValueError:
        return set()


def sin_boilerplate(s: str, bp: set) -> str:
    """Los tokens en minúscula, sin los 5-gramas de boilerplate."""
    w = re.findall(TOKEN, s.lower())
    if len(w) < 5 or not bp:
        return " ".join(w)
    marca = np.zeros(len(w), bool)
    for i in range(len(w) - 4):
        if " ".join(w[i:i + 5]) in bp:
            marca[i:i + 5] = True
    return " ".join(x for x, m in zip(w, marca) if not m)


def consulta_tfidf(lic: pd.DataFrame, bp: set) -> pd.Series:
    """Título ×3 + descripción sin boilerplate."""
    return (lic.title + " ") * 3 + lic.description.map(lambda x: sin_boilerplate(x, bp))


class TfIdf:
    """Uni y bigramas, min_df 3, max_df 0,5, 100.000 términos, sublinear. Con más de 350.000 documentos, el vocabulario
    se arma con una muestra fija. Llamado con textos, devuelve su TF-IDF normalizado (l2)."""

    def __init__(self, corpus: pd.Series):
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        if len(corpus) > 350_000:
            corpus = corpus.sample(350_000, random_state=7)
        self.cv = CountVectorizer(lowercase=True, token_pattern=TOKEN, stop_words="english", ngram_range=(1, 2),
                                  min_df=3, max_df=0.5, max_features=100_000, dtype=np.float32)
        self.tt = TfidfTransformer(sublinear_tf=True).fit(self.cv.fit_transform(corpus))

    def __call__(self, x) -> sp.csr_matrix:
        from sklearn.preprocessing import normalize
        x = np.asarray(x, dtype=object)
        return sp.vstack([normalize(self.tt.transform(self.cv.transform(x[i:i + 200_000])))
                          for i in range(0, len(x), 200_000)]).astype(np.float32).tocsr()


@functools.lru_cache(maxsize=None)
def tfidf_vertical(v: str):
    """(TF-IDF, boilerplate) de la vertical de la vertical: licitaciones de train y contratos anteriores a
    `CORTE_TFIDF`."""
    lic, h = T.licitaciones(v), T.historial(v)
    tr = lic[lic.split == "train"]
    corpus = pd.concat([tr.title + " " + tr.description,
                        h.descripcion[(h.fecha < pd.Timestamp(CORTE_TFIDF)) & (h.descripcion.str.len() > 0)]])
    return TfIdf(corpus), lista_boilerplate(tr.description)


# ---------------------------------------------------------------- embeddings guardados

def carpeta_emb(modelo: str):
    return GEN / "embeddings" / modelo


@functools.lru_cache(maxsize=None)
def emb_historial(modelo: str) -> tuple[np.ndarray, pd.Index]:
    """(vectores de los textos distintos del historial, float16 l2; sus textos)."""
    d = carpeta_emb(modelo)
    return np.load(d / "hist.npy"), pd.Index(pd.read_parquet(d / "hist_textos.parquet").texto.values)


@functools.lru_cache(maxsize=None)
def emb_licitaciones(modelo: str, fuente: str) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """(índice "<v>:<id>", título, descripción sin boilerplate) de las licitaciones de la fuente."""
    d = carpeta_emb(modelo) / "lic" / fuente
    if not (d / "ids.parquet").exists():
        raise SystemExit(f"faltan los embeddings {modelo} de las licitaciones de {fuente} ({d})")
    ids = pd.read_parquet(d / "ids.parquet")
    return (pd.Index(ids.vertical + ":" + ids.id), np.load(d / "titulo.npy").astype(np.float32),
            np.load(d / "desc.npy").astype(np.float32))


def consultas(modelo: str, fuente: str, v: str, ids, w: float) -> np.ndarray:
    """w·título + (1 − w)·descripción, normalizada (sin título o sin descripción, la otra)."""
    idx, Et, Ed = emb_licitaciones(modelo, fuente)
    pos = idx.get_indexer(v + ":" + pd.Index(ids).astype(str))
    assert (pos >= 0).all(), f"hay licitaciones de {v} sin embeddings {modelo}"
    return mezclar(Et[pos], Ed[pos], w)


def mezclar(t: np.ndarray, d: np.ndarray, w: float) -> np.ndarray:
    q = w * t + (1 - w) * d
    sin_d, sin_t = ~d.any(axis=1), ~t.any(axis=1)
    q[sin_d], q[sin_t] = t[sin_d], d[sin_t]
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.where(n > 0, n, 1)


class Densos:
    """Cosenos licitación × contrato del historial de una vertical (ordenado por fecha) con un modelo de embeddings."""

    def __init__(self, v: str, modelo: str, fuente: str):
        H, textos = emb_historial(modelo)
        idx = textos.get_indexer(T.historial(v).descripcion.values)
        assert (idx >= 0).all(), "hay contratos sin embedding"
        filas, self.idx_h = np.unique(idx, return_inverse=True)      # sólo los textos de la vertical
        self.U = H[filas].astype(np.float32)
        self.v, self.modelo, self.fuente = v, modelo, fuente

    def sims(self, ids, w: float, nmax: int) -> np.ndarray:
        """(nmax primeros contratos por fecha × licitaciones) cosenos."""
        return (self.U @ consultas(self.modelo, self.fuente, self.v, ids, w).T)[self.idx_h[:nmax]]
