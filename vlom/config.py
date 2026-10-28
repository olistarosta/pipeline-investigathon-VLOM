"""Rutas, verticales, modelos y todos los parámetros fijos de los dos pipelines.

Nada de esto se vuelve a elegir: son los valores con que se evaluaron los pipelines en val (2023S2). Los grafos y sus
recorridos se eligieron en train para `j` y se usan igual en las cinco verticales; los umbrales del grafo TF-IDF se
recalculan con los datos (`embeddings.umbrales`).
"""
from __future__ import annotations

import pathlib

RAIZ = pathlib.Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data"                    # los datos del task, tal cual (bajar_datos.py)
GEN = RAIZ / "generados"                 # todo lo que se calcula (y el release precalculado)
PREDICCIONES = RAIZ / "predicciones"     # los rankings sobre datos nuevos

VERTICALES = ("j", "r", "d", "it", "rd")
FUENTE = "data"                          # las licitaciones de data/ (con ganadores); los datos nuevos tienen su nombre

# ---------------------------------------------------------------- split y modelos

SPLIT = {"2022S1": "train", "2022S2": "train", "2023S1": "train", "2023S2": "val", "2024S1": "test"}
MODELOS = {"todo": ("train", "val", "test"),         # por defecto: entrenado con todo data/
           "sin_test": ("train", "val"),             # sin 2024S1
           "train": ("train",)}                      # el que se evalúa en val (2023S2)
NUEVO = "nuevo"                                      # semestre y split de las licitaciones de datos nuevos

# ---------------------------------------------------------------- texto

TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b"
CORTE_TFIDF = "2023-07-01"           # TF-IDF y LSA: contratos anteriores y licitaciones de train
EMBEDDINGS = {
    "bge-small": dict(hf="BAAI/bge-small-en-v1.5", consulta="Represent this sentence for searching relevant passages: ",
                      documento="", max_tokens=512, dtype="float16", lote=256, config_kwargs=None),
    "gemma2": dict(hf="google/embeddinggemma-2", consulta="task: search result | query: ",
                   documento="title: none | text: ", max_tokens=1024, dtype="bfloat16", lote=128,
                   config_kwargs={"vision_config": None, "audio_config": None}),
}
LSA = "tfidf-lsa"                    # el TF-IDF reducido a 256 dimensiones, como un modelo de embeddings más
DIM_LSA = 256
# peso del título en la consulta (w·título + (1 − w)·descripción), elegido en train por vertical
W_TITULO = {"bge-small": {"j": 0.75, "r": 0.5, "d": 0.75, "it": 0.5, "rd": 1.0},
            "gemma2": {"j": 0.75, "r": 0.5, "d": 0.75, "it": 0.5, "rd": 0.5}}
W_TITULO_GRAFO = 0.75                # la consulta con que se recorre el grafo, en las cinco

# ---------------------------------------------------------------- grafo

INICIO_GRAFO = "2022-02-01"          # un grafo por mes; las licitaciones de enero de 2022 no tienen grafo
CONSTRUCCION = dict(rondas=3, s0=0.80, p=1.0, s_fuente=0.75, k_lado=15, M=2, gamma=1.0, w_min=0.02, tope=400_000,
                    peso_medio=0.6, kappa=0.25, k_inf=3)
GRAFOS = {"gemb": "bge-small", "gtf": LSA}           # grafo de embeddings y grafo TF-IDF (construido con la LSA)
# recorridos: (grafo, R, tipo). "suma": masa que llega desde la oficina en K pasos de la caminata; "camino": el camino
# más probable. Costo de una arista: (−log w − β·log h(coseno) − β_t·log h_t(coseno TF-IDF)) · (λ si es interna)
RECORRIDOS = {
    "gemb_R1": dict(grafo="gemb", R=1, tipo="suma", K=6, beta=8.0, s_q=0.6, lam=0.25),
    "gtf_R1": dict(grafo="gtf", R=1, tipo="suma", K=10, beta=0.0, s_q=0.6, lam=0.5, beta_t=4.0, s_ref=0.3, piso=0.05),
    "gemb_R3": dict(grafo="gemb", R=3, tipo="camino", beta=8.0, s_q=0.6, lam=0.25),
    "gtf_R3": dict(grafo="gtf", R=3, tipo="camino", beta=0.0, s_q=0.6, lam=0.25, beta_t=4.0, s_ref=0.3, piso=0.05),
}
TOP = 100                            # candidatas por fuente del pool
LOTE_CORTE = 200                     # licitaciones por archivo de observables

# ---------------------------------------------------------------- pipeline limpio

POOL = ["pool_gemb_R1", "pool_gtf_R1", "pool_gemb_R3", "pool_gtf_R3", "pool_texto"]
NIVELES = ["b0n_op", "b0n_psc", "b0n_naics", "b0n_todo", "b0n_psc_sa"]       # en el orden de b0.NIVELES
VARIABLES_LIMPIO = [
    "gemb_R1_puesto", "r_gemb_R1", "gtf_R1_puesto", "r_gtf_R1",              # grafo: suma sobre caminos (R = 1)
    "sim_max_e", "sim_top3_e", "sim_max",                                    # texto: bge-small y TF-IDF
    "frac_psc",                                                              # rubro
    "gemb_R3_puesto_c", "gtf_R3_puesto_c",                                   # grafo: camino más corto (R = 3)
    *NIVELES,                                                                # historial: los 5 niveles del B0
]
LAMBDARANK = dict(objective="lambdarank", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                  reg_lambda=1.0, lambdarank_truncation_level=30, random_state=0, verbose=-1, deterministic=True,
                  force_row_wise=True)

# ---------------------------------------------------------------- pipeline completo

NOMBRES = ["nom_esta", "nom_titulo", "nom_n"]
CE = ["ce", "ce_dif", "ce_q"]
VARIABLES_COMPLETO = VARIABLES_LIMPIO + ["sim_max_g"] + NOMBRES + CE
TOP_CE = 50                          # candidatas por licitación que puntúa el cross-encoder
CE_BASE = "cross-encoder/ms-marco-MiniLM-L12-v2"
CE_PRM = dict(max_len=384, epocas=2, lr=3e-5, grupos_por_lote=4, negativas=15)
KA, KB = 5, 3                        # evidencia: 5 contratos más parecidos + 3 más parecidos en la oficina
