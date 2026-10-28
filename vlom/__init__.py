"""Los dos pipelines finales del grupo VLOM. `b0.py` y `evaluar.py` (del repo de los organizadores, sin cambios) están
en la raíz del repo."""
import sys

from .config import RAIZ

if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
