"""Pytest bootstrap for the ``src`` layout.

The package lives in ``src/okh_rag`` and is not installed into the test interpreter, so
``import okh_rag`` needs ``src`` on the path. Putting that here rather than in every test
module keeps the tests free of path juggling, and it is what lets ``pytest`` be run from the
repository root without an editable install.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Windows + Anaconda: pip-installed torch and MKL-backed numpy each ship their own
# libiomp5md.dll; loading both in one process aborts with "OMP: Error #15".
# Must be set before numpy/torch initialize their OpenMP runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
