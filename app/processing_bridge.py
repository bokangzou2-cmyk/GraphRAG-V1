from __future__ import annotations

import sys

from app.config import ROOT


PROCESSING_DIR = ROOT / "processing"
if str(PROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PROCESSING_DIR))
