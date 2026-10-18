"""Tambahkan folder router_final/ ke sys.path (skrip boleh dijalankan dari folder mana pun)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
