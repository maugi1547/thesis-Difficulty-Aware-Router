"""
state_dict M0 (tools/m0_state_dict.py) -> checkpoint DualBranchDetector (arsitektur final, lingkungan stok).

Untuk VALIDASI KODE saja: menjalankan fit_gate/evaluate router_final pada bobot M0 harus mereproduksi angka
PyTorch FP32 Tahap 4 (gate, thr0, aktivasi, mAP). Bukan model hasil resep final.

    python tools/convert_m0.py --sd m0_sd.pt --out m0_final.pt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfinal.compat import map_m0_state_dict  # noqa: E402
from rfinal.env import require_stock_ultralytics  # noqa: E402
from rfinal.model import DualBranchDetector  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sd", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
require_stock_ultralytics()
d = torch.load(a.sd, map_location="cpu", weights_only=False)
sd, dropped = map_m0_state_dict(d["state_dict"])
nc = int(sd["model.35.cv3.0.2.weight"].shape[0])
m = DualBranchDetector(nc=nc, verbose=False)
m.load_state_dict(sd, strict=True)
m.names = d.get("names") or m.names
torch.save(dict(model=m, ema=None, epoch=d.get("epoch"), train_args=d.get("train_args"),
                converted_from=d.get("source"), dropped_keys=dropped), a.out)
print(f"✅ {a.out} ({len(dropped)} kunci router dibuang)")
