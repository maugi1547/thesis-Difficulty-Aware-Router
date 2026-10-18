"""
Ekstrak state_dict M0 (checkpoint fork lama) menjadi berkas tensor biasa, supaya bisa dibaca lingkungan stok.
Jalankan dengan fork lama di sys.path (dari root repo, branch eksperimen-uji):

    python router_final/tools/m0_state_dict.py --ckpt M0/weights/last.pt --out m0_sd.pt
Lalu (lingkungan stok): python tools/convert_m0.py --sd m0_sd.pt --out m0_final.pt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
src = ck["ema"] if ck.get("ema") is not None else ck["model"]
torch.save(dict(state_dict={k: (v.float() if v.is_floating_point() else v) for k, v in src.state_dict().items()},
                names=dict(getattr(src, "names", {})), epoch=ck.get("epoch"), train_args=ck.get("train_args"),
                source=str(a.ckpt)), a.out)
print("✅", a.out)
