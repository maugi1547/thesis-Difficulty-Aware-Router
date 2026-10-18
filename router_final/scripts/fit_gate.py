"""
Tahap 2 — gate post-hoc di atas detektor beku.

    python scripts/fit_gate.py --config configs/kitti.yaml --weights RUN/weights/last.pt --out RUN/gate

Langkah: cache train (fitur 960 + deteksi A/B) -> target Δ soft_small -> StandardScaler + LogReg (C) -> lipat jadi
(w, b) -> skor val -> thr0 = kuantil (1 − ρ) skor val (threshold awal B2).
Keluaran: gate.json (w, b, thr0, info), target_train.csv, skor_val.csv, provenance.json.
"""
import argparse
from pathlib import Path

import _path  # noqa: F401
import numpy as np
import pandas as pd
import torch

from rfinal.budget import thr0_from_val
from rfinal.data import load_data, make_loader
from rfinal.env import load_config, provenance, sha256_file, write_json
from rfinal.gate import cache_split, fit_gate, save_gate, score_np, soft_small_delta
from rfinal.model import load_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    g = cfg["gate"]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model, info = load_detector(a.weights, a.device)
    data = load_data(cfg["data"], cfg.get("expected_counts"))
    imgsz = int(cfg["train"]["imgsz"])

    print(f"[1/3] cache {g['fit_split']} (fitur + deteksi A/B) ...")
    tr_loader = make_loader(data, g["fit_split"], 1, imgsz, a.workers)
    Ctr = cache_split(model, tr_loader, a.device, g["nms"])
    T = soft_small_delta(Ctr, tr_loader.dataset, data["names"], out / f"gt_{g['fit_split']}.json",
                          tuple(cfg["eval"]["coco_max_dets"]))
    pd.DataFrame({"file": Ctr["file"], "delta_soft_small": T["delta"], "n_gt_small": T["n_gt_small"]}).to_csv(
        out / f"target_{g['fit_split']}.csv", index=False, float_format="%.17g")

    print("[2/3] fit gate ...")
    gate = fit_gate(Ctr["feat"], T["delta"], C=float(g["C"]), max_iter=int(g["max_iter"]))
    print("   ", gate["info"])

    print("[3/3] skor val -> thr0 ...")
    va_loader = make_loader(data, "val", 1, imgsz, a.workers)
    Cva = cache_split(model, va_loader, a.device, g["nms"], dets=False)
    s_va = score_np(Cva["feat"], gate)
    rho = float(cfg["budget"]["rho"])
    thr0 = thr0_from_val(s_va, rho)
    pd.DataFrame({"file": Cva["file"], "score": s_va}).to_csv(out / "skor_val.csv", index=False, float_format="%.17g")
    print(f"    thr0 = {thr0:+.6f} | aktivasi val pada thr0 = {(np.float32(s_va) > np.float32(thr0)).mean():.4f}")

    save_gate(out / "gate.json", gate, extra=dict(
        thr0=thr0, rho=rho, weights=str(a.weights), weights_sha256=sha256_file(a.weights), detector_info=info,
        target=g["target"], fit_split=g["fit_split"], n_val=len(s_va),
        stats_train_A=T["stats_A"], stats_train_B=T["stats_B"]))
    write_json(out / "provenance.json", provenance(files=[cfg["_config_path"], cfg["data"], a.weights]))
    print("✅", out / "gate.json")


if __name__ == "__main__":
    main()
