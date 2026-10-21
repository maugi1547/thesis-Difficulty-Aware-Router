"""
Ablasi pasca-training (G1 target, G3 fitur, G4 classifier, G5 biaya label, B anggaran) di atas detektor beku.

    python scripts/ablate.py --config configs/kitti.yaml --weights RUN/weights/last.pt --out RUN/ablasi

Pilihan desain dinilai di VAL (gain mAP_small vs acak, ρ = 0,5); seri B dilaporkan pada aliran TEST (ketahanan
aktivasi, bukan seleksi). Keluaran: ablasi.csv + ringkasan per seri di stdout + provenance.json.
Ablasi deployment (A3 vs rute paksa, B2 terkunci vs efisien, FP16 vs FP32), routing acak, heuristik tanpa label,
kaskade, dan campuran V/P2S dihitung oleh scripts/suite.py.
"""
import argparse
from pathlib import Path

import _path  # noqa: F401
import torch

from rfinal.ablation import Split, per_image_losses, run_ablation
from rfinal.data import load_data, make_loader
from rfinal.env import load_config, provenance, require_stock_ultralytics, write_json
from rfinal.model import load_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-test", action="store_true", help="lewati seri B (tanpa cache test)")
    ap.add_argument("--no-loss", action="store_true", help="lewati target selisih loss (G1)")
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    require_stock_ultralytics()
    cfg = load_config(a.config)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model, _ = load_detector(a.weights, device)
    data = load_data(cfg["data"], cfg.get("expected_counts"))
    imgsz, nms, md = int(cfg["train"]["imgsz"]), cfg["gate"]["nms"], tuple(cfg["eval"]["coco_max_dets"])
    ld = lambda s: make_loader(data, s, 1, imgsz, a.workers)

    print("[1/4] cache train ...")
    tr_loader = ld(cfg["gate"]["fit_split"])
    tr = Split(model, tr_loader, device, nms, out / "gt_train.json", md)
    losses = None
    if not a.no_loss:
        print("[2/4] loss per gambar train (target selisih loss) ...")
        losses = per_image_losses(model, tr_loader, device)
    print("[3/4] cache val ...")
    va = Split(model, ld("val"), device, nms, out / "gt_val.json", md)
    te = None
    if not a.no_test:
        print("[4/4] cache test (seri B) ...")
        te = Split(model, ld("test"), device, nms, out / "gt_test.json", md)
    T = run_ablation(tr, va, te, losses, rho=float(cfg["budget"]["rho"]), C=float(cfg["gate"]["C"]),
                     n_random=a.n_random, b2=(int(cfg["budget"]["window"]), int(cfg["budget"]["min_hist"])))
    T.to_csv(out / "ablasi.csv", index=False, float_format="%.6g")
    keys = ["seri", "varian", "split", "aktivasi", "gain_mAP_small", "gain_mAP50", "gain_mAP50-95"]
    show = T.groupby(["seri", "varian", "split"], sort=False)[keys[3:]].mean().reset_index()
    print(show.round(4).to_string(index=False))
    write_json(out / "provenance.json", provenance(files=[cfg["_config_path"], a.weights],
                                                   extra=dict(n_random=a.n_random)))
    print("✅", out / "ablasi.csv")


if __name__ == "__main__":
    main()
