"""
Evaluasi sistem final pada satu split (default test): Branch A/B paksa, router (gate + B2), acak aktivasi sama.

    python scripts/evaluate.py --config configs/kitti.yaml --weights RUN/weights/last.pt --gate RUN/gate/gate.json \
        --out RUN/eval_test --backend trt --engine RUN/deploy/a3_fp16.engine
    python scripts/evaluate.py ... --backend pytorch          # acuan FP32 (tanpa engine)

Keluaran: metrik.csv (pycocotools, koordinat asli), keputusan_per_gambar.csv, provenance.json.
Analisis statistik lengkap tesis (bootstrap berpasangan, H1–H7, latensi ujung-ke-ujung): scripts/suite.py.
"""
import argparse
from pathlib import Path

import _path  # noqa: F401
import torch

from rfinal.data import load_data, make_loader
from rfinal.env import load_config, provenance, write_json
from rfinal.evaluate import metrics, per_image_table, run_split
from rfinal.gate import load_gate
from rfinal.model import load_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--backend", default="pytorch", choices=("pytorch", "ort", "trt"))
    ap.add_argument("--engine", default=None, help="ONNX A3 (ort) atau .engine (trt)")
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfg = load_config(a.config)
    assert a.backend == "pytorch" or a.engine, "--engine wajib untuk backend ort/trt"
    device = "cuda:0" if torch.cuda.is_available() and a.backend != "ort" else "cpu"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model, _ = load_detector(a.weights, device)
    gate = load_gate(a.gate)
    data = load_data(cfg["data"], cfg.get("expected_counts"))
    loader = make_loader(data, a.split, 1, int(cfg["train"]["imgsz"]), a.workers)
    R = run_split(model, gate, cfg["budget"], gate["thr0"], loader, device, cfg["gate"]["nms"], a.backend, a.engine)
    table = metrics(R, loader.dataset, data["names"], out / f"gt_{a.split}.json", n_random=a.n_random,
                    max_dets=tuple(cfg["eval"]["coco_max_dets"]))
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    table.to_csv(out / "metrik.csv", float_format="%.17g")
    per_image_table(R).to_csv(out / "keputusan_per_gambar.csv", index=False, float_format="%.17g")
    write_json(out / "provenance.json", provenance(files=[cfg["_config_path"], a.weights, a.gate, a.engine],
                                                   extra=dict(backend=a.backend, split=a.split)))
    print("✅", out)


if __name__ == "__main__":
    main()
