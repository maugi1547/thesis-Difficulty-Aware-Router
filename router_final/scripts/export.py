"""
Tahap 4 — ekspor engine A3 (satu graf ONNX dengan If) + verifikasi ONNX Runtime + (opsional) build TensorRT.

    python scripts/export.py --config configs/kitti.yaml --weights RUN/weights/last.pt --gate RUN/gate/gate.json \
        --out RUN/deploy [--trt]
"""
import argparse
from pathlib import Path

import _path  # noqa: F401
import torch

from rfinal.data import load_data, make_loader
from rfinal.env import load_config, provenance, sha256_file, write_json
from rfinal.export import export_a3, verify_a3
from rfinal.gate import load_gate
from rfinal.model import load_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", default=None, help="fp16 / fp32 (default: configs deploy.precision)")
    ap.add_argument("--trt", action="store_true", help="build engine TensorRT dari ONNX presisi terpilih")
    ap.add_argument("--n-verify", type=int, default=4)
    a = ap.parse_args()
    cfg = load_config(a.config)
    prec = a.precision or cfg["deploy"]["precision"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if prec == "fp16" and device == "cpu":
        raise SystemExit("FP16 butuh GPU (operator half CPU tidak lengkap); pakai --precision fp32 untuk uji lokal")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model, _ = load_detector(a.weights, "cpu")
    gate = load_gate(a.gate)
    data = load_data(cfg["data"], cfg.get("expected_counts"))
    loader = make_loader(data, "val", 1, int(cfg["train"]["imgsz"]), workers=0)
    imgs = []
    for i, b in enumerate(loader):
        if i >= a.n_verify:
            break
        imgs.append(b["img"].float() / 255.0)
    if prec == "fp16":
        model_dev = model.to(device)
        paths, mods = export_a3(model_dev, gate, imgs[0], out, "fp16", device, int(cfg["deploy"]["opset"]))
    else:
        paths, mods = export_a3(model, gate, imgs[0], out, "fp32", "cpu", int(cfg["deploy"]["opset"]))
    worst = verify_a3(paths["fp32"]["a3"], mods, imgs)
    print("verifikasi ONNX FP32 vs PyTorch:", worst)
    rec = dict(onnx={p: {k: str(v) for k, v in d.items()} for p, d in paths.items()},
               onnx_sha256={str(v): sha256_file(v) for d in paths.values() for v in d.values()},
               verify_fp32=worst, precision=prec)
    if a.trt:
        from rfinal.trt import build_engine
        rec["engine"] = build_engine(paths[prec]["a3"], out / f"a3_{prec}.engine", fp16_flag=prec == "fp16")
        rec["engine_sha256"] = sha256_file(out / f"a3_{prec}.engine")
        print("engine:", rec["engine"])
    write_json(out / "deploy.json", rec)
    write_json(out / "provenance.json", provenance(files=[cfg["_config_path"], a.weights, a.gate]))
    print("✅", out)


if __name__ == "__main__":
    main()
